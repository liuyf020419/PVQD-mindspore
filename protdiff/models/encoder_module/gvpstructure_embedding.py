import numpy as np
import math

from ..esm.util import get_rotation_frames, nan_to_num, rotate
from ..esm.features import GVPGraphEmbedding, DihedralFeatures, GVPInputFeaturizer
from ..esm.gvp_modules import GVPConvLayer
from ..esm.gvp_utils import unflatten_graph

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class GVPStructureEmbedding(nn.Cell):
    def __init__(self, config, global_config, conitnious_res_num=1) -> None:
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.conitnious_res_num = conitnious_res_num
        self.pad_res_num = self.conitnious_res_num//2

        self.embed_dim = embed_dim = self.config.encoder_embed_dim
        self.embed_scale = math.sqrt(embed_dim)

        if self.conitnious_res_num == 1:
            self.embed_gvp_input_features = nn.Dense(6 + 9, embed_dim)
        else:
            self.embed_gvp_input_features = nn.Dense((6 + 9) * self.conitnious_res_num, embed_dim)
        self.embed_confidence = nn.Dense(16, embed_dim)
        gvp_args = self.config.gvp
        self.gvp_encoder = GVPEncoder(gvp_args)

        self.embed_dihedrals = DihedralFeatures(embed_dim, self.conitnious_res_num)
        self.aatype_embedding = nn.Embedding(22, embed_dim)
        if self.conitnious_res_num > 1:
            self.aatype_embedding_continuous = nn.Dense(embed_dim * self.conitnious_res_num, embed_dim)

        gvp_out_dim = (gvp_args.node_hidden_dim_scalar + (3 *
                gvp_args.node_hidden_dim_vector)) * self.conitnious_res_num
        self.embed_gvp_output = nn.Dense(gvp_out_dim, embed_dim)


    def construct(self, batch):
        coord_dict = {
            'coord': batch['gt_pos'].astype(np.float32),
            'backbone_frame': batch['gt_backbone_frame'].astype(np.float32)
        }
        data_dict = self.make_data_dict(coord_dict, batch['single_mask'], batch['single_res_rel'])

        coords = data_dict['coord'][..., :3, :]
        batchsize, res_num = coords.shape[:2]

        padding_mask = ~batch['single_mask'].astype(np.bool)
        confidence = data_dict['confidence']
        single_mask_shape = batch['single_mask'].shape
        if (self.global_config.aatype_embedding and self.global_config.aatype_embedding_in_encoder):
            if not self.training:
                aatype_drop_p = 0.0
                aatype_replace_p = 0.0
                aatype_maintain_p = 0.0
            else:
                aatype_drop_p = self.global_config.aatype_drop_p
                aatype_replace_p = self.global_config.aatype_replace_p
                aatype_maintain_p = self.global_config.aatype_maintain_p

            aatype_mask = (np.rand(single_mask_shape) > aatype_drop_p).astype(np.float32) * batch['single_mask']
            batch['aatype_mask'] = (1 - aatype_mask) * batch['single_mask']
            input_aatype = batch['aatype']
            aatype_replace_unmask = 1 - ( (np.rand(single_mask_shape) * (1 - aatype_mask)) > aatype_replace_p).astype(np.float32) * batch['single_mask']
            aatype_maintain_replace_unmask = 1 - ((np.rand(single_mask_shape) * (1 - aatype_mask) * (1 - aatype_replace_unmask) ) > aatype_maintain_p).float() * batch['single_mask']

            aatype_replace_only = (aatype_replace_unmask - aatype_mask) * batch['single_mask']
            aatype_maintain_only = (aatype_maintain_replace_unmask - aatype_replace_unmask) * batch['single_mask']

            aatype = (input_aatype * aatype_mask + (1 - aatype_mask) * 21).long()
            aatype = np.where(aatype_replace_only.astype(np.bool), np.random.randint(0, 20, aatype.shape), aatype)
            aatype = np.where(aatype_maintain_only.astype(np.bool), input_aatype, aatype)

        else:
            aatype = np.zeros(single_mask_shape)
            batch['aatype_mask'] = None

        # R = data_dict['rot']
        coords = Tensor(coords, ms.float32)
        R = get_rotation_frames(coords)

        res_idx = data_dict['res_idx']
        res_idx = res_idx * ~padding_mask + self.global_config.pad_num*padding_mask

        coord_mask = ops.all(ops.all(ops.isfinite(coords), axis=-1), axis=-1)
        coords = nan_to_num(coords)

        # GVP encoder out
        res_idx = Tensor(res_idx, ms.int32)
        padding_mask = Tensor(padding_mask, ms.int32)
        confidence = Tensor(confidence, ms.float32)
        aatype = Tensor(aatype, ms.int32)
        
        gvp_out_scalars, gvp_out_vectors = self.gvp_encoder(coords,
                coord_mask, res_idx, padding_mask, confidence)

        if self.conitnious_res_num == 1:
            gvp_out_features = ops.cat([
                gvp_out_scalars,
                ops.flatten(rotate(gvp_out_vectors, R.swapaxes(-2, -1)), start_dim=-2, end_dim=-1),
            ], axis=-1)
            # import pdb; pdb.set_trace()
        else:
            # # # BxLxD
            low_resolution_select_idx = ops.arange(0, res_num, self.conitnious_res_num)
            expanded_gvp_out_scalars = gvp_out_scalars.reshape(batchsize, len(low_resolution_select_idx), -1)

            # BxLxLxNx3
            rotated_vector_mat = ops.sum(gvp_out_vectors[:, None, ..., None] * R[:, :, None, ..., None, :, :], -2)
            # BxLxCxNx3 -> LxBxCxNx3
            expanded_gvp_out_vectors = ops.stack(
                [rotated_vector_mat[:, first_v_idx, first_v_idx + ops.arange(self.conitnious_res_num).to(device)] \
                    for first_v_idx in low_resolution_select_idx],1)
            expanded_gvp_out_vectors = expanded_gvp_out_vectors.reshape(batchsize, len(low_resolution_select_idx), -1)

            gvp_out_features = ops.cat([
                expanded_gvp_out_scalars,expanded_gvp_out_vectors], axis=-1)

        components = dict()

        # raw feature seq 0 9|while read line; do /bin/rm -rf checkpoint_$line*; done
        components["diherals"] = self.embed_dihedrals(coords)

        components["gvp_out"] = self.embed_gvp_output(gvp_out_features)
        # components["confidence"] = self.embed_confidence(rbf(confidence, 0., 1.))
        scalar_features, vector_features = GVPInputFeaturizer.get_node_features(
            coords, coord_mask, with_coord_mask=False)
        
        if self.conitnious_res_num > 1:
            components["tokens"] = self.aatype_embedding_continuous(
                (self.aatype_embedding(aatype) * self.embed_scale).reshape(
                    batchsize, len(low_resolution_select_idx), -1))
        else:
            components["tokens"] = self.aatype_embedding(aatype) * self.embed_scale

        features = ops.cat([
            scalar_features,
            ops.flatten(rotate(vector_features, R.swapaxes(-2, -1)), start_dim=-2, end_dim=-1),
        ], axis=-1)


        components["gvp_input_features"] = self.embed_gvp_input_features(features)

        embed = sum(components.values()).astype(ms.float32)
        return embed


    def make_data_dict(self, coord_dict: dict, seq_mask_traj, res_idx):
        batchsize, L = coord_dict['coord'].shape[:2]
        coord = coord_dict['coord'][..., :3, :]
        if not coord_dict.__contains__('rot'):
            new_shape = list(coord_dict['backbone_frame'].shape[:-2]) + [3, 3]
            rot = coord_dict['backbone_frame'][..., 0, :9].reshape(new_shape)
            rot = rot.reshape(batchsize, L, 3, 3)
        else:
            rot = coord_dict['rot']
        pseudo_aatype = np.zeros((batchsize, L))
        data_dict = {'coord': coord, 'encoder_padding_mask': seq_mask_traj.astype(np.bool), 
                    'confidence': np.ones((batchsize, L)), 'rot': rot, 
                    'res_idx': res_idx,
                    'aatype': pseudo_aatype}
        
        return data_dict


class GVPStructureEmbeddingV2(nn.Cell):
    def __init__(self, config, global_config, conitnious_res_num=1) -> None:
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.conitnious_res_num = conitnious_res_num
        self.pad_res_num = self.conitnious_res_num//2

        self.embed_dim = embed_dim = self.config.encoder_embed_dim
        self.embed_scale = math.sqrt(embed_dim)

        if self.conitnious_res_num == 1:
            self.embed_gvp_input_features = nn.Dense(6 + 9, embed_dim)
        else:
            self.embed_gvp_input_features = nn.Dense((6 + 9) * self.conitnious_res_num, embed_dim)
        self.embed_confidence = nn.Dense(16, embed_dim)
        gvp_args = self.config.gvp
        self.gvp_encoder = GVPEncoder(gvp_args)

        self.embed_dihedrals = DihedralFeatures(embed_dim, self.conitnious_res_num)
        self.aatype_embedding = nn.Embedding(22, embed_dim)
        if self.conitnious_res_num > 1:
            self.aatype_embedding_continuous = nn.Dense(embed_dim * self.conitnious_res_num, embed_dim)

        gvp_out_dim = (gvp_args.node_hidden_dim_scalar + (3 *
                gvp_args.node_hidden_dim_vector)) * self.conitnious_res_num
        self.embed_gvp_output = nn.Dense(gvp_out_dim, embed_dim)


    def construct(self, batch, condition_mask):
        dtype = batch['gt_backbone_pos'].dtype
        padding_mask = ~batch['single_mask'].astype(np.bool_)
        gvp_mask = ~condition_mask.astype(np.bool_)

        coord_dict = {
            'coord': batch['gt_backbone_pos'].astype(np.float32) * condition_mask[..., None, None],
            'backbone_frame': batch['gt_backbone_frame'].astype(np.float32) * condition_mask[..., None, None]
        }
        data_dict = self.make_data_dict(coord_dict, condition_mask, batch['single_res_rel'])

        backbone_coords = batch['gt_backbone_pos'][..., :3, :].astype(np.float32) * condition_mask[..., None, None]

        confidence = data_dict['confidence']

        # GVP encoder out
        aatype = np.zeros(condition_mask.shape)
        batch['aatype_mask'] = None

        backbone_coords = Tensor(backbone_coords, ms.float32)
        R = get_rotation_frames(backbone_coords)
        res_idx = data_dict['res_idx']
        res_idx = res_idx * ~padding_mask + self.global_config.pad_num*padding_mask
        
        
        condition_mask = Tensor(condition_mask, ms.float32)
        coord_mask = ops.all(ops.all(ops.isfinite(backbone_coords), axis=-1), axis=-1)
        coord_mask = (coord_mask * condition_mask).bool()
        coords = nan_to_num(backbone_coords)


        # GVP encoder out
        res_idx = Tensor(res_idx, ms.int32)
        gvp_mask = Tensor(gvp_mask, ms.int32)
        confidence = Tensor(confidence, ms.float32)
        aatype = Tensor(aatype, ms.int32)
    
        gvp_out_scalars, gvp_out_vectors = self.gvp_encoder(coords,
                coord_mask, res_idx, gvp_mask, confidence)

        gvp_out_features = ops.cat([
            gvp_out_scalars,
            ops.flatten(rotate(gvp_out_vectors, R.swapaxes(-2, -1)), start_dim=-2, end_dim=-1),
        ], axis=-1)

        components = dict()

        # raw feature seq 0 9|while read line; do /bin/rm -rf checkpoint_$line*; done
        components["diherals"] = self.embed_dihedrals(backbone_coords)

        components["gvp_out"] = self.embed_gvp_output(gvp_out_features)
        # components["confidence"] = self.embed_confidence(rbf(confidence, 0., 1.))
        scalar_features, vector_features = GVPInputFeaturizer.get_node_features(
            coords, coord_mask, with_coord_mask=False)


        components["tokens"] = self.aatype_embedding(aatype) * self.embed_scale * condition_mask[..., None]

        features = ops.cat([
            scalar_features,
            ops.flatten(rotate(vector_features, R.swapaxes(-2, -1)), start_dim=-2, end_dim=-1),
        ], axis=-1)

        features = features * condition_mask[..., None]
        components["gvp_input_features"] = self.embed_gvp_input_features(features)

        embed = sum(components.values()).astype(ms.float32)
        return embed


    def make_data_dict(self, coord_dict: dict, seq_mask_traj, res_idx):
        batchsize, L = coord_dict['coord'].shape[:2]
        coord = coord_dict['coord'][..., :3, :]
        if not coord_dict.__contains__('rot'):
            new_shape = list(coord_dict['backbone_frame'].shape[:-2]) + [3, 3]
            rot = coord_dict['backbone_frame'][..., 0, :9].reshape(new_shape)
            rot = rot.reshape(batchsize, L, 3, 3)
        else:
            rot = coord_dict['rot']
        pseudo_aatype = np.zeros((batchsize, L))
        data_dict = {'coord': coord, 'encoder_padding_mask': seq_mask_traj.astype(np.bool), 
                    'confidence': np.ones((batchsize, L)), 'rot': rot, 
                    'res_idx': res_idx,
                    'aatype': pseudo_aatype}
        
        return data_dict


class GVPEncoder(nn.Cell):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_graph = GVPGraphEmbedding(args)

        node_hidden_dim = (args.node_hidden_dim_scalar,
                args.node_hidden_dim_vector)
        edge_hidden_dim = (args.edge_hidden_dim_scalar,
                args.edge_hidden_dim_vector)
        
        conv_activations = (ops.ReLU(), ops.Sigmoid())
        self.encoder_layers = nn.CellList([
                GVPConvLayer(
                    node_hidden_dim,
                    edge_hidden_dim,
                    drop_rate=args.dropout,
                    vector_gate=True,
                    attention_heads=0,
                    n_message=3,
                    conv_activations=conv_activations,
                    n_edge_gvps=0,
                    eps=1e-4,
                    layernorm=True,
                ) 
            for i in range(args.num_encoder_layers)]
        )

    def construct(self, coords, coord_mask, res_idx, padding_mask, confidence):
        node_embeddings, edge_embeddings, edge_index = self.embed_graph(
                coords, coord_mask, res_idx, padding_mask, confidence)
        for i, layer in enumerate(self.encoder_layers):
            node_embeddings, edge_embeddings = layer(node_embeddings,
                    edge_index, edge_embeddings)

        node_embeddings = unflatten_graph(node_embeddings, coords.shape[0])
        return node_embeddings
