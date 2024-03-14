import os, sys
import logging
from collections import Counter
import numpy as np

from .encoder_module.gvpstructure_embedding import GVPStructureEmbedding
from .codebook import ResidualCodebook

from .stack_attention import TransformerRotaryComplex, IPAattentionStackedDecoderComplex
from .folding_af2 import r3

from .nn_utils import make_low_resolution_mask, TransformerPositionEncoding, \
    distogram_loss, aatype_ce_loss, fape_loss_multichain, mask_loss,\
    get_coords_dict_from_affine
from .protein_utils.add_o_atoms import add_atom_O
from .protein_utils.write_pdb import write_multichain_from_atoms, fasta_writer

from .protein_utils.symmetry_loss import center_mass_loss_batch

import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore import Parameter
import mindspore.ops as ops


af2_restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
af2_aatype_to_index = {restype: i for i, restype in enumerate(af2_restypes)}
af2_index_to_aatype = {v:k for k, v in af2_aatype_to_index.items()}


def loss_dict_fp32(dict:dict):
    fp32_dict = {}
    for k, v in dict.items():
        fp32_dict[k] = v.float()

    return fp32_dict


def make_mask(lengths, batchsize, max_len, batch_dict, dtype):
    seq_mask = np.zeros((batchsize, max_len))
    pair_mask = np.zeros((batchsize, max_len, max_len))
    for idx in range(len(lengths)):
        length = lengths[idx]
        seq_mask[idx, :length] = np.ones((length))
        pair_mask[idx, :length, :length] = np.ones((length, length))

    batch_dict['affine_mask'] = seq_mask
    batch_dict['pair_mask'] = pair_mask
    batch_dict['single_mask'] = seq_mask

class VQStructure(nn.Cell):
    def __init__(self, config, global_config):
        super(VQStructure, self).__init__()
        self.config = config
        self.global_config = global_config
        self.down_sampling_scale = self.global_config.down_sampling_scale

        ### feature extraction
        if self.global_config.feature_encoder_type == 'GVPEncoder':
            self.encoder = GVPStructureEmbedding(
                self.config.gvp_embedding, self.global_config, self.down_sampling_scale)
        else:
            raise ValueError(f'{self.global_config.feature_encoder_type} unknown')

        ### feature transform eg. transformerrotary
        self.stacked_encoder = self.make_stackedtransformer(
            self.global_config.low_resolution_encoder_type, 
            self.config, single_in_dim=self.encoder.embed_dim, 
            layer_num=self.global_config.encoder_layer)

        ### binary codebook

        self.codebook = ResidualCodebook(
            self.config.residual_codebook, self.encoder.embed_dim, 
            codebook_num=self.config.residual_codebook.codebook_num,
            shared_codebook=self.config.residual_codebook.shared_codebook,
            codebook_dropout=self.config.residual_codebook.codebook_dropout)

        ### decoder
        self.single_res_embedding = TransformerPositionEncoding(
            global_config.max_seq_len, self.encoder.embed_dim)
        self.single_chain_embedding = nn.Embedding(
            global_config.max_chain_len, self.encoder.embed_dim)
        self.single_entity_embedding = nn.Embedding(
            global_config.max_entity_len, self.encoder.embed_dim)

        self.high_resolution_decoder = self.make_stackedtransformer(
            self.global_config.high_resolution_decoder_type, 
            self.config, self.encoder.embed_dim, self.global_config.single_rep_dim,
            layer_num=self.global_config.high_resolution_decoder_layer)
        self.decoder = self.high_resolution_decoder


        distogram_in_ch = self.decoder.pair_channel
        aatype_in_ch = self.global_config.single_rep_dim

        self.distogram_predictor = self.build_distogram_predictor(distogram_in_ch)
        if self.global_config.loss_weight.aatype_celoss > 0.0:
            self.aatype_predictor = self.build_aatype_predictor(aatype_in_ch)


    def make_stackedtransformer(self, decoder_type, config, single_in_dim, out_dim=None, layer_num=None):

        if decoder_type == 'TransformerRotary':
            stacked_decoder = TransformerRotaryComplex(
                config.transformerRotary, self.global_config, 
                single_in_dim, out_dim, layer_num= layer_num
                )
        elif decoder_type == 'IPAattention':
            stacked_decoder = IPAattentionStackedDecoderComplex(
                config.ipa_attention, self.global_config, single_in_dim, out_dim=out_dim)
        else:
            raise ValueError(f'{stacked_decoder} unknown')

        return stacked_decoder


    def build_distogram_predictor(self, pair_channel):
        out_num = self.config.distogram_pred.distogram_args[-1]
        distogram_predictor = nn.SequentialCell(
            nn.Dense(pair_channel, out_num),
            nn.ReLU(),
            nn.Dense(out_num, out_num))
        
        return distogram_predictor


    def build_aatype_predictor(self, single_channel):
        aatype_ce_head = nn.SequentialCell(
            nn.Dense(single_channel, single_channel),
            nn.ReLU(),
            nn.LayerNorm((single_channel,)),
            nn.Dense(single_channel, 20))

        return aatype_ce_head


    def construct(self, batch, return_all=False, use_codebook_num=4):
        dtype = batch['gt_pos'].dtype
        batchsize, L, N, _ = batch['gt_pos'].shape
        make_mask(batch['len'], batchsize, L, batch, dtype)
        
        codebook_mapping, codebook_indices, q_loss, q_loss_dict, encoded_feature, z_q_out = self.encode(batch, return_all, use_codebook_num)

        single_mask = Tensor(batch['single_mask'], ms.int32)
        single_res_rel = Tensor(batch['single_res_rel'], ms.int32)
        chain_idx = Tensor(batch['chain_idx'], ms.int32)
        entity_idx = Tensor(batch['entity_idx'], ms.int32)
        pair_res_idx = Tensor(batch['pair_res_idx'], ms.int32)
        pair_chain_idx = Tensor(batch['pair_chain_idx'], ms.int32)
        pair_same_entity = Tensor(batch['pair_same_entity'], ms.int32)

        reps = self.decode(codebook_mapping, single_mask, \
                single_res_rel, chain_idx, entity_idx,\
                pair_res_idx, pair_chain_idx, pair_same_entity)
        affine_p, single_rep, pair_rep_act = reps

        affine_p = affine_p[None].float()
        representations = {
            "single": single_rep,
            "pair": pair_rep_act
            }

        ## distogram loss
        pred_distogram = self.distogram_predictor(representations['pair'].float())
        gt_pos = Tensor(batch['gt_pos'], ms.float32)
        pair_mask = Tensor(batch['pair_mask'], ms.float32)
        dist_loss = distogram_loss(
            gt_pos, pred_distogram, self.config.distogram_pred, pair_mask)
        ## aa loss
        if self.global_config.loss_weight.aatype_celoss > 0.0:
            if self.global_config.decode_aatype_in_codebook:
                if self.global_config.additional_aatype_decoder_in_codebook:
                    pred_aatype = self.aatype_predictor(
                        codebook_mapping, batch['single_res_rel'], batch['single_mask'], batch=batch).float()
                else:
                    pred_aatype = self.aatype_predictor(codebook_mapping.float())
            elif self.global_config.decode_aatype_in_deocoder:
                pred_aatype = self.aatype_predictor(representations['single'].float())
            else:
                pred_aatype = self.aatype_predictor(encoded_feature.float())
            if self.training:
                if batch['aatype_mask'] is not None:
                    aa_mask = batch['aatype_mask']
                else:
                    aa_mask = batch['single_mask']
            else:
                aa_mask = batch['single_mask']
            aatype = Tensor(batch['aatype'], ms.int32)
            aa_mask = Tensor(aa_mask, ms.int32)
            aatype_loss, aatype_acc = aatype_ce_loss(aatype, pred_aatype, aa_mask)
        ## aa loss
        affine_0 = r3.rigids_to_quataffine_m(r3.rigids_from_tensor_flat12(Tensor(batch['gt_backbone_frame']).float())).to_tensor()[..., 0, :]

        losses_dict, fape_dict = fape_loss_multichain(
            affine_p, affine_0, Tensor(batch['gt_pos'][..., :3, :]), Tensor(batch['single_mask']), Tensor(batch['chain_idx']), self.global_config.fape)

        print("fape loss:", losses_dict)
        # q_loss, dist_loss, ceaatype loss in dict
        atom14_positions = fape_dict['coord'][-1]
        losses_dict.update({'q_loss': q_loss})
        losses_dict.update(q_loss_dict)
        losses_dict.update({'ditogram_classify_loss': dist_loss})

        if self.global_config.loss_weight.aatype_celoss > 0.0:
            losses_dict.update({'aatype_celoss': aatype_loss})
            losses_dict.update({'aatype_acc': aatype_acc})
        if self.global_config.loss_weight.mass_center_loss > 0.0:
            mass_center_loss = center_mass_loss_batch(
                atom14_positions, batch['gt_pos'][..., :3, :], 
                batch['single_mask'], batch['chain_idx'])
            losses_dict.update({'mass_center_loss': mass_center_loss})

        loss_mask = Tensor(batch['loss_mask'])
        losses_dict = mask_loss(loss_mask, losses_dict)

        loss = sum([losses_dict[k].mean() * 
                        self.global_config.loss_weight[k] \
                for k in self.global_config.loss_weight if k in losses_dict.keys()])
        losses_dict['loss'] = loss
        
        losses_dict = loss_dict_fp32(losses_dict)
        fape_dict = loss_dict_fp32(fape_dict)

        if return_all:
            all_dict = {
                "codebook_mapping": z_q_out,
                "codebook_indices": codebook_indices,
                "decoder_rep": representations,
                "pred_aatype": pred_aatype,
                "affine_p": affine_p,
                "coords_dict": fape_dict,
                "loss": losses_dict
            }
            return all_dict
        else:
            return losses_dict, codebook_indices


    def encode(self, batch, return_all_indices=False,use_codebook_num=4):
        encoded_feature = self.encoder(batch)
        encoder_single_mask = single_mask = batch['single_mask']
        batchsize, res_num, _ = encoded_feature.shape
        dtype = encoded_feature.dtype

        if self.down_sampling_scale > 1:
            encoder_single_mask = make_low_resolution_mask(single_mask, self.down_sampling_scale).to(dtype)
        if (batch.__contains__('chain_idx') and batch.__contains__('encode_split_chain')):
            between_chain_mask = np.where((batch['chain_idx'][:, None] - batch['chain_idx'][:, :, None]) == 0, 0., 1.)
            encode_split_chain = batch['encode_split_chain'][:, None, None] # 1. means split (mask), 0. means merge (visiable)
            pair_mask = between_chain_mask * encode_split_chain # 0. means visiable, 1. means mask
            pair_mask = 1. - pair_mask # 1. means visiable, 0. means mask
            pair_mask = Tensor(pair_mask, ms.float32)
            single_res_rel = Tensor(batch['single_res_rel'], ms.int32)
            encoder_single_mask = Tensor(encoder_single_mask, ms.float32)
            encoded_feature = self.stacked_encoder(
                encoded_feature, single_res_rel, single_mask=encoder_single_mask, pair_mask=pair_mask)
        else:
            encoded_feature = self.stacked_encoder(encoded_feature, single_mask=encoder_single_mask)
        codebook_mapping, codebook_indices, q_loss, q_loss_dict, stacked_z_q, z_q_out = self.codebook(encoded_feature, return_all_indices, use_codebook_num)
        q_loss_reduced = q_loss * encoder_single_mask
        q_loss_reduced = ops.sum(q_loss_reduced) / (ops.sum(encoder_single_mask) + 1e-6)

        q_loss_dict = {k: ops.sum(v * encoder_single_mask) / (ops.sum(encoder_single_mask) + 1e-6)\
                for k, v in q_loss_dict.items()}

        if self.codebook.head_num == 2:
            q_loss_dict['min_indices_num_1_count'] = ms.Tensor([len(Counter(list(codebook_indices[0].asnumpy())))])/batchsize
            q_loss_dict['min_indices_num_2_count'] = ms.Tensor([len(Counter(list(codebook_indices[1].asnumpy())))])/batchsize
        else:
            q_loss_dict['min_indices_num_count'] = ms.Tensor([len(Counter(list(codebook_indices[0].asnumpy())))])/batchsize
            
        return codebook_mapping, codebook_indices, q_loss_reduced.float(), q_loss_dict, encoded_feature, z_q_out


    def decode(self, single, single_mask, single_idx=None, chain_idx=None, entity_idx=None, pair_res_idx=None, pair_chain_idx=None, pair_same_entity=None):
        dtype = single.dtype
        if self.global_config.high_resolution_decoder_type != 'TransformerRotary':
            padding_mask = ~single_mask.bool()
            single_idx = single_idx * ~padding_mask + self.global_config.pad_num*padding_mask
            chain_idx = (chain_idx * ~padding_mask).long() + self.global_config.pad_chain_num*padding_mask
            entity_idx = (entity_idx * ~padding_mask).long() + self.global_config.pad_entity_num*padding_mask
            # single = single + self.position_embedding(single_idx, index_select=True).to(dtype)
            single = single + self.single_res_embedding(single_idx, index_select=True).to(dtype) + \
                self.single_chain_embedding(chain_idx).to(dtype) + \
                    self.single_entity_embedding(entity_idx).to(dtype)

        high_resolution_single = self.decoder(single, single_mask, pair_res_idx, pair_chain_idx, pair_same_entity)

        return high_resolution_single



    def sampling(self, batch, pdb_prefix, return_all=True, save_rep=False, verbose_indices=True, compute_sc_identity=True, compute_fape=False):
        pdbname_list = batch['pdbname']
        all_rep_dict = self(batch, return_all=True)
        
        batchsize, res_num = all_rep_dict['pred_aatype'].shape[:2]
        coords_dict = all_rep_dict['coords_dict']
        codebook_num = len(all_rep_dict['codebook_indices'])
        
        if return_all:
            for batch_idx in range(batchsize):
                os.makedirs(f'{pdb_prefix}/{pdbname_list[batch_idx]}', exist_ok=True)
                reduced_chain_idx = list(set(batch['chain_idx'][batch_idx].tolist()))
                gt_coord4 = []
                for chain_label in reduced_chain_idx:
                    chain_coords = batch['gt_pos'][batch_idx][batch['chain_idx'][batch_idx] == chain_label]
                    gt_coord4.append(add_atom_O(chain_coords[..., :3, :]).reshape(-1, 3))
                write_multichain_from_atoms(gt_coord4, 
                    f'{pdb_prefix}/{pdbname_list[batch_idx]}/{pdbname_list[batch_idx]}_vqrecon_gt.pdb', natom=4)

            # for debug
            if (all_rep_dict.__contains__('loss') and compute_fape):
                losses_dict = all_rep_dict['loss']
                intra_fape = losses_dict['intra_unclamp_fape_loss'].item()
                intra_clamp_fape = losses_dict['intra_clamp_fape_loss'].item()
                inter_fape = losses_dict['inter_unclamp_fape_loss'].item()
                inter_clamp_fape = losses_dict['inter_clamp_fape_loss'].item()
                print(f'intra-fape loss: {round(intra_fape, 3)}; intra-clamp fape: {round(intra_clamp_fape, 3)}')
                print(f'inter-fape loss: {round(inter_fape, 3)}; inter-clamp fape: {round(inter_clamp_fape, 3)}')

            if verbose_indices:
                if len(all_rep_dict['codebook_indices']) == 2:
                    indices_0_counter = Counter(all_rep_dict['codebook_indices'][0].tolist())
                    indices_1_counter = Counter(all_rep_dict['codebook_indices'][1].tolist())
                    indices_0_used = len(indices_0_counter)
                    indices_1_used = len(indices_1_counter)
                    mostcommon_10_used_indices_0 = indices_0_counter.most_common(10)
                    mostcommon_10_used_indices_1 = indices_1_counter.most_common(10)
                    print(f'codebook0: {round(indices_0_used/res_num, 3)} used; codebook1: {round(indices_1_used/res_num, 3)} used;')
                    print(f'codebook0 10 mostcommon: {mostcommon_10_used_indices_0}')
                    print(f'codebook1 10 mostcommon: {mostcommon_10_used_indices_1}')

            gt_aatype_af2idx = batch['aatype']
            pred_aatype = all_rep_dict['pred_aatype']

            for batch_idx in range(batchsize):
                traj_coord_0 = []
                for chain_label in reduced_chain_idx:
                    chain_coords = coords_dict['coord'][-1, batch_idx][Tensor(batch['chain_idx'][batch_idx] == chain_label)]
                    chain_coords = chain_coords.asnumpy()
                    traj_coord_0.append(
                        add_atom_O(chain_coords[..., :3, :]).reshape(-1, 3))
                write_multichain_from_atoms(traj_coord_0, 
                    f'{pdb_prefix}/{pdbname_list[batch_idx]}/{pdbname_list[batch_idx]}_vqrecon.pdb', natom=4)

                gt_aatype_str = []
                pred_aatype_str = []
                for chain_label in reduced_chain_idx:
                    chain_gt_aatype_str = ''.join([af2_index_to_aatype[aa] for aa in \
                        gt_aatype_af2idx[batch_idx][batch['chain_idx'][batch_idx] == chain_label].tolist()])
                    gt_aatype_str.append(chain_gt_aatype_str)
                    chain_pred_aatype = pred_aatype[batch_idx][Tensor(batch['chain_idx'][batch_idx] == chain_label)]
                    chain_pred_aatype_logits = ops.argmax(chain_pred_aatype, -1).reshape((-1, ))
                    chain_pred_aatype_str = ''.join([af2_index_to_aatype[int(aa.asnumpy())] for aa in chain_pred_aatype_logits])
                    pred_aatype_str.append(chain_pred_aatype_str)
                fasta_dict = {'native_seq': ':'.join(gt_aatype_str)}
                fasta_dict.update({f'predicted': ':'.join(pred_aatype_str)})
                fasta_writer(fasta_f=f'{pdb_prefix}/{pdbname_list[batch_idx]}/{pdbname_list[batch_idx]}_vqrecon.fasta', fasta_dict=fasta_dict)

        if save_rep:
            np.save(f'{pdb_prefix}_vqstructure_rep.npy', all_rep_dict)
            ident = (np.array(list(''.join(pred_aatype_str))) == np.array(list(''.join(gt_aatype_str)))).sum() / len(''.join(gt_aatype_str))
            return {
                'intra_fape': intra_fape, 'intra_clamp_fape': intra_clamp_fape, 
                'inter_fape': inter_fape, 'inter_clamp_fape': inter_clamp_fape, 'ident': ident
                }

        else:
            reshaped_indices = ops.stack([all_rep_dict['codebook_indices'][cb_idx].reshape(batchsize, res_num)\
                 for cb_idx in range(codebook_num)]).permute(1, 0, 2) # B, C, N

            return reshaped_indices
      
    def gen_structure_from_indices(
        self, 
        indices, 
        single_mask, 
        single_res_rel, 
        chain_idx, 
        entity_idx, 
        pair_res_idx, 
        pair_chain_idx, 
        pair_same_entity, 
        prefix='',
        save_dict=False
        ):
        assert len(indices.shape) == 3
        codeook_num, batchsize, res_num = indices.shape
        reduced_chain_idx = list(set(chain_idx[0].tolist()))

        pseudo_single_mask = single_mask
        pseudo_pair_mask = single_mask[:, None] * single_mask[:, :, None]
        pseudo_single_res_rel = np.arange(res_num).astype(np.int32)[None]

        pair_res_uplimit = self.global_config.pair_res_range[1]
        pair_res_mask_num = pair_res_uplimit + 1
        pair_res_rel = pseudo_single_res_rel[:, :, None] - pseudo_single_res_rel[:, None]
        pair_res_rel_idx = np.where(np.any(np.stack([pair_res_rel > pair_res_uplimit, 
                                pair_res_rel < -pair_res_uplimit]), 0), pair_res_mask_num, pair_res_rel)
        pseudo_pair_res_rel = pair_res_rel_idx.astype(np.int32) + pair_res_uplimit
        indices = Tensor(indices, dtype=ms.int32)
        codebook_mapping = self.codebook.get_feature_from_indices(indices)
        single_mask = Tensor(single_mask, dtype=ms.int32)
        single_res_rel = Tensor(single_res_rel, dtype=ms.int32)
        chain_idx = Tensor(chain_idx, dtype=ms.int32)
        entity_idx = Tensor(entity_idx, dtype=ms.int32)
        pair_res_idx = Tensor(pair_res_idx, dtype=ms.int32)
        pair_chain_idx = Tensor(pair_chain_idx, dtype=ms.int32)
        pair_same_entity = Tensor(pair_same_entity, dtype=ms.int32)

        reps = self.decode(
            codebook_mapping, single_mask, \
                single_res_rel, chain_idx, entity_idx,\
                pair_res_idx,pair_chain_idx, pair_same_entity)
        affine_p, single_rep, pair_rep_act = reps
        affine_p = affine_p[None]
        representations = {
            "single": single_rep,
            "pair": pair_rep_act
            }

        ## distogram loss
        pred_distogram = self.distogram_predictor(representations['pair'])
        ## aa loss
        if self.global_config.loss_weight.aatype_celoss > 0.0:
            pred_aatype = self.aatype_predictor(representations['single'])
        ## coords loss
        coords_dict = get_coords_dict_from_affine(affine_p)

        gen_dict = {
            'continous_feature': codebook_mapping,
            'coords_dict': coords_dict,
            'pred_distogram': pred_distogram,
            'pred_aatype': pred_aatype
        }
        
        if save_dict:
            np.save(f'{prefix}_gen_dict.npy', gen_dict)
        else:
            traj_coord_0 = []
            for chain_label in reduced_chain_idx:
                traj_coord_0.append(
                    add_atom_O(
                        coords_dict['coord'][-1, 0][chain_idx[0] == chain_label].asnumpy()[..., :3, :]).reshape(-1, 3)
                    )
            write_multichain_from_atoms(traj_coord_0, f'{prefix}_vqgen_from_indice.pdb', natom=4)

            pred_aatype_logits = ops.argmax(pred_aatype[-1], -1).reshape((-1, )).asnumpy()
            pred_aatype_str = ''.join([af2_index_to_aatype[aa] for aa in pred_aatype_logits])
            fasta_dict = {f'gen_seq': pred_aatype_str}
            fasta_writer(fasta_f=f'{prefix}_vqrecon.fasta', fasta_dict=fasta_dict)

        return gen_dict
                
