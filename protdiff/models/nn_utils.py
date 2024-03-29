import logging

import math
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms
from mindspore import Tensor


from .folding_af2 import quat_affine
from .protein_geom_utils import get_descrete_dist, get_internal_angles3


from .protein_utils.rigid import quat_to_rot


from .protein_utils.backbone import backbone_frame_to_atom3_std, backbone_fape_loss, backbone_fape_loss_multichain



NOISE_SCALE = 5000

logger = logging.getLogger(__name__)

class TransformerPositionEncoding(nn.Cell):
    def __init__(self, max_len, d_model):
        super(TransformerPositionEncoding, self).__init__()

        self.d_model = d_model
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)
        half_dim = d_model // 2
        ## emb.shape (hald_dim, )
        emb = np.exp(np.arange(half_dim) * -(math.log(10000) / half_dim))
        # Compute the positional encodings once in log space.
        pe[: ,: half_dim] = np.sin(position[:, None] * emb)
        pe[: ,half_dim: ] = np.cos(position[:, None] * emb)

        self.pe = pe

    def construct(self, timesteps, index_select=False):
        """
        return [:seqlen, d_model]
        """
        if not index_select:
            assert len(timesteps.shape) == 1
            return self.pe[:timesteps.shape[0]]
        else:
            B, L = timesteps.shape
            timesteps = timesteps.asnumpy()
            timesteps = timesteps.astype(np.int32)
            return Tensor(self.pe[timesteps.reshape(-1, 1)].reshape(B, L, self.d_model), ms.float32)


class ContinousNoiseSchedual(nn.Cell):
    """
    noise.shape (batch_size, )
    """
    def __init__(self, d_model):
        super(ContinousNoiseSchedual, self).__init__()

        half_dim = d_model // 2
        emb = math.log(10000) / float(half_dim - 1)
        emb = ops.exp(ops.arange(half_dim) * -emb)
        # emb.shape (half_dim, )
        self.register_buffer("emb", emb, persistent=True)

    def forward(self, noise):
        """
        noise [B, 1]
        return [:seqlen, d_model]
        """
        if len(noise.shape) > 1:
            noise = noise.squeeze(-1)
        assert len(noise.shape) == 1

        exponents = NOISE_SCALE * noise[:, None] * self.emb[None, :]
        return ops.cat([exponents.sin(), exponents.cos()], axis=-1)



def generate_new_affine(sequence_mask, return_frame=True):
    dtype = sequence_mask.dtype
    batch_size, num_residues = sequence_mask.shape[:2]
    quaternion = ms.Tensor([1., 0., 0., 0.]).to(dtype)
    quaternion = quaternion[None,None, :].tile((batch_size, num_residues, 1))

    translation = ops.zeros([batch_size, num_residues, 3]).to(dtype)
    affine = quat_affine.QuatAffine(quaternion, translation, unstack_inputs=True).to_tensor()
    if return_frame:
        return affine_to_frame12(affine)[:, :, None, :].to(dtype)
    else:
        return affine.to(dtype)


def get_batch_quataffine(pos):
    batchsize, nres, natoms, _ = pos.shape
    alanine_idx = residue_constants.restype_order_with_x["A"]
    aatype = Tensor([alanine_idx] * nres)[None].repeat(batchsize, 1)
    all_atom_positions = F.pad(pos, (0, 0, 0, 37-natoms, 0, 0), "constant", 0)
    all_atom_mask = ops.ones(batchsize, nres, 37)
    frame_dict = atom37_to_frames(aatype, all_atom_positions, all_atom_mask)

    return frame_dict['rigidgroups_gt_frames']


def make_mask(lengths, batchsize, max_len, batch_dict, dtype=np.float32):
    batch_dict['aatype']
    seq_mask = np.zeros((batchsize, max_len))
    pair_mask = np.zeros((batchsize, max_len, max_len))
    # import pdb; pdb.set_trace()
    for idx in range(len(lengths)):
        length = lengths[idx]
        seq_mask[idx, :length] = np.ones((length,))
        pair_mask[idx, :length, :length] = np.ones((length, length))

    batch_dict['affine_mask'] = seq_mask.astype(dtype)
    batch_dict['pair_mask'] = pair_mask.astype(dtype)
    batch_dict['single_mask'] = seq_mask.astype(dtype)



def downsampling_single_idx(single_idx, downsampling_scale):
    batchsize, res_num = single_idx.shape

    low_resolution_select_idx = ops.arange(0, res_num, downsampling_scale).float()
    low_resolution_idx = single_idx[:, low_resolution_select_idx]

    return low_resolution_idx


def downsampling_pair_idx(pair_idx, downsampling_scale):
    batchsize, res_num = pair_idx.shape[:2]

    low_resolution_select_idx = ops.arange(0, res_num, downsampling_scale).float()
    low_resolution_pair_idx = pair_idx[:, low_resolution_select_idx][:, :, low_resolution_select_idx]

    return low_resolution_pair_idx


def make_low_resolution_mask(single_mask, down_sampling_scale):
    batchsize, res_num = single_mask.shape
    low_resolution_select_idx = np.arange(0, res_num, down_sampling_scale)
    expanded_single_mask = single_mask.reshape(batchsize, len(low_resolution_select_idx), -1)

    single_mask = np.any(expanded_single_mask, -1)
    return single_mask



def make_pairidx_from_singleidx(single_idx, pair_res_range):
    pair_res_idx = single_idx[:, None] - single_idx
    pair_res_idx = np.where(np.any(np.stack([pair_res_idx > pair_res_range[1], 
            pair_res_idx < pair_res_range[0]]), 0), pair_res_range[1]+1, pair_res_idx)
    pair_res_idx = pair_res_idx - pair_res_range[0]

    return pair_res_idx



def fape_loss(
    affine_p, 
    affine_0, 
    coord_0, 
    mask, 
    fape_config, 
    cond=None, 
    ):
    affine_p = affine_p.float()
    affine_0 = affine_0.float()
    coord_0 = coord_0.float()
    mask = mask.float()

    quat_0 = affine_0[..., :4]
    trans_0 = affine_0[..., 4:]
    rot_0 = quat_to_rot(quat_0)

    batch_size, num_res = affine_0.shape[:2]

    clamp_distance = fape_config.clamp_distance
    loss_unit_distance = fape_config.loss_unit_distance
    clamp_weight = fape_config.clamp_weight
    traj_weight = fape_config.traj_weight

    rot_list, trans_list, coord_list = [], [], []
    num_ouputs = affine_p.shape[0]
    loss_unclamp, loss_clamp = [], []
    for i in range(num_ouputs):
        quat = affine_p[i, ..., :4]
        trans = affine_p[i, ..., 4:]
        rot = quat_to_rot(quat)
        coord = backbone_frame_to_atom3_std(
            ops.reshape(rot, (-1, 3, 3)),
            ops.reshape(trans, (-1, 3)),
        )
        # import pdb; pdb.set_trace()
        coord = ops.reshape(coord, (batch_size, num_res, 3, 3))
        coord_list.append(coord)
        rot_list.append(rot)
        trans_list.append(trans)

        if cond is None:
            mask_2d = mask[..., None] * mask[..., None, :]
            affine_p_ = affine_p[i]
            coord_p_ = coord
        else:
            mask_2d = 1 - (cond[..., None] * cond[..., None, :])
            mask_2d = mask_2d * (mask[..., None] * mask[..., None, :])   
            # import pdb; pdb.set_trace()
            affine_p_ = ops.where(cond[..., None] == 1, affine_0, affine_p[i])
            coord_p_ = ops.where(cond[..., None, None] == 1, coord_0[:, :, :3], coord)

        quat_p_ = affine_p_[..., :4]
        trans_p_ = affine_p_[..., 4:]
        rot_p_ = quat_to_rot(quat_p_)

        fape, fape_clamp = backbone_fape_loss(
            coord_p_, rot_p_, trans_p_,
            coord_0, rot_0, trans_0, mask,
            clamp_dist=clamp_distance,
            length_scale=loss_unit_distance,
            mask_2d=mask_2d
        )
        loss_unclamp.append(fape)
        loss_clamp.append(fape_clamp)
    
    loss_unclamp = ops.stack(loss_unclamp)
    loss_clamp = ops.stack(loss_clamp)
    loss = loss_unclamp * (1.0 - clamp_weight) + loss_clamp * clamp_weight
    
    last_loss = loss[-1]
    traj_loss = loss.mean()
    # import pdb; pdb.set_trace()
    if num_ouputs != 1:
        loss = last_loss + traj_weight * traj_loss
    else:
        loss = last_loss

    losses = {
        'fape_loss': loss,
        'clamp_fape_loss': loss_clamp[-1],
        'unclamp_fape_loss': loss_unclamp[-1],
        'last_loss': last_loss,
        'traj_loss': traj_loss,
    }
    coord_dict = {
        'coord': ops.stack(coord_list),
        'rot': ops.stack(rot_list),
        'trans': ops.stack(trans_list)
    }
    return losses, coord_dict



def distogram_loss(gt_pos, pred_maps_descrete, distogram_pred_config, pair_mask):
    batchsize, nres = pred_maps_descrete.shape[:2]
    distogram_list = []
    dist_type_name = ['ca-ca']

    for dist_type_idx, dist_type in enumerate(dist_type_name):
        gt_map_descrete = get_descrete_dist(
            gt_pos, dist_type, distogram_pred_config.distogram_args)
        dim_start = (dist_type_idx) * distogram_pred_config.distogram_args[-1]
        dim_end = (dist_type_idx + 1) * distogram_pred_config.distogram_args[-1]
        pred_map = pred_maps_descrete[..., dim_start: dim_end]
        pred_map = pred_map.reshape(-1, distogram_pred_config.distogram_args[-1])
        gt_map_descrete = gt_map_descrete.reshape(-1).astype(ms.int32)
        distogram_loss = ops.cross_entropy(pred_map, gt_map_descrete, reduction='none').reshape(batchsize, nres, nres)
        distogram_list.append(distogram_loss)
    
    distogram_loss = ops.stack(distogram_list).mean(0)
    distogram_loss = distogram_loss * pair_mask
    distogram_loss_reduce = ops.sum(distogram_loss) / (ops.sum(pair_mask) + 1e-6)
    return distogram_loss_reduce


def aatype_ce_loss(gt_aatype, pred_aatype, single_mask):
    batchsize, num_res = pred_aatype.shape[:2]
    aatype_loss = ops.cross_entropy(pred_aatype.reshape(-1, 20), gt_aatype.reshape(-1), reduction='none').reshape(batchsize, num_res)
    aatype_loss = aatype_loss * single_mask
    aatype_loss_reduce = ops.sum(aatype_loss) / (ops.sum(single_mask) + 1e-6)

    pred = ops.argmax(pred_aatype, dim=-1)
    acc = ops.sum((pred == gt_aatype).float() * single_mask) / (ops.sum(single_mask) + 1e-6)
    return aatype_loss_reduce, acc


def dihedral_loss(pred_coords, gt_coords, single_mask):
    pred_dihedral = get_internal_angles3(pred_coords.float())
    pred_dihedral_feature = ops.cat((ops.cos(pred_dihedral), ops.sin(pred_dihedral)), 2)
    gt_dihedral = get_internal_angles3(gt_coords.float())
    gt_dihedral_feature = ops.cat((ops.cos(gt_dihedral), ops.sin(gt_dihedral)), 2)
    diff_dihedral = ops.mse_loss(pred_dihedral_feature,  gt_dihedral_feature, reduction='none').mean(-1)
    diff_dihedral = diff_dihedral * single_mask
    dihedral_mse_reduce = ops.sum(diff_dihedral) / (ops.sum(single_mask) + 1e-6)
    return dihedral_mse_reduce


def get_coords_dict_from_affine(affine):
    coord_list, rot_list, trans_list = [], [], []
    num_ouputs, batch_size, num_res = affine.shape[:3]

    for i in range(num_ouputs):
        quat = affine[i, ..., :4]
        trans = affine[i, ..., 4:]
        rot = quat_to_rot(quat)
        coord = backbone_frame_to_atom3_std(
            ops.reshape(rot, (-1, 3, 3)),
            ops.reshape(trans, (-1, 3)),
        )
        # import pdb; pdb.set_trace()
        coord = ops.reshape(coord, (batch_size, num_res, 3, 3))
        coord_list.append(coord)
        rot_list.append(rot),
        trans_list.append(trans)


    coord_dict = {
        'coord': ops.stack(coord_list),
        'rot': ops.stack(rot_list),
        'trans': ops.stack(trans_list)
    }
    return coord_dict
def fape_loss_multichain(
    affine_p, 
    affine_0, 
    coord_0, 
    mask, 
    chain_idx,
    fape_config, 
    cond=None, 
    ):
    affine_p = affine_p.float()
    affine_0 = affine_0.float()
    coord_0 = coord_0.float()
    mask = mask.float()

    quat_0 = affine_0[..., :4]
    trans_0 = affine_0[..., 4:]
    rot_0 = quat_to_rot(quat_0)

    batch_size, num_res = affine_0.shape[:2]

    intra_clamp_distance = fape_config.intra_clamp_distance
    intra_loss_unit_distance = fape_config.intra_loss_unit_distance
    inter_clamp_distance = fape_config.inter_clamp_distance
    inter_loss_unit_distance = fape_config.inter_loss_unit_distance
    clamp_weight = fape_config.clamp_weight
    traj_weight = fape_config.traj_weight

    rot_list, trans_list, coord_list = [], [], []
    num_ouputs = affine_p.shape[0]
    intra_loss_unclamp, intra_loss_clamp = [], []
    inter_loss_unclamp, inter_loss_clamp = [], []

    for i in range(num_ouputs):
        quat = affine_p[i, ..., :4]
        trans = affine_p[i, ..., 4:]
        rot = quat_to_rot(quat)
        coord = backbone_frame_to_atom3_std(
            ops.reshape(rot, (-1, 3, 3)),
            ops.reshape(trans, (-1, 3)),
        )

        coord = ops.reshape(coord, (batch_size, num_res, 3, 3))
        coord_list.append(coord)
        rot_list.append(rot),
        trans_list.append(trans)

        if cond is None:
            mask_2d = mask[..., None] * mask[..., None, :]
            affine_p_ = affine_p[i]
            coord_p_ = coord

        quat_p_ = affine_p_[..., :4]
        trans_p_ = affine_p_[..., 4:]
        rot_p_ = quat_to_rot(quat_p_)

        fape_loss_dict = backbone_fape_loss_multichain(
            coord_p_, rot_p_, trans_p_,
            coord_0, rot_0, trans_0, mask,
            chain_idx=chain_idx,
            intra_clamp_dist=intra_clamp_distance,
            intra_length_scale=intra_loss_unit_distance,
            inter_clamp_dist=inter_clamp_distance,
            inter_length_scale=inter_loss_unit_distance,
            mask_2d=mask_2d
        )
        intra_loss_unclamp.append(fape_loss_dict['intra_loss'])
        intra_loss_clamp.append(fape_loss_dict['intra_loss_clamp'])
        inter_loss_unclamp.append(fape_loss_dict['inter_loss'])
        inter_loss_clamp.append(fape_loss_dict['inter_loss_clamp'])
    
    intra_loss_unclamp = ops.stack(intra_loss_unclamp)
    intra_loss_clamp = ops.stack(intra_loss_clamp)
    inter_loss_unclamp = ops.stack(inter_loss_unclamp)
    inter_loss_clamp = ops.stack(inter_loss_clamp)

    intra_loss = intra_loss_unclamp * (1.0 - clamp_weight) + intra_loss_clamp * clamp_weight
    inter_loss = inter_loss_unclamp * (1.0 - clamp_weight) + inter_loss_clamp * clamp_weight

    loss = intra_loss + inter_loss
    last_loss = loss[-1]
    traj_loss = loss.mean()
    if num_ouputs != 1:
        loss = last_loss + traj_weight * traj_loss
    else:
        loss = last_loss

    losses = {
        'intra_fape_loss': intra_loss,
        'intra_clamp_fape_loss': intra_loss_clamp[-1],
        'intra_unclamp_fape_loss': intra_loss_unclamp[-1],
        'inter_fape_loss': inter_loss,
        'inter_clamp_fape_loss': inter_loss_clamp[-1],
        'inter_unclamp_fape_loss': inter_loss_unclamp[-1],
        'fape_loss': loss,
        'last_loss': last_loss,
        'traj_loss': traj_loss,
    }
    coord_dict = {
        'coord': ops.stack(coord_list),
        'rot': ops.stack(rot_list),
        'trans': ops.stack(trans_list)
    }
    return losses, coord_dict


def mask_loss(loss_masks, loss_dict):
    """ items need to be mask is filled with True, else is False
    """
    assert isinstance(loss_masks, Tensor)
    loss_masks = 1 - loss_masks.int()
    
    def _apply(x):
        # if isinstance(loss_masks, Tensor):
        #     len_shape = len(x.shape)
        #     [loss_masks.unsqueeze_(1) for _ in range(len_shape-1)]
        #     return loss_masks * x
        if isinstance(x, dict):
            return {key: _apply(x[key]) for key in x.keys()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x
    
    return _apply(loss_dict)
