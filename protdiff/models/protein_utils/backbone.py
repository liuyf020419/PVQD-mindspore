import mindspore.ops as ops
import mindspore as ms
from mindspore import Tensor


STD_RIGID_COORD = Tensor(
    [[-0.525,  1.363,  0.000],
     [ 0.000,  0.000,  0.000],
     [ 1.526, -0.000, -0.000],
     [ 0.627,  1.062,  0.000]], ms.float32
)



def backbone_frame_to_atom3_std(rot, trans, atomnum=3):
    assert (atomnum == 3) or (atomnum == 4)
    std_coord = STD_RIGID_COORD[:atomnum]
    atom3 = ops.bmm(std_coord[None].tile((trans.shape[0], 1, 1)), ops.swapaxes(rot, -1,-2)) + trans.unsqueeze(-2)
    return atom3



def convert_to_local(
        coord,  # [B, L, N, 3]
        rot,    # [B, L, 3，3]
        trans   # [B, L, 3]
    ):
    batch_size, num_res, num_atoms = coord.shape[:3]
    coord_expand = ops.tile(coord[:, None], (1, num_res, 1, 1, 1))
    trans_expand = ops.tile(trans[:, :, None], (1, 1, num_res, 1))
    coord_expand = coord_expand - trans_expand.unsqueeze(-2)

    inv_rot = ops.swapaxes(rot, -1, -2)
    rot_expand = ops.tile(inv_rot[:, :, None], (1, 1, num_res, 1, 1))
    
    coord_flat = ops.reshape(coord_expand, (-1, num_atoms, 3))
    rot_flat = ops.reshape(rot_expand, (-1, 3, 3))

    local_coord = ops.bmm(coord_flat, rot_flat.swapaxes(-1, -2))
    local_coord = ops.reshape(local_coord, (batch_size, num_res, num_res, num_atoms, 3))
    return local_coord


def backbone_fape_loss(
        pred_coord,     # [B, L, N, 3]
        pred_rot,       # [B, L, 3, 3]
        pred_trans,     # [B, L, 3]
        ref_coord,      # [B, L, N, 3]
        ref_rot,        # [B, L, 3, 3]
        ref_trans,      # [B, L, 3]
        mask,           # [B, L]
        clamp_dist = 10.0,
        length_scale = 1.0,
        mask_2d = None,
        return_nosum = False
    ):

    pred_coord_local = convert_to_local(pred_coord, pred_rot, pred_trans)   # [B, L, L, N, 3]
    ref_coord_local = convert_to_local(ref_coord, ref_rot, ref_trans)       # [B, L, L, N, 3]

    if mask_2d is None:
        mask2d = mask[..., None] * mask[..., None, :]
    else:
        mask2d = mask_2d
    dist_map = ops.sqrt(ops.sum((pred_coord_local - ref_coord_local) ** 2, -1) + 1e-6)
    dist_map = ops.mean(dist_map, -1)
    dist_map_clamp = dist_map.clamp(max = clamp_dist)

    dist_map = dist_map / length_scale
    dist_map_clamp = dist_map_clamp / length_scale

    loss = ops.sum(dist_map * mask2d) / (ops.sum(mask2d) + 1e-6)
    loss_clamp = ops.sum(dist_map_clamp * mask2d) / (ops.sum(mask2d) + 1e-6)
    if return_nosum:
        return loss, loss_clamp, dist_map * mask2d
    else:
        return loss, loss_clamp


def backbone_fape_loss_multichain(
        pred_coord,     # [B, L, N, 3]
        pred_rot,       # [B, L, 3, 3]
        pred_trans,     # [B, L, 3]
        ref_coord,      # [B, L, N, 3]
        ref_rot,        # [B, L, 3, 3]
        ref_trans,      # [B, L, 3]
        mask,           # [B, L]
        chain_idx,      # [B, L]
        intra_clamp_dist = 10.0,
        intra_length_scale = 10.0,
        inter_clamp_dist = 30.0,
        inter_length_scale = 20.0,
        mask_2d = None, 
    ):
    monomer_inter_mask = ops.any(chain_idx > 0, 1)
    intra_pair_mask = ops.where((chain_idx[:, None] - chain_idx[:, :, None]) == 0, ms.Tensor(1.), ms.Tensor(0.))
    inter_pair_mask = 1. - intra_pair_mask
    
    pred_coord_local = convert_to_local(pred_coord, pred_rot, pred_trans)   # [B, L, L, N, 3]
    ref_coord_local = convert_to_local(ref_coord, ref_rot, ref_trans)       # [B, L, L, N, 3]
    if mask_2d is None:
        mask2d = mask[..., None] * mask[..., None, :]
    else:
        mask2d = mask_2d
    dist_map = ops.sqrt(ops.sum((pred_coord_local - ref_coord_local) ** 2, -1) + 1e-6)
    dist_map = ops.mean(dist_map, -1)
    # intra clamp fape
    intra_dist_map_clamp = dist_map.clamp(max = intra_clamp_dist)
    intra_dist_map = dist_map / intra_length_scale
    intra_dist_map_clamp = intra_dist_map_clamp / intra_length_scale
    intra_loss = ops.sum(intra_dist_map * mask2d * intra_pair_mask) / (ops.sum(mask2d * intra_pair_mask) + 1e-6)
    intra_loss_clamp = ops.sum(intra_dist_map_clamp * mask2d * intra_pair_mask) / (ops.sum(mask2d * intra_pair_mask) + 1e-6)
    # inter clamp fape
    inter_dist_map_clamp = dist_map.clamp(max = inter_clamp_dist)
    inter_dist_map = dist_map / inter_length_scale
    inter_dist_map_clamp = inter_dist_map_clamp / inter_length_scale
    inter_loss = ops.sum(inter_dist_map * mask2d * inter_pair_mask) / (ops.sum(mask2d * inter_pair_mask) + 1e-6)
    inter_loss_clamp = ops.sum(inter_dist_map_clamp * mask2d * inter_pair_mask) / (ops.sum(mask2d * inter_pair_mask) + 1e-6)
    # import pdb; pdb.set_trace()
    return {
        'intra_loss': intra_loss,
        'intra_loss_clamp': intra_loss_clamp,
        'inter_loss': inter_loss,
        'inter_loss_clamp': inter_loss_clamp
    }
