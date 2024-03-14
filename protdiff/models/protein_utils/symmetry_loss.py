
import mindspore.numpy as mnp
from mindspore.common.tensor import Tensor
import mindspore.ops as ops

def get_asym_mask(asym_id):
    """get the mask for each asym_id. [*, NR] -> [*, NC, NR]"""
    # this func presumes that valid asym_id ranges [1, NC] and is dense.
    asym_type = mnp.arange(1, ops.amax(asym_id) + 1)  # [NC]
    return (asym_id[..., None, :] == asym_type[:, None]).float()


def center_mass_loss_batch(
    assembly_pred_coord, # B, NR * NO, NA, 3
    assembly_native_coord, # B, NR * NO, NA, 3
    single_mask, # B, NR * NO
    asym_id, # B, NR * NO
    eps: float = 1e-10):
    """
    NR: AU_len, NO: Ops_num, NC: Chain_num
    e.g. ER: 12; NO: 2; NC: 6
    """
    # import pdb; pdb.set_trace()
    asym_id = Tensor(asym_id)
    single_mask = Tensor(single_mask)
    assembly_native_coord = Tensor(assembly_native_coord)

    monomer_mask = ops.any(asym_id > 0, 1)
    asym_mask = get_asym_mask(asym_id + 1) * single_mask[..., None, :] # B, NC, NR * NO
    asym_exists = ops.any(asym_mask, axis=-1).float()  # [B, NC]
    pred_atom_positions = assembly_pred_coord[..., 1, :].float()  # [B, NR * NO, 3]
    true_atom_positions = assembly_native_coord[..., 1, :].float()  # [B, NR * NO, 3]
    # batch_size, NC = asym_mask.shape[:2]
    # device = asym_mask.device

    def get_asym_centres(pos):
        pos = pos[..., None, :, :] * asym_mask[..., :, :, None]  # [B, NC, NR * NO, 3]
        return ops.sum(pos, dim=-2) / (ops.sum(asym_mask, dim=-1)[..., None] + eps)
    
    pred_centres = get_asym_centres(pred_atom_positions)  # [B, NC, 3]
    true_centres = get_asym_centres(true_atom_positions)  # [B, NC, 3]

    def get_dist(p1, p2):
        return ops.sqrt(
            (p1[..., :, None, :] - p2[..., None, :, :]).square().sum(-1) + eps
        )

    pred_centres2 = pred_centres
    true_centres2 = true_centres
    pred_dists = get_dist(pred_centres, pred_centres2)  # [B, NC, NC]
    true_dists = get_dist(true_centres, true_centres2)  # [B, NC, NC]
    losses = (pred_dists - true_dists + 4).clamp(max=0).square() * 0.0025
    loss_mask = asym_exists[..., :, None] * asym_exists[..., None, :]  # [B, NC, NC]
    # loss_mask = loss_mask * entity1_chain_mask[..., None]
    center_mass_loss = ops.sum((loss_mask * losses)[monomer_mask]) / ( eps + ops.sum(loss_mask[monomer_mask]) )

    return center_mass_loss



