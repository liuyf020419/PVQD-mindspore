import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms
from mindspore import Tensor

QUAT_TO_ROT = np.zeros((4, 4, 3, 3), dtype=np.float32)

QUAT_TO_ROT[0, 0] = [[ 1, 0, 0], [ 0, 1, 0], [ 0, 0, 1]]  # rr
QUAT_TO_ROT[1, 1] = [[ 1, 0, 0], [ 0,-1, 0], [ 0, 0,-1]]  # ii
QUAT_TO_ROT[2, 2] = [[-1, 0, 0], [ 0, 1, 0], [ 0, 0,-1]]  # jj
QUAT_TO_ROT[3, 3] = [[-1, 0, 0], [ 0,-1, 0], [ 0, 0, 1]]  # kk

QUAT_TO_ROT[1, 2] = [[ 0, 2, 0], [ 2, 0, 0], [ 0, 0, 0]]  # ij
QUAT_TO_ROT[1, 3] = [[ 0, 0, 2], [ 0, 0, 0], [ 2, 0, 0]]  # ik
QUAT_TO_ROT[2, 3] = [[ 0, 0, 0], [ 0, 0, 2], [ 0, 2, 0]]  # jk

QUAT_TO_ROT[0, 1] = [[ 0, 0, 0], [ 0, 0,-2], [ 0, 2, 0]]  # ir
QUAT_TO_ROT[0, 2] = [[ 0, 0, 2], [ 0, 0, 0], [-2, 0, 0]]  # jr
QUAT_TO_ROT[0, 3] = [[ 0,-2, 0], [ 2, 0, 0], [ 0, 0, 0]]  # kr

QUAT_TO_ROT = Tensor(QUAT_TO_ROT, ms.float32)

# calculate on average of valid atoms in 3GCB_A
STD_RIGID_COORD = Tensor(
    [[-1.4589e+00, -2.0552e-07,  2.1694e-07],
     [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
     [ 5.4261e-01,  1.4237e+00,  1.0470e-07],
     [ 5.2744e-01, -7.8261e-01, -1.2036e+00],]
)


def quat_to_rot(quat):
    """Convert a normalized quaternion to a rotation matrix."""
    rot_tensor = ops.sum(
        # np.reshape(QUAT_TO_ROT.to(normalized_quat.device), (4, 4, 9)) *
        QUAT_TO_ROT.view(4, 4, 9) *
        quat[..., :, None, None] *
        quat[..., None, :, None],
        dim=(-3, -2))
    
    new_shape = [s for s in quat.shape[:-1]] + [3, 3]
    rot = ops.reshape(rot_tensor, new_shape)
    return rot
