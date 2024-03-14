import numpy as np

def L2_normalize(vec):
    l2_norm = np.linalg.norm(vec)
    vec_l2_normalized = vec / l2_norm
    return vec_l2_normalized

def nerf(a, b, c, l, theta, chi):
    assert -np.pi <= theta <= np.pi, "theta must be in radians and in [-pi, pi]. theta = " + str(theta)

    W_hat = L2_normalize(b - a)
    x_hat = L2_normalize(c - b)
    
    # calculate unit normals n = AB x BC
    # and p = n x BC
    n_unit = np.cross(W_hat, x_hat)
    z_hat = L2_normalize(n_unit)
    y_hat = np.cross(z_hat, x_hat)

    # create rotation matrix [BC; p; n] (3x3)
    M = np.stack([x_hat, y_hat, z_hat], axis=1)
    # import pdb; pdb.set_trace()
    # calculate coord pre rotation matrix
    D = np.stack([np.squeeze(-l * np.cos(theta)),
                  np.squeeze(l * np.sin(theta) * np.cos(chi)),
                  np.squeeze(l * np.sin(theta) * np.sin(chi))])
    # import pdb; pdb.set_trace()
    # calculate with rotation as our final output
    # TODO: is the squeezing necessary?
    D = np.expand_dims(D, 1)
    res = c + np.matmul(M, D).squeeze()
    return res.squeeze()



def torsion_v0(x1, x2=None, x3=None, x4=None, degrees = False, axis=2):
    """Praxeolitic formula
    1 sqrt, 1 cross product"""
    if (x2 is None) or (x3 is None) or (x4 is None):
        x1, x2, x3, x4 = x1
    b0 = -1.0*(x2 - x1)
    b1 = x3 - x2
    b2 = x4 - x3
    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1, axis=axis, keepdims=True)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.sum(b0*b1, axis=axis, keepdims=True) * b1
    w = b2 - np.sum(b2*b1, axis=axis, keepdims=True) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x

    x = np.sum(v*w, axis=axis)
    b1xv = np.cross(b1, v, axisa=axis, axisb=axis)
    y = np.sum(b1xv*w, axis=axis)
    if degrees:
        return np.float32(180.0 / np.pi) * np.arctan2(y, x)
    else:
        return np.arctan2(y, x)


def add_atom_O(coord3):
    CO_bond = 1.229
    CACO_angle = np.array([2.0944])
    assert isinstance(coord3, np.ndarray)
    assert len(coord3.shape) == 3
    seqlen, _, _ = coord3.shape

    def calc_psi_tors(coord3):
        assert len(coord3.shape) == 3
        N_atoms = coord3[:, 0]
        CA_atoms = coord3[:, 1]
        C_atoms = coord3[:, 2]

        n1_atoms = N_atoms[:-1]
        ca_atoms = CA_atoms[:-1]
        c_atoms = C_atoms[:-1]
        n2_atoms = N_atoms[1:]

        psi_tors = torsion_v0(n1_atoms, ca_atoms, c_atoms, n2_atoms, axis=1)
        return np.concatenate([psi_tors, [0]])

    psi_tors = np.array(calc_psi_tors(coord3))
    coord3 = np.array(coord3)
    # import pdb; pdb.set_trace()
    atomO_coord = [nerf(atom3[-3], atom3[-2], atom3[-1], CO_bond, CACO_angle, psi_tors[resabsD]-np.pi) \
                                    for resabsD, atom3 in enumerate(coord3)]
    new_coord = np.concatenate([coord3.reshape(seqlen, -1), np.stack(atomO_coord)], 1).reshape(seqlen, 4, 3)
    return new_coord


def rebiuld_from_atom_crd(crd_list, chain="A", filename='testloop.pdb', natom=4, natom_dict=None):
    from Bio.PDB.StructureBuilder import StructureBuilder
    from Bio.PDB import PDBIO
    from Bio.PDB.Atom import Atom
    if natom_dict is None:
        natom_dict = {3: {0:'N', 1:'CA', 2: 'C'},
                      4: {0:'N', 1:'CA', 2: 'C', 3:'O'}}
    natom_num = natom_dict[natom]
    sb = StructureBuilder()
    sb.init_structure("pdb")
    sb.init_seg(" ")
    sb.init_model(0)
    chain_id = chain
    sb.init_chain(chain_id)
    for num, line in enumerate(crd_list):
        name = natom_num[num % natom]

        line = np.around(np.array(line, dtype='float'), decimals=3)
        res_num = num // natom
        # print(num//4,line)
        atom = Atom(name=name, coord=line, element=name[0:1], bfactor=1, occupancy=1, fullname=name,
                    serial_number=num,
                    altloc=' ')
        sb.init_residue("GLY", " ", res_num, " ")  # Dummy residue
        sb.structure[0][chain_id].child_list[res_num].add(atom.copy())

    structure = sb.structure
    io = PDBIO()
    io.set_structure(structure)
    io.save(filename)

