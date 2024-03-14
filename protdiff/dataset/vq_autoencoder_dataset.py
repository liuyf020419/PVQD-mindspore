import os
import logging
import numpy as np
import collections


from ..models.folding_af2.all_atom import atom37_to_frames
from ..models.folding_af2 import residue_constants
from .protein_coord_parser_new import PoteinCoordsParser

logger = logging.getLogger(__name__)


restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # := 20.
unk_restype_index = restype_num  # Catch-all index for unknown restypes.

restypes_with_x = restypes + ['X']
restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}
idx_to_restype_with_x = {i: restype for i, restype in enumerate(restypes_with_x)}

mpnnalphabet = 'ACDEFGHIKLMNPQRSTVWYX'

STD_RIGID_COORD = np.array(
    [[-0.525, 1.363, 0.000],
    [0.000, 0.000, 0.000],
    [1.526, -0.000, -0.000],
    [0.627, 1.062, 0.000]]
)


class ProtDiffDataset:
    def __init__(self, config, data_list, train=True, pdbroot=None, \
        noising_mode=None, validate=False, permute_data=False, random_seed=None, 
        multichain_inference=False, af2_data=False, batch_size=1):
        super().__init__()
        self.data_list= data_list
        self.config = config.model
        self.config_data = config.data
        self.global_config = self.config.global_config
        self.train_mode = train
        self.validate = validate
        self.af2_data = af2_data
        self.multichain_inference = multichain_inference
        self.batch_size = batch_size

        if self.train_mode:
            self.dataroot = config.data.pdb_data_path
            self.structure_root = config.data.base_path
        else:
            self.dataroot = pdbroot
            self.noising_mode = noising_mode

        self.protein_list = []
        self._epoch = 0
        if validate:
            self.max_len = 10000
        else:
            self.max_len = self.global_config.max_len
        self.enlarge_gap = self.global_config.enlarge_gap

        try:
            self.gap_between_chain = self.config_data.common.gap_between_chain
        except:
            pass

        with open(data_list, 'r') as f:
            for line in f:
                if self.train_mode:
                    self.protein_list.append( '_'.join(line.strip().split()[:2]) )
                else:
                    if af2_data:
                        pdbname, query_name = line.strip().split()[:2]
                        self.protein_list.append((pdbname,query_name))
                    else:
                        line_split = line.strip().split()
                        name = line_split[0]
                        if not self.multichain_inference:
                            chain = line_split[1]
                            self.protein_list.append((name, chain))
                        else:
                            self.protein_list.append(name)


        logger.info(f'list size: {len(self.protein_list)}')

        if permute_data:
            if random_seed is None:
                np.random.seed(None)
            else:
                assert isinstance(random_seed, int)
                np.random.seed(random_seed)
            np.random.shuffle(self.protein_list)


    def __len__(self):
        return len(self.protein_list)

    def data_sizes(self):
        return [l[1] for l in self.protein_list]
    
    def reset_data(self, epoch):
        self._epoch = epoch

    def __getitem__(self, item:int):
        assert self.dataroot is not None

        protein, chain = self.protein_list[item]
        try:
            pdbfile = f'{self.dataroot}/{protein}.pdb'
            data_dict = self.make_from_pdb_file(pdbfile, chain)
        except:
            pdbfile = f'{self.dataroot}/{protein}.cif'
            data_dict = self.make_from_pdb_file(pdbfile, chain)
        pdbname = protein + '_' + chain
        query_name = protein[1:3]

        # if int(os.getenv('MULTI_CHAIN')) != 1:
        #     resrange = (-self.global_config.pair_res_range[1], self.global_config.pair_res_range[1])
        #     resmask_num = self.global_config.pair_res_range[1] + 1
        #     chainrange = (-self.global_config.pair_chain_range[1], self.global_config.pair_chain_range[1])
        #     chainmask_num = self.global_config.pair_chain_range[1] + 1
            
        #     relpdb_residx = np.array(data_dict['pdbresID']) - np.array(data_dict['pdbresID']).min()
        #     self.get_position_embedding(data_dict, relpdb_residx=relpdb_residx, 
        #                                 enlarge_gap=self.enlarge_gap, resrange=resrange,
        #                                 resmask_num=resmask_num, chainrange=chainrange, chainmask_num=chainmask_num)
            
        #     data_dict['pdbname'] = pdbname
        #     data_dict['query_name'] = query_name
        #     data_dict['pdbresID'] = data_dict['pdbresID']
        #     data_dict["loss_mask"] = [False]
        #     data_dict['max_len'] =  data_dict['len']
            
        #     return data_dict['pdbresID'], data_dict['aatype'], data_dict['mpnn_aatype'], data_dict['len'], \
        #         data_dict['gt_pos'], data_dict['gt_backbone_frame'], \
        #         data_dict['single_res_rel'], data_dict['pair_res_rel'], data_dict['pair_chain_rel'], data_dict['pdbname'],\
        #         data_dict['query_name'], data_dict['loss_mask'], data_dict['max_len']
        # else:
        protein_len = data_dict['len'].item()
        # multichain_length_dict = {'A': protein_len}
        # raw_pdb_res_id = {'A': data_dict['pdbresID']}
        multichain_length_dict = {c_: len(pdbresID) for c_, pdbresID in data_dict['pdbresID'].items()}
        raw_pdb_res_id = data_dict['pdbresID']
        # if (len(multichain_length_dict) >= 2):
        data_dict.pop('pdbresID')
        chains_len = list(multichain_length_dict.values())
        raw_single_res_id_list = [ chain_resid for chain_resid in raw_pdb_res_id.values()]
        merged_sequence_str = data_dict['sequence']

        merged_pdbresID = self.make_multichain_single_res_idx(raw_single_res_id_list, self.gap_between_chain)
        chain_rel_pos_dict = self.add_assembly_feature([len(merged_pdbresID)], merged_pdbresID, merged_sequence_str)

        # chain_rel_pos_dict = self.add_assembly_feature(chains_len, merged_pdbresID, merged_sequence_str)
        self.make_multichains_rel_pos(data_dict, chain_rel_pos_dict)
        
        data_dict['pdbname'] = pdbname
        data_dict['query_name'] = query_name
        data_dict["loss_mask"] = np.array([True])
        data_dict['max_len'] =  data_dict['len']

        encode_split_chain = 0.
        if len(chains_len) > 1:
            encode_split_chain = 1
        data_dict['encode_split_chain'] = np.array([encode_split_chain])
        return data_dict['aatype'], data_dict['mpnn_aatype'], data_dict['len'], data_dict['gt_pos'], data_dict['gt_backbone_frame'], \
            data_dict['traj_pos'], data_dict['traj_backbone_frame'], data_dict['sequence'], data_dict['pair_res_idx'], data_dict['pair_same_entity'], data_dict['pair_chain_idx'], \
            data_dict['pair_same_chain'], data_dict['single_res_rel'], data_dict['chain_idx'], data_dict['entity_idx'], data_dict['pdbname'], data_dict['query_name'], \
            data_dict['loss_mask'], data_dict['max_len'], data_dict['encode_split_chain']


    def make_from_pdb_file(self, poteinfile, chain=None):
        data_dict = {}
        assert chain is not None
        # if int(os.getenv('MULTI_CHAIN')) != 1:
        #     PDBparser = PoteinCoordsParser(poteinfile, chain=chain)
        #     gt_pos = PDBparser.chain_main_crd_array.reshape(-1,5,3)
        #     gt_pos = gt_pos.astype(np.float32)
        #     pos_center = np.concatenate([gt_pos]).mean(0)
        #     gt_pos = gt_pos - pos_center
        #     gt_backbone_frame = get_quataffine(gt_pos)
        #     sequence = PDBparser.get_sequence(chain)
        #     data_dict['pdbresID'] = np.array(list(PDBparser.get_pdbresID2absID(chain).keys()))
        # else:
        chain = chain.split('+')
        PDBparser = PoteinCoordsParser(poteinfile, chain=chain)
        gt_pos = PDBparser.chain_main_crd_array.reshape(-1,5,3).astype(np.float32)
        pos_center = np.concatenate([gt_pos]).mean(0)
        gt_pos = gt_pos - pos_center
        gt_backbone_frame = get_quataffine(gt_pos)
        sequence = PDBparser.get_sequence()
        data_dict['pdbresID'] = {c_: list(PDBparser.get_pdbresID2absID(c_).keys()) for c_ in chain}
       
        aatype = [restype_order_with_x[aa] for aa in sequence]
        data_dict["aatype"] = np.array(aatype, dtype=np.int32)
        mpnn_aatype = np.array([
            mpnnalphabet.index(aa) for aa in sequence], dtype=np.int32)
        data_dict['mpnn_aatype'] = mpnn_aatype
        data_dict["len"] = np.array([len(aatype)], dtype=np.int32)
        
        data_dict["gt_pos"] = gt_pos
        data_dict["gt_backbone_frame"] = gt_backbone_frame

        data_dict["traj_pos"] = gt_pos
        data_dict["traj_backbone_frame"] = gt_backbone_frame
        data_dict['sequence'] = sequence

        return data_dict


    def get_position_embedding(self, data_dict, relpdb_residx, resrange=(-32, 32), resmask_num=33, 
                                    chainrange=(-4, 4), chainmask_num=5, enlarge_gap=True, gap_size=100):

        split_idx = np.arange(len(relpdb_residx))[np.append(np.diff(relpdb_residx) != 1, False)] + 1
        # last chain
        chain_num = len(split_idx) + 1
        chain_lens = np.diff(np.append(np.concatenate([[0], split_idx]), len(relpdb_residx) ))

        if enlarge_gap:
            res_rel_idx = []
            for idx, chain_len in enumerate(chain_lens):
                if idx != 0:
                    res_rel_idx.extend(np.arange(chain_len) + res_rel_idx[-1] + gap_size)
                else:
                    res_rel_idx.extend(np.arange(chain_len))

            data_dict["single_res_rel"] = np.array(res_rel_idx, dtype=np.int32)

        else:
            single_part_res_rel_idx = np.concatenate([np.arange(chain_len) for chain_len in chain_lens])
            single_all_chain_rel_idx = np.concatenate([np.ones(chain_len[1], dtype=np.int32) * chain_len[0] \
                                                        for chain_len in enumerate(chain_lens)])

            single_all_res_rel_idx = relpdb_residx - relpdb_residx[0]
            data_dict["single_all_res_rel"] = single_all_res_rel_idx
            data_dict["single_part_res_rel"] = single_part_res_rel_idx
            data_dict["single_all_chain_rel"] = single_all_chain_rel_idx


        pair_res_rel_idx = relpdb_residx[:, None] - relpdb_residx

        unclip_single_chain_rel_idx = np.repeat(np.arange(chain_num), chain_lens)
        pair_chain_rel_idx = unclip_single_chain_rel_idx[:, None] - unclip_single_chain_rel_idx
        
        pair_res_rel_idx = np.where(np.any(np.stack([pair_res_rel_idx > resrange[1], 
                                pair_res_rel_idx < resrange[0]]), 0), resmask_num, pair_res_rel_idx)

        pair_chain_rel_idx = np.where(np.any(np.stack([pair_chain_rel_idx > chainrange[1], 
                                pair_chain_rel_idx < chainrange[0]]), 0), chainmask_num, pair_chain_rel_idx)

        data_dict["pair_res_rel"] = pair_res_rel_idx.astype(np.int64) - resrange[0]
        data_dict["pair_chain_rel"] = pair_chain_rel_idx.astype(np.int64) - chainrange[0]

    def make_multichain_single_res_idx(self, raw_single_res_id_list, gap_between_chain):
        merged_single_res_idx = []
        for c_idx, raw_c_pdbres_idx in enumerate(raw_single_res_id_list):
            raw_c_pdbres_idx = np.array(raw_c_pdbres_idx)
            if c_idx == 0:
                new_c_pdbres_idx = raw_c_pdbres_idx - raw_c_pdbres_idx[0]
            else:
                new_c_pdbres_idx = raw_c_pdbres_idx - raw_c_pdbres_idx[0] + merged_single_res_idx[-1][-1] + gap_between_chain
            merged_single_res_idx.append(new_c_pdbres_idx)

        merged_single_res_idx = np.concatenate(merged_single_res_idx).astype(np.int32)

        return merged_single_res_idx

    def add_assembly_feature(self, chain_lens, merged_pdbresID, seq_str):
        rel_all_chain_features = {}
        seq_to_entity_id = {}
        grouped_chains_length = collections.defaultdict(list)
        chain_length_summed = 0
        for chain_len in chain_lens:
            start_index = chain_length_summed
            chain_length_summed += int(chain_len)
            end_index = chain_length_summed
            seq = seq_str[start_index: end_index]
            if seq not in seq_to_entity_id:
                seq_to_entity_id[seq] = len(seq_to_entity_id) + 1
            grouped_chains_length[seq_to_entity_id[seq]].append(chain_len)

        asym_id_list, sym_id_list, entity_id_list, num_sym_list = [], [], [], []
        chain_id = 0
        for entity_id, group_chain_features in grouped_chains_length.items():
            num_sym = len(group_chain_features)  # zy
            for sym_id, seq_length in enumerate(group_chain_features, start=1):
                asym_id_list.append(chain_id * np.ones(seq_length)) # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                sym_id_list.append(sym_id * np.ones(seq_length)) # [1,2,3,1,2,3,1,2,3,1,2,3,1,2,3]
                entity_id_list.append(entity_id * np.ones(seq_length)) # [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5]
                num_sym_list.append(num_sym * np.ones(seq_length)) # [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
                chain_id += 1

        rel_all_chain_features['asym_id'] = np.concatenate(asym_id_list)
        rel_all_chain_features['sym_id'] = np.concatenate(sym_id_list)
        rel_all_chain_features['entity_id'] = np.concatenate(entity_id_list)
        rel_all_chain_features['num_sym'] = np.concatenate(num_sym_list)
        rel_all_chain_features['res_id'] = merged_pdbresID

        return rel_all_chain_features

    def make_multichains_rel_pos(self, data_dict: str, chain_rel_pos_dict: str, rmax=32, smax=5):
        # pair pos
        diff_aym_id = (chain_rel_pos_dict['asym_id'][None, :] - chain_rel_pos_dict['asym_id'][:, None])
        diff_res_id = (chain_rel_pos_dict['res_id'][None, :] - chain_rel_pos_dict['res_id'][:, None])
        diff_sym_id = (chain_rel_pos_dict['sym_id'][None, :] - chain_rel_pos_dict['sym_id'][:, None])
        diff_entity_id = (chain_rel_pos_dict['entity_id'][None, :] - chain_rel_pos_dict['entity_id'][:, None])

        clamp_res_id = np.clip(diff_res_id+rmax, a_min=0, a_max=2*rmax)
        pair_res_idx = np.where(diff_aym_id.astype(np.int32) == 0, clamp_res_id.astype(np.int32), 2*rmax+1) # 2*rmax + 2

        same_chain = (chain_rel_pos_dict['asym_id'][None, :] == chain_rel_pos_dict['asym_id'][:, None]).astype(np.int32)
        same_entity = (chain_rel_pos_dict['entity_id'][None, :] == chain_rel_pos_dict['entity_id'][:, None]).astype(np.int32) # 2 + 1

        clamp_sym_id = np.clip(diff_sym_id+smax, a_min=0, a_max=2*smax)
        pair_chain_idx = np.where(diff_entity_id.astype(np.int32) == 0, clamp_sym_id.astype(np.int32), 2*smax+1) # 2*smax + 2

        pair_rel_pos_dict = {
            'pair_res_idx': pair_res_idx,
            'pair_same_entity': same_entity,
            'pair_chain_idx': pair_chain_idx,
            'pair_same_chain': same_chain,
            'single_res_rel': chain_rel_pos_dict['res_id'],
            'chain_idx': chain_rel_pos_dict['asym_id'],
            'entity_idx': chain_rel_pos_dict['entity_id']-1
        }
        data_dict.update(pair_rel_pos_dict)


def get_quataffine(pos):
    assert len(pos.shape)
    nres, natoms, _ = pos.shape
    assert natoms == 5
    alanine_idx = residue_constants.restype_order_with_x["A"]
    aatype = np.array([alanine_idx] * nres, dtype=np.float32)
    all_atom_positions = np.pad(pos, ((0, 0), (0, 37-5), (0, 0)), "constant", constant_values=0)
    all_atom_mask = np.ones((nres, 37))
    frame_dict = atom37_to_frames(aatype, all_atom_positions, all_atom_mask)
    return frame_dict['rigidgroups_gt_frames']


