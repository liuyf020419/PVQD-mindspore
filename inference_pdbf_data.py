import contextlib
import logging
import sys
import json
import ml_collections
from collections import OrderedDict
import time
import sys,os
import argparse
import pickle
import numpy as np
from tqdm import tqdm, trange

from typing import Any, Dict, List
from ml_collections import ConfigDict


from protdiff.models.vqstructure import VQStructure
from protdiff.dataset import VQStructureDataset
from protdiff.models.nn_utils import make_mask

import mindspore as ms
from mindspore import set_context, load_checkpoint
import mindspore.dataset as ds


set_context(device_target="GPU")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger= logging.getLogger(__name__)


def build_parser():
    parser = argparse.ArgumentParser(description='Alphafold2')
    parser.add_argument('--gen_dir', type=str, default=None, help='generate dir')
    parser.add_argument('--model_path', type=str, default=None, help='path to checkpoint file')
    parser.add_argument('--fixed_model_path', type=str, default=None, help='path to fixed checkpoint file')
    parser.add_argument('--root_dir', type=str, default=None, help='project path')
    parser.add_argument('--test_list', type=str, default=None, help='test list')
    parser.add_argument('--gen_tag', type=str, default='', help='gen_tag')
    parser.add_argument('--sample_from_raw_pdbfile', action='store_true', help='sample from raw pdbfile or processed file')
    parser.add_argument('--max_sample_num', type=int, default=10000000, help='maximum number of samples for testing or application')
    parser.add_argument('--write_pdbfile', action='store_true', help='write pdb file when testing model')
    parser.add_argument('--pdb_root', type=str, default=None, help='pdb root for application')
    parser.add_argument('--batch_size', type=int, default=1, help='batchsize for each protein')
    parser.add_argument('--num_workers', type=int, default=1, help='num_workers for dataloader')
    parser.add_argument('--save_all', action='store_true', help='write pdb file when testing model')
    parser.add_argument('--return_structure', action='store_true', help='write pdb file when testing model')

    return parser


def load_config(path)->ml_collections.ConfigDict:
    return ml_collections.ConfigDict(json.loads(open(path).read()))


def main(args):
    config_file = os.path.join(args.root_dir, 'config.json')
    assert os.path.exists(config_file), f'config file not exist: {config_file}'
    config= load_config(config_file)

    # modify config for inference
    config.data.train_mode = False
    config.args = args

    logger.info('start preprocessing...')
    model_config = config.model
    global_config = config.model.global_config

    model = VQStructure(model_config.vqstructure, global_config)
    ms.load_checkpoint(args.model_path, model)

    vq_data = VQStructureDataset(config, args.test_list, train=False, pdbroot=args.pdb_root, af2_data=False)
    # dataloader = ds.GeneratorDataset(source=vq_data, column_names=['pdbresID', 'aatype', 'mpnn_aatype', 'len', 'gt_pos', \
    #     'gt_backbone_frame', 'single_res_rel', 'pair_res_rel', 'pair_chain_rel', 'pdbname', 'query_name', 'loss_mask', 'max_len'])
    dataloader = ds.GeneratorDataset(source=vq_data, column_names=['aatype', 'mpnn_aatype', 'len', 'gt_pos', 'gt_backbone_frame', 'traj_pos', 'traj_backbone_frame', \
        'sequence', 'pair_res_idx', 'pair_same_entity', 'pair_chain_idx', 'pair_same_chain', 'single_res_rel', 'chain_idx', 'entity_idx', 'pdbname', 'query_name', 'loss_mask',
            'max_len', 'encode_split_chain'])
    batch_size = 1
    dataloader = dataloader.batch(batch_size, True)
    dataloader = dataloader.create_dict_iterator(num_epochs=1, output_numpy=True)

    output_dir= args.gen_dir
    os.makedirs(output_dir, exist_ok=True)

    

    for batch in dataloader:
        logger.info(batch['pdbname'][0])
        for batch_idx in range(len(batch['pdbname'])):
            batch['len'] = batch['len'].squeeze(0)
            batch['loss_mask'] = batch['loss_mask'].squeeze(0)
            batch['max_len'] = batch['max_len'].squeeze(0)
            batch['encode_split_chain'] = batch['encode_split_chain'].squeeze(0)

            stacked_codebook_indices = model.sampling(batch, output_dir, return_all=args.return_structure, \
                verbose_indices=False, compute_sc_identity=False, save_rep=args.save_all)
            dtype = batch['gt_pos'].dtype
            batchsize, L, N, _ = batch['gt_pos'].shape
            make_mask(batch['len'], batchsize, L, batch, dtype)
            
            saved_data_dict = {}
            pdbname = batch['pdbname'][batch_idx]
            save_query = f'{pdbname}'
            single_mask = batch['single_mask'][batch_idx]
            single_mask = single_mask.astype(np.bool_)
            # saved_data_dict[save_query] = batch['aatype'][batch_idx][single_mask]
            stacked_codebook_indices = stacked_codebook_indices.asnumpy()
            codebook_indices = stacked_codebook_indices[batch_idx].transpose(1, 0)[single_mask].transpose(1, 0) # B, C, N
            # saved_data_dict[save_query] = np.array(codebook_indices[0].tolist())
            
            all_data_f = f'{output_dir}/{pdbname}/{pdbname}.npy'
            saved_data_dict = {
                'pdbname': pdbname,
                'res_idx': batch['single_res_rel'][batch_idx][single_mask],
                'aatype': batch['aatype'][batch_idx][single_mask],
                'structure_ary': np.array(codebook_indices[0].tolist())
            }
            np.save(all_data_f, saved_data_dict)

    # filename = os.path.basename(args.test_list).split('.txt')[0] + '.npy'
    # all_data_f = f'{output_dir}/{filename}'
    # np.save(all_data_f, saved_data_dict)


if __name__=='__main__':
    parser= build_parser()
    args= parser.parse_args()
    main(args)

