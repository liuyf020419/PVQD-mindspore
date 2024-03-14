#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib

import numpy as np

import sys, os
import mindspore.dataset as ds

from mindsponge import PipeLine
import logging




logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger= logging.getLogger(__name__)

class FastaBatchedDataset(object):
    def __init__(self, sequence_labels, sequence_strs):
        self.sequence_labels = list(sequence_labels)
        self.sequence_strs = list(sequence_strs)

    @classmethod
    def from_file(cls, fasta_file):
        sequence_labels, sequence_strs = [], []
        cur_seq_label = None
        buf = []

        def _flush_current_seq():
            nonlocal cur_seq_label, buf
            if cur_seq_label is None:
                return
            sequence_labels.append(cur_seq_label)
            sequence_strs.append("".join(buf))
            cur_seq_label = None
            buf = []

        with open(fasta_file, "r") as infile:
            for line_idx, line in enumerate(infile):
                if line.startswith(">"):  # label line
                    _flush_current_seq()
                    line = line[1:].strip()
                    if len(line) > 0:
                        cur_seq_label = line
                    else:
                        cur_seq_label = f"seqnum{line_idx:09d}"
                else:  # sequence line
                    buf.append(line.strip())

        _flush_current_seq()

        assert len(set(sequence_labels)) == len(
            sequence_labels
        ), "Found duplicate sequence labels"

        return cls(sequence_labels, sequence_strs)

    def __len__(self):
        return len(self.sequence_labels)

    def __getitem__(self, idx):
        return self.sequence_labels[idx], self.sequence_strs[idx]

    def get_batch_indices(self, toks_per_batch, extra_toks_per_seq=0):
        sizes = [(len(s), i) for i, s in enumerate(self.sequence_strs)]
        sizes.sort()
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0

        for sz, i in sizes:
            sz += extra_toks_per_seq
            if max(sz, max_len) * (len(buf) + 1) > toks_per_batch:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(i)

        _flush_current_buf()
        return batches


def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"  # noqa
    )

    parser.add_argument(
        "fasta_file",
        type=pathlib.Path,
        help="FASTA file on which to extract representations",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="output directory for extracted representations",
    )

    parser.add_argument("--toks_per_batch", type=int, default=4096, help="maximum batch size")
    parser.add_argument(
        "--repr_layers",
        type=int,
        default=[-1],
        nargs="+",
        help="layers indices from which to extract representations (0 to num_layers, inclusive)",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        choices=["mean", "per_tok", "bos", "contacts"],
        default="per_tok",
        help="specify which representations to return"
    )
    parser.add_argument(
        "--truncation_seq_length",
        type=int,
        default=1022,
        help="truncate sequences longer than the given value",
    )

    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser


def main(args):
    # model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    # model.eval()

    dataset = FastaBatchedDataset.from_file(args.fasta_file)
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    batches = batches[0]
    data_loader = ds.GeneratorDataset(source=dataset, column_names=['sequence_labels', 'sequence_strs'], sampler=batches)
    data_loader = data_loader.create_dict_iterator(num_epochs=1, output_numpy=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = "contacts" in args.include

    pipeline = PipeLine('ESM2')
    pipeline.initialize(config_path="ckpt/structure_prediction/esm2_config.yaml")
    pipeline.model.from_pretrained("ckpt/structure_prediction/esm2.ckpt")


    if True:
        protein_data = []
        for batch_idx, data in enumerate(data_loader):
            labels = str(data['sequence_labels'])
            strs = str(data['sequence_strs'])

            protein_data.append((labels, strs))
            
            logger.info(
                f"Processing {batch_idx + 1} of {len(batches)} batches)"
            )
        logits, representations = pipeline.predict(protein_data)


        for i, data in enumerate(protein_data):

            label = data[0]
            strs = data[1]
            if label[0] == '>':
                label = label[1:]

            os.makedirs(f'{args.output_dir}/{label}', exist_ok=True)
            output_file = f'{args.output_dir}/{label}/{label}_esm.npy'
            # if os.path.isfile(output_file):
            #     continue

            result = {"label": label}
            truncate_len = min(args.truncation_seq_length, len(strs))
            if "per_tok" in args.include:
                sub_representations = representations[i, None]
                sub_representations = sub_representations.squeeze(0)
                result["representations"] = {
                    33: sub_representations[1 : truncate_len + 1, :].asnumpy()
                }

            np.save(
                output_file,
                {'esm2': result}
            )

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
