import os
import torch
import mindspore as ms
from mindspore import Tensor
import sys
import numpy as np
import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger= logging.getLogger(__name__)

def convert_checkpoint(checkpoint_path):
    last_cp= checkpoint_path
    if not os.path.exists(last_cp):
        logger.error(f'checkpoint file {last_cp} not exist, ignore load_checkpoint')
        return
    with open(last_cp,'rb') as f:
        logger.info(f'load checkpoint: {checkpoint_path}')
        state = torch.load(f, map_location=torch.device("cpu"))

    from mindspore import save_checkpoint, Tensor

    new_ckpt_list = []

    for k, v in state["model"].items():
        if "high_resolution_decoder." in k:
            continue
        if "in_proj_" in k:
            prefix = ".".join(k.split(".")[:-1])
            v = v.float().numpy()
            v_split = np.split(v, 3)
            
            new_k = [prefix + f".{index}_proj." + k.split("_")[-1] for index in ["q", "k", "v"]]
            for x, y in zip(new_k, v_split):
                print(x, y.shape, y.mean())
                new_ckpt_list.append({"data": Tensor(y), "name": x})

        else:
            k_new = k.replace("norm.weight", "norm.gamma")
            k_new = k_new.replace("norm.bias", "norm.beta")
            k_new = k_new.replace("norm1.weight", "norm1.gamma")
            k_new = k_new.replace("norm1.bias", "norm1.beta")
            k_new = k_new.replace("norm2.weight", "norm2.gamma")
            k_new = k_new.replace("norm2.bias", "norm2.beta")

            k_new = k_new.replace("ln.weight", "ln.gamma")
            k_new = k_new.replace("ln.bias", "ln.beta")

            for i in range(4):            
                k_new = k_new.replace(f"ln_{i}.weight", f"ln_{i}.gamma")
                k_new = k_new.replace(f"ln_{i}.bias", f"ln_{i}.beta")

            k_new = k_new.replace("pe_predictor.2.weight", "pe_predictor.2.gamma")
            k_new = k_new.replace("pe_predictor.2.bias", "pe_predictor.2.beta")
            k_new = k_new.replace("aatype_embedding.weight", "aatype_embedding.embedding_table")
            k_new = k_new.replace("codebook_layer.0.weight", "codebook_layer.0.embedding_table")
            k_new = k_new.replace("r_position_embedding.weight", "r_position_embedding.embedding_table")

            k_new = k_new.replace("single_chain_embedding.weight", "single_chain_embedding.embedding_table")
            k_new = k_new.replace("single_entity_embedding.weight", "single_entity_embedding.embedding_table")
            k_new = k_new.replace("decoder.pair_res_embedding.weight", "decoder.pair_res_embedding.embedding_table")
            k_new = k_new.replace("decoder.pair_chain_embedding.weight", "decoder.pair_chain_embedding.embedding_table")
            k_new = k_new.replace("decoder.pair_chain_entity_embedding.weight", "decoder.pair_chain_entity_embedding.embedding_table")

            k_new = k_new.replace("x0_pred_net.ldm.x_embedder.wtb.weight", "x0_pred_net.ldm.x_embedder.wtb.embedding_table")
            k_new = k_new.replace("x0_pred_net.ldm.y_embedder.embedding_table.weight", "x0_pred_net.ldm.y_embedder.embedding_table.embedding_table")

            if "stacked_encoder.layers" in k or "decoder.preprocess_layers" in k or "stacked_encoder.norm." in k:
                k_new = k
            v = Tensor(v.float().numpy())
            # print(k_new, v.shape, v.asnumpy().mean())

            # print(f"Mapping table Mindspore:{k_new:<30} \t Torch:{k:<30} with shape {v.shape}")
            new_ckpt_list.append({"data": v, "name": k_new})
        print(k, "**",k_new)


    save_checkpoint(new_ckpt_list, "inference_fa_cry.ckpt")

checkpoint_path = "/home/liuyf/proteins/PVQD-mindspore/utils/checkpoint_last.pt"
convert_checkpoint(checkpoint_path)