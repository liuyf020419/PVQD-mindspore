from tqdm import trange

import mindspore as ms
import mindspore.nn as nn
from mindspore import Parameter
from mindspore import jit
import mindspore.ops as ops
from mindspore.common.tensor import Tensor

from .conditioner.esm_conditioner import ESMConditioner
from .conditioner.gvp_conditioner import GVPConditioner

from .latentdiff.latent_diff_model import LatentDiffModel
from .nn_utils import make_mask, TransformerPositionEncoding

class DDPM(nn.Cell):
    def __init__(self, config, global_config, use_context=False, inpainting=False) -> None:
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.use_context = use_context
        self.inpainting = inpainting

        beta_start, beta_end = global_config.diffusion.betas
        T = global_config.diffusion.T
        self.T = T
        betas = ops.linspace(beta_start, beta_end, T).astype(ms.float32)

        alphas = 1. - betas
        alphas_cumprod = ops.cumprod(alphas, dim=0)
        alphas_cumprod_prev = ops.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        self.betas = Parameter(betas.float(), requires_grad=False)
        self.alphas_cumprod = Parameter(alphas_cumprod.float(), requires_grad=False)
        self.alphas_cumprod_prev = Parameter(alphas_cumprod_prev.float(), requires_grad=False)
        self.sqrt_alphas_cumprod = Parameter(ops.sqrt(alphas_cumprod).float(), requires_grad=False)
        self.sqrt_one_minus_alphas_cumprod = Parameter(ops.sqrt(1. - alphas_cumprod).float(), requires_grad=False)

        # Calculations for posterior q(y_{t-1} | y_t, y_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_variance = ops.cat([posterior_variance[1][None], posterior_variance[1:]])
        posterior_log_variance_clipped = posterior_variance.log()
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        posterior_mean_coef1 = betas * alphas_cumprod_prev.sqrt() / (1 - alphas_cumprod)
        posterior_mean_coef2 = (1 - alphas_cumprod_prev) * alphas.sqrt() / (1 - alphas_cumprod)

        self.posterior_log_variance_clipped= Parameter(posterior_log_variance_clipped.float(), requires_grad=False)
        self.posterior_mean_coef1= Parameter(posterior_mean_coef1.float(), requires_grad=False)
        self.posterior_mean_coef2= Parameter(posterior_mean_coef2.float(), requires_grad=False)

        if self.use_context:
            self.conditioner = ESMConditioner(config.esm_conditioner, global_config)
            self.x0_pred_net = LatentDiffModel(config, global_config, self.conditioner.embed_dim)
        elif self.inpainting:
            self.conditioner = GVPConditioner(config, global_config)
            self.x0_pred_net = LatentDiffModel(config, global_config, self.conditioner.embed_dim)	    
        else:
            self.x0_pred_net = LatentDiffModel(config, global_config)

        self.if_norm_latent = if_norm_latent = getattr(self.global_config.latentembedder, 'norm_latent', False)

        if if_norm_latent:
            self.latent_scale = 1.0
        else:
            self.latent_scale = self.global_config.latent_scale # 3.2671

        self.single_res_embedding = TransformerPositionEncoding(
            global_config.max_seq_len, config.DiTEncoder.embed_dim)
        self.single_chain_embedding = nn.Embedding(
            global_config.max_chain_len, config.DiTEncoder.embed_dim)
        self.single_entity_embedding = nn.Embedding(
            global_config.max_entity_len, config.DiTEncoder.embed_dim)


    def q_sample(self, x0_dict: dict, t, noising_scale=1.0):
        # Calculations for posterior q(x_{t} | x_0, mu)
        xt_dict = {}
        if x0_dict.__contains__('latent_rep'):
            xt_esm = self.degrad_latent(x0_dict['latent_rep'], t, noising_scale=noising_scale)
            xt_dict['latent_rep'] = xt_esm

        return xt_dict


    @jit
    def degrad_latent(self, latent_0, t, noising_scale=1.0):
        t1 = t[..., None, None]
        noise = ops.randn_like(latent_0) * self.sqrt_one_minus_alphas_cumprod[t1] * noising_scale * self.latent_scale
        degraded_latent = latent_0 * self.sqrt_alphas_cumprod[t1] + noise

        return degraded_latent


    def sampling(
        self, 
        batch: dict, 
        pdb_prefix: str, 
        step_num: int, 
        init_noising_scale=1.0,
        diff_noising_scale=1.0,
        mapping_nn=False
        ):
        batch_size, num_res = batch['aatype'].shape[:2]
        latent_dim = self.global_config.in_channels
        latent_rep_nosie = ops.randn((batch_size, num_res, latent_dim), dtype=ms.float32) * init_noising_scale
        # latent_rep_nosie = ops.ones((1, num_res, latent_dim), dtype=ms.float32) * init_noising_scale
        alatent_rep_t = latent_rep_nosie * self.latent_scale
        make_mask(batch['len'], batch_size, num_res, batch)

        if self.use_context:
            condition_mask = Tensor(batch['condition_mask'], ms.int32)
            esm_rep = Tensor(batch['esm_rep'], ms.float32)
            condition_embed = self.conditioner(condition_mask, esm_rep)
            if (getattr(self.global_config.loss_weight, "sidechain_embed_loss", 0.0) > 0.0 or (getattr(self.global_config.loss_weight, "sidechain_simi_loss", 0.0) > 0.0) ):
                condition_embed, sc_condtion_rep = condition_embed
            batch['condition_embed'] = condition_embed
        elif self.inpainting:
            condition_embed = self.conditioner(batch)
            batch['condition_embed'] = condition_embed
        else:
            condition_embed = None

        batch['protein_state'] = batch['protein_state'][0]

        xt_dict = {
            'latent_rep': alatent_rep_t
        }
        batch['xt_dict'] = xt_dict

        batch['single_res_rel'] = Tensor(batch['single_res_rel'], ms.int32)
        batch['chain_idx'] = Tensor(batch['chain_idx'], ms.int32)
        batch['entity_idx'] = Tensor(batch['entity_idx'], ms.int32)

        batch['xt_dict']['latent_rep'] = Tensor(batch['xt_dict']['latent_rep'])
        batch['protein_state'] = Tensor(batch['protein_state'])
        batch['single_mask'] = Tensor(batch['single_mask']).float()

        t_scheme = list(range(self.T-1, -1, -step_num))

        single_mask = batch['single_mask']
        padding_mask = ~single_mask.bool()
        single_idx = batch['single_res_rel']
        chain_idx = batch['chain_idx']
        entity_idx = batch['entity_idx']

        single_idx = single_idx * ~padding_mask + self.global_config.pad_num*padding_mask
        chain_idx = (chain_idx * ~padding_mask).float() + self.global_config.pad_chain_num*padding_mask
        entity_idx = (entity_idx * ~padding_mask).float() + self.global_config.pad_entity_num*padding_mask

        single_condition = self.single_res_embedding(single_idx, index_select=True).to(ms.float32) + \
            self.single_chain_embedding(chain_idx.astype(ms.int32)).to(ms.float32) + \
                self.single_entity_embedding(entity_idx.astype(ms.int32)).to(ms.float32)
        if t_scheme[-1] != 0:
            t_scheme.append(0)

        input_hidden = batch['xt_dict']['latent_rep']
        y = batch['protein_state']
        single_mask = batch['single_mask']
        t = Tensor([0] * batch_size)
        pred_latent, l2_distance = self.x0_pred_net(t, y, single_mask, input_hidden, single_condition, condition_embed)


        for t_idx in trange(len(t_scheme)):
            input_hidden = batch['xt_dict']['latent_rep']
            y = batch['protein_state']
            single_mask = batch['single_mask']

            t = t_scheme[t_idx]
            t = Tensor([t] * batch_size)
            batch['t'] =  t

            x0_dict = {}
            t = batch['t']

            pred_latent, l2_distance = self.x0_pred_net(t, y, single_mask, input_hidden, single_condition, condition_embed)
            x0_dict['pred_latent'] = pred_latent
            x0_dict['l2_distance'] = l2_distance
            if not mapping_nn:
                x0_dict['latent_rep'] = x0_dict['pred_latent']
            else:
                x0_dict['latent_rep'] = self.find_nn_latent(x0_dict['pred_latent'])
            x_t_1_dict = self.q_sample(x0_dict, t, noising_scale=diff_noising_scale)
            batch['xt_dict'] = x_t_1_dict
        return x0_dict

    @jit
    def find_nn_latent(self, pred_latent):
        wtb_weight = self.x0_pred_net.ldm.x_embedder.wtb.embedding_table
        l2_distance = ops.sum((pred_latent[..., None, :] - wtb_weight[None, None])**2, -1)  # B, L, N
        nn_token = ops.argmin(l2_distance, axis=-1)
        nn_latent_rep = ops.gather(wtb_weight, nn_token, 0)
        return nn_latent_rep
