
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops



class ResidualCodebook(nn.Cell):
    def __init__(self, args, input_dim, out_dim=None, codebook_num=1, shared_codebook=True, codebook_dropout=False):
        super(ResidualCodebook, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = args.latent_dim
        self.codebook_num = self.head_num = codebook_num
        self.shared_codebook = shared_codebook
        self.codebook_dropout = codebook_dropout

        if out_dim is None:
            out_dim = self.input_dim
        self.beta = args.beta

        self.quant_act = nn.Dense(input_dim, self.latent_dim)

        if shared_codebook:
            self.codebook_layer = []
            for _ in range(codebook_num):
                self.codebook_layer.append(
                    nn.Embedding(self.num_codebook_vectors, self.latent_dim)
                )
        else:
            self.codebook_layer = [nn.Embedding(self.num_codebook_vectors, self.latent_dim)]
        self.codebook_layer  = nn.CellList(self.codebook_layer)

        self.post_quant = nn.Dense(self.latent_dim, out_dim)

    def compute_each_codebook(self, codebook_l, z):
        z_flattened = z.view(-1, self.latent_dim) # BxHxW, C
        z_q_emb = codebook_l.embedding_table

        z_flattened = z.view(-1, self.latent_dim)

        d = ops.sum(z_flattened**2, dim=1, keepdim=True) + ops.sum(z_q_emb**2, dim=1) - 2*( ops.matmul(z_flattened, z_q_emb.transpose()) )

        min_encoding_indices = ops.argmin(d, axis=1) # BxHxW
        z_q = codebook_l(min_encoding_indices).view(z.shape) # B, H, W, C
        # compute loss for embedding
        encoder_qloss = ops.mean((z_q - z)**2, -1)
        code_qloss = ops.mean((z_q - z)**2, -1)
        loss = encoder_qloss + code_qloss * self.beta
        # preserve gradients
        z_q = z + (z_q - z)

        return z_q, min_encoding_indices, loss


    def construct(self, z, return_all_indices=False, use_codebook_num=4):
        dtype = z.dtype

        z = self.quant_act(z.float())
        min_encoding_indices_list = []
        loss = 0
        loss_dict = {}
        z_q_out = 0
        z_q_list = []
        residual = z

        if (self.codebook_dropout and self.training):
            if self.args.only_codebook_num is not None:
                curr_codebook_num = self.args.only_codebook_num
            else:
                codebook_dropout_from = self.args.codebook_dropout_from
                curr_codebook_num = np.random.randint(codebook_dropout_from, self.codebook_num+1, 1)[0]
        else:
            curr_codebook_num = min(use_codebook_num, self.codebook_num)

        for h_idx in range(curr_codebook_num):
            if self.shared_codebook:
                z_q_l, min_encoding_indices_l, loss_l = self.compute_each_codebook(
                    self.codebook_layer[h_idx], residual)
            else:
                z_q_l, min_encoding_indices_l, loss_l = self.compute_each_codebook(
                    self.codebook_layer[0], residual)
            
            z_q_out = z_q_out + z_q_l
            residual = residual - z_q_l
            loss = loss + loss_l
            z_q_list.append(z_q_l[:, None])

            min_encoding_indices_list.append(min_encoding_indices_l)

        loss = loss * 1/len(self.codebook_layer)

        z_q = self.post_quant(z_q_out)

        if return_all_indices:
            return z_q.to(dtype), min_encoding_indices_list, loss.to(dtype), loss_dict, ops.cat(z_q_list, 1), z_q_out
        else:
            return z_q.to(dtype), [min_encoding_indices_list[0]], loss.to(dtype), loss_dict, ops.cat(z_q_list, 1), z_q_out


    def get_feature_from_indices(self, indices):
        assert len(indices.shape) == 3

        codbk_num = indices.shape[0]
        z_q_out = 0

        for h_idx in range(codbk_num):
            if self.shared_codebook:
                z_q_l = self.codebook_layer[h_idx](indices[h_idx])
            else:
                z_q_l = self.codebook_layer[0](indices[h_idx])
            
            z_q_out = z_q_out + z_q_l

        z_q = self.post_quant(z_q_out)
        return z_q