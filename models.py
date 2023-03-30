"""Top-level model classes.
Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn


def printsize(emb, string):
    print(f"\n \n {string} size: {emb.size()}. \n \n")


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.
    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).
    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.
    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """

    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.0):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(
            word_vectors=word_vectors,
            char_vectors=char_vectors,
            hidden_size=hidden_size,
            drop_prob=drop_prob,
        )

        self.enc_block = layers.QANetEnc(
            in_dim=5 * hidden_size, num_conv=4, drop_prob=0.1, num_heads=8
        )

        # changed input size to 500 as transition to QANet
        self.enc = layers.RNNEncoder(
            input_size=5 * hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            drop_prob=drop_prob,
        )

        self.att = layers.BiDAFAttention(
            hidden_size=128, drop_prob=drop_prob
        )  # returns 4*h

        self.mod = layers.RNNEncoder(
            input_size=4 * 128, hidden_size=128, num_layers=2, drop_prob=drop_prob,
        )

        self.ModEncBlocks = nn.ModuleList(
            [
                layers.QANetEnc(in_dim=128, num_conv=2, drop_prob=0.1, num_heads=8)
                for i in range(7)
            ]
        )

        self.attentionResize = layers.DWConv(128 * 4)

        self.out = layers.BiDAFOutput(hidden_size=128, drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs, cc_idxs)  # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)  # (batch_size, q_len, hidden_size)

        # c_enc = self.enc(c_emb, c_len)  # (batch_size, c_len, 2 * hidden_size)
        # q_enc = self.enc(q_emb, q_len)  # (batch_size, q_len, 2 * hidden_size)

        c_enc = self.enc_block(c_emb, True)  # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc_block(q_emb, True)  # (batch_size, q_len, 2 * hidden_size)

        # printsize(c_enc, "outputs returned by encoder block")

        att = self.att(
            c_enc, q_enc, c_mask, q_mask
        )  # (batch_size, c_len, 8 * hidden_size)

        # printsize(att, "outputs returned by attention")

        # mod = self.mod(att, c_len)  # (batch_size, c_len, 2 * hidden_size)

        mod = att
        mod = mod.transpose(1, 2)
        mod = self.attentionResize(mod)
        mod = mod.transpose(1, 2)

        for block in self.ModEncBlocks:
            mod = block(mod, False)

        mod2 = mod
        for block in self.ModEncBlocks:
            mod2 = block(mod2, False)

        mod3 = mod
        for block in self.ModEncBlocks:
            mod3 = block(mod3, False)

        # printsize(mod, "outputs returned by encoder modeling layer")

        out = self.out(mod, mod2, mod3, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
