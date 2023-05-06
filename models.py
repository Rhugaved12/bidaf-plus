"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn


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

    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.hidden_size = 3 * hidden_size  # resizing due to char embeddings
        self.hops = 2  # number of hops for reattention
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)
        self.context_emb = layers.Context_Embedding(word_vectors=word_vectors,
                                                    char_vectors=char_vectors,
                                                    hidden_size=hidden_size,
                                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=self.hidden_size,
                                     hidden_size=self.hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        # ------------------------------------
        # Interactive aligning, self aligning and aggregating
        doc_hidden_size = 2 * self.hidden_size
        # RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
        # CELL_TYPES = {'lstm': nn.LSTMCell, 'gru': nn.GRUCell, 'rnn': nn.RNNCell}
        self.interactive_aligners = nn.ModuleList()
        self.interactive_SFUs = nn.ModuleList()
        self.self_aligners = nn.ModuleList()
        self.self_SFUs = nn.ModuleList()
        self.aggregate_rnns = nn.ModuleList()
        for i in range(self.hops):
            # interactive aligner
            self.interactive_aligners.append(
                layers.SeqAttnMatch(doc_hidden_size, identity=True))
            self.interactive_SFUs.append(layers.SFU(
                doc_hidden_size, 3 * doc_hidden_size))
            # self aligner
            self.self_aligners.append(layers.SelfAttnMatch(
                doc_hidden_size, identity=True, diag=False))
            self.self_SFUs.append(layers.SFU(
                doc_hidden_size, 3 * doc_hidden_size))
            # aggregating
            self.aggregate_rnns.append(
                layers.StackedBRNN(
                    input_size=doc_hidden_size,
                    hidden_size=self.hidden_size,
                    num_layers=1,
                    dropout_rate=drop_prob,
                    dropout_output=drop_prob,
                    concat_layers=False,
                    # rnn_type=self.RNN_TYPES[args.rnn_type],
                    # padding=args.rnn_padding,
                )
            )

        # -------------------------------------

        self.att = layers.BiDAFAttention(hidden_size=2 * self.hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * self.hidden_size,
                                     hidden_size=self.hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=self.hidden_size,
                                      drop_prob=drop_prob)

    # tag_idxs, ent_idxs, context_tfs, em_lowers, cw_idxs, qw_idxs)
    def forward(self, tag_idxs, ent_idxs, context_tfs, em_lowers, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        # c_mask and q_mask will be boolean tensors with the same shape as cw_idxs and qw_idxs, respectively.
        # The values in these tensors will be True where the corresponding elements in cw_idxs and qw_idxs are
        # non-zero, and False where the corresponding elements in cw_idxs and qw_idxs are zero.
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_x1_mask = torch.zeros_like(cw_idxs) == cw_idxs
        q_x1_mask = torch.zeros_like(qw_idxs) == qw_idxs
        # c_len = x1_mask.sum(-1)

        # (batch_size, c_len, hidden_size)
        c_emb = self.context_emb(tag_idxs, ent_idxs, context_tfs,
                                 em_lowers, cw_idxs, cc_idxs)
        # (batch_size, q_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)
        # print("After embed forward: ", c_emb.shape,
        #       c_emb.dtype, q_emb.shape, c_len, q_len)
        # (batch_size, c_len, 2 * hidden_size)
        # print("in models: ", c_len, c_emb.shape, cw_idxs.shape,
        #       q_emb.shape, c_mask.shape, q_mask.shape)
        c_enc = self.enc(c_emb, c_len)
        # (batch_size, q_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)

        # -------------------------------------------
        # Align and aggregate
        for i in range(self.hops):
            q_tilde = self.interactive_aligners[i].forward(
                c_enc, q_enc, q_x1_mask)
            c_bar = self.interactive_SFUs[i].forward(c_enc, torch.cat(
                [q_tilde, c_enc * q_tilde, c_enc - q_tilde], 2))
            c_tilde = self.self_aligners[i].forward(c_bar, c_x1_mask)
            c_hat = self.self_SFUs[i].forward(c_bar, torch.cat(
                [c_tilde, c_bar * c_tilde, c_bar - c_tilde], 2))
            c_enc = self.aggregate_rnns[i].forward(c_hat, c_x1_mask)

        # ---------------------------------------------
        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        # (batch_size, c_len, 2 * hidden_size)
        mod = self.mod(att, c_len)

        # 2 tensors, each (batch_size, c_len)
        out = self.out(att, mod, c_mask)

        return out
