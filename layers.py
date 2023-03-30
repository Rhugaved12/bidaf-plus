"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

from turtle import forward
from numpy import pad
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax


def printsize(emb, string):
    print(f"\n \n {string} size: {emb.size()}. \n \n")


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """

    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.1):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        self.char_embed = nn.Embedding.from_pretrained(char_vectors, freeze=False)
        self.emb_dim = self.char_embed.embedding_dim
        self.conv = nn.Sequential(
            nn.Conv2d(
                self.emb_dim,
                self.emb_dim,
                kernel_size=7,
                groups=self.emb_dim,
                padding=7 // 2,
            ),
            nn.Conv2d(self.emb_dim, 200, kernel_size=1, padding=0),
        )
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    """
    TO-DO: Dimensionality adjustment. Right now I'm outputting a 100-d embedding so the encoder layer won't
    complain as it hasn't been adjusted. However, for QANet it should be a 500-d (300 from w and 200 from c)
    """

    # TO-DO: replace max and squeeze with maxpooling
    def forward(self, w_idxs, c_idxs):
        char_emb = self.char_embed(c_idxs)  # (b, seq_len, word_len, embed_size)
        char_emb = char_emb.permute(0, 3, 1, 2)  # (b, embed_size, seq_len, word_len)
        char_emb = F.dropout(char_emb, 0.05, self.training)
        char_emb = self.conv(char_emb)  # (b, embed_size, seq_len, word_len)
        char_emb = F.relu(char_emb)  # (b, embed_size, seq_len, word_len)
        char_emb, idxs = torch.max(char_emb, dim=-1)  # (b, embed_size, seq_len)
        char_emb = char_emb.permute(0, 2, 1)

        word_emb = self.word_embed(w_idxs)  # (b, seq_len, embed_size)
        word_emb = F.dropout(word_emb, self.drop_prob, self.training)
        # word_emb = self.proj(word_emb)

        emb = torch.cat([word_emb, char_emb], dim=-1)
        emb = self.hwy(emb)  # (b, seq_len, 500)

        # printsize(emb, "Output from embedding layer")
        return emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """

    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList(
            [nn.Linear(5 * hidden_size, 5 * hidden_size) for _ in range(num_layers)]
        )
        self.gates = nn.ModuleList(
            [nn.Linear(5 * hidden_size, 5 * hidden_size) for _ in range(num_layers)]
        )

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class DWConv(nn.Module):
    def __init__(self, emb_dim):
        super(DWConv, self).__init__()
        self.depth = nn.Conv1d(emb_dim, 128, kernel_size=7, groups=128, padding=7 // 2,)
        self.point = nn.Conv1d(128, 128, kernel_size=1, padding=0)

    def forward(self, x):
        a = self.depth(x)
        b = self.point(a)
        return b


class MultiSelfAttention(nn.Module):
    def __init__(self, in_dim, num_heads, drop_prob, batch_first=True):
        super(MultiSelfAttention, self).__init__()
        self.multihead_self_attention = nn.MultiheadAttention(
            in_dim, num_heads, drop_prob, batch_first
        )
        self.queries = nn.Linear(in_dim, in_dim)
        self.keys = nn.Linear(in_dim, in_dim)
        self.values = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        Q = self.queries(x)
        K = self.keys(x)
        V = self.values(x)

        return self.multihead_self_attention(Q, K, V)


import math
from torch.autograd import Variable


class PositionEncoding(nn.Module):
    def __init__(self, model_dim, max_length=512):

        super().__init__()
        self.model_dim = model_dim
        pos_encoding = torch.zeros(max_length, model_dim)
        for pos in range(max_length):
            for i in range(0, model_dim, 2):
                pos_encoding[pos, i] = math.sin(pos / (10000 ** ((2 * i) / model_dim)))
                pos_encoding[pos, i + 1] = math.cos(
                    pos / (10000 ** ((2 * (i + 1)) / model_dim))
                )

        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, x):
        enc = Variable(self.pos_encoding[:, : x.shape[1]], requires_grad=False)
        x = x + enc
        return x


class QANetEnc(nn.Module):
    def __init__(self, in_dim, num_conv, drop_prob, num_heads):
        super(QANetEnc, self).__init__()
        self.position_encoding = PositionEncoding(in_dim)
        self.layernorm = nn.LayerNorm(128)
        self.ffnorm = nn.LayerNorm(128)
        self.sanorm = nn.LayerNorm(128)
        self.num_conv = num_conv
        self.drop_prob = drop_prob
        self.layernorms = nn.ModuleList([nn.LayerNorm(128) for i in range(num_conv)])
        self.convolutions = nn.ModuleList([DWConv(128) for i in range(num_conv)])
        self.multihead_self_attention = MultiSelfAttention(
            128, num_heads, drop_prob, True
        )
        self.mapConv = nn.Conv1d(in_dim, 128, 7, padding=7 // 2)
        self.ff = nn.Linear(128, 128)

    def forward(self, x, is_from_emb):
        # printsize(x, "Input received by QANetEnc layer")
        x = self.position_encoding(x)
        # map to 128 per spec. If you don't do this, you get the error: embed_dim must be divisible by num_heads
        # do it only for the encoder block. This module is used also by the modeling block
        if is_from_emb:
            x = x.transpose(1, 2)
            x = self.mapConv(x)
            x = x.transpose(1, 2)
        # printsize(x, "Input aftr mapping to 128 with conv1d")

        # Layernorm + Conv block
        for i in range(self.num_conv):
            residual = x
            layernorm = self.layernorms[i]
            x = layernorm(x)
            # printsize(x, "Output after layernorm")
            conv = self.convolutions[i]
            x = x.transpose(1, 2)
            x = conv(x)
            x = F.relu_(x)
            x = x.transpose(1, 2)
            # printsize(x, "Output after convolution")
            # x = x + residual
            # printsize(x, "Output after resnet")
            if (i + 1) % 2 == 0:
                x = F.dropout(x + residual, self.drop_prob, self.training)

        # printsize(x, "Output after 4 conv miniblocks")

        # Self-attention block
        residual2 = x
        x2 = self.sanorm(x)
        # x2 = F.dropout(x2, p=0.1)
        x2, _ = self.multihead_self_attention(x2)  # ret outputs and weights
        # printsize(x2, "Output after multihead self attention")
        x2 = F.dropout(x + residual2, p=0.1)
        # x2 += residual2

        # feedforward
        residual3 = x2
        x3 = self.ffnorm(x2)
        x3 = self.ff(x3)
        x3 = F.relu(x3)
        x3 = F.dropout(x3 + residual3, p=0.1)
        # x3 += residual3  # (batch_size, seq_len, 128)

        # printsize(x3, "Output returned by encoder block")
        return x3


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self, input_size, hidden_size, num_layers, drop_prob=0.0):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=drop_prob if num_layers > 1 else 0.0,
        )

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]  # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]  # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)  # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)  # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)  # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2).expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.W1 = nn.Linear(2 * hidden_size, 1, bias=False)
        self.W2 = nn.Linear(2 * hidden_size, 1, bias=False)

        # self.rnn = RNNEncoder(
        #     input_size=2 * hidden_size,
        #     hidden_size=hidden_size,
        #     num_layers=1,
        #     drop_prob=drop_prob,
        # )

        # self.att_linear_2 = nn.Linear(4 * hidden_size, 1)
        # self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, m_0, m_1, m_2, mask):

        logits_1 = torch.cat([m_0, m_1], dim=2)
        logits_1 = self.W1(logits_1)
        logits_2 = torch.cat([m_0, m_2], dim=2)
        logits_2 = self.W2(logits_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
