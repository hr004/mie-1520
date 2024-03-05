import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import sparsemax


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        # combine query and keys and apply non-linearization
        # to compute the alignment score
        # batch, seq_len, 1
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        # 32,1,10
        scores = scores.squeeze(2).unsqueeze(1)
        # compute attention probabilities
        weights = F.softmax(scores, dim=-1)
        # compute context vector
        context = torch.bmm(weights, keys)

        return context, weights


class SparseBahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SparseBahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, encoder_output, decoder_hidden):
        query = self.Wa(decoder_hidden).unsqueeze(1)
        keys = self.Ua(encoder_output)
        energy = self.Va(torch.tanh(keys + query)).squeeze(2)

        attention_weights = sparsemax(energy, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_output)

        return context, attention_weights


class LocalLuongAttention(nn.Module):
    def __init__(self, hidden_size):
        super(LocalLuongAttention, self).__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):

        # decoder_hidden shape (batch_size, hiddeen_dim)
        # encoder_outputs shape (batch_size, seq_len, hidden_dim)
        # energy shape (batch_size, seq_len, hidden_dim)

        # "energy" is used to refer to the result of combining the encoder outputs
        # and the decoder hidden state to calculate the attention scores

        # concat version of luong attention
        energy = torch.tanh(self.W(encoder_outputs + decoder_hidden))
        # attention scores should be of shape (batch_size, seq_len)
        attention_scores = self.v(energy).squeeze(-1)

        # alignment_scores shape (batch_size, seq_len)
        alignment_scores = F.softmax(attention_scores, dim=-1)

        # compute context vector
        # alignment_scores shape (batch_size, 1, seq_len)
        alignment_scores = alignment_scores.unsqueeze(1)
        context = torch.bmm(alignment_scores, encoder_outputs)

        return context, alignment_scores
