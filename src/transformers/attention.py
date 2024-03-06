import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Self-Attention module that performs multi-head self-attention on the input tensor.

    Args:
        embedding_dim (int): The dimensionality of the input embeddings.
        heads (int, optional): The number of attention heads. Defaults to 8.
        mask (bool, optional): Whether to apply masking during attention computation. Defaults to False.
    """

    def __init__(self, embedding_dim, heads=8, mask=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.heads = heads
        # sanity check
        assert (
            embedding_dim % heads == 0
        ), f"embedding dim {embedding_dim} should be multiple of heads {heads}"
        self.mask = mask
        # stacking all heads in a single matrix thats why output is embedding_dim * heads dimensional
        self.tokeys = nn.Linear(embedding_dim, embedding_dim * heads)
        self.toqueries = nn.Linear(embedding_dim, embedding_dim * heads)
        self.tovalues = nn.Linear(embedding_dim, embedding_dim * heads)
        # concate heads output and reduce to embedding_dim
        self.unifyheads = nn.Linear(embedding_dim * heads, embedding_dim)

    def forward(self, x):
        """
        Forward pass of the SelfAttention module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, sequence_length, embedding_dim).
        """
        # batch first should be true in torchtext
        b, t, e = x.size()
        h = self.heads
        assert (
            e == self.embedding_dim
        ), f"mismatch between input emb {e} and layer emb {self.embedding_dim}"
        # split the embedding into h heads
        keys = self.tokeys(x)
        keys = self.split_heads(keys, b, t)

        keys = self.group_heads(keys, b, t)

        queries = self.toqueries(x)
        queries = self.split_heads(queries, b, t)
        queries = self.group_heads(queries, b, t)

        values = self.tovalues(x)
        values = self.split_heads(values, b, t)
        values = self.group_heads(values, b, t)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))
        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b * h, t, t)

        if self.mask:
            # TODO move this to utils.py
            mask_diagnoal = False
            indices = torch.triu_indices(t, e, offset=0 if mask_diagnoal else 1)
            dot[:, indices[0], indices[1]] = 0.0

        dot = F.softmax(dot, dim=2)
        # dot has now row wise self attention probabilities

        out = torch.bmm(dot, values).view(b, h, t, e)
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        return self.unifyheads(out)

    def split_heads(self, x, b, t):
        """
        Split the input tensor into multiple heads.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, embedding_dim).
            b (int): The batch size.
            t (int): The sequence length.

        Returns:
            torch.Tensor: The tensor with shape (batch_size, sequence_length, heads, embedding_dim).
        """
        # https://github.com/Atcold/pytorch-Deep-Learning/blob/master/15-transformer.ipynb
        return x.view(b, t, self.heads, self.embedding_dim)

    def group_heads(self, x, b, t):
        """
        Group the heads of the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, heads, embedding_dim).
            b (int): The batch size.
            t (int): The sequence length.

        Returns:
            torch.Tensor: The tensor with shape (batch_size * heads, sequence_length, embedding_dim).
        """
        return (
            x.transpose(1, 2).contiguous().view(b * self.heads, t, self.embedding_dim)
        )


if __name__ == "__main__":
    x = torch.randn((32, 512, 128))
    attn = SelfAttention(embedding_dim=128)
    attn(x)
