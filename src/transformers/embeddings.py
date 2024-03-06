import torch
import torch.nn as nn


class FixedPositionalEmbedding(nn.Module):
    """
    FixedPositionalEmbedding module that adds positional encoding to the input embeddings.

    Args:
        d_model (int): The dimension of the input embeddings.
        max_len (int, optional): The maximum length of the input sequence. Defaults to 512.
    """

    def __init__(self, d_model, max_len=512):
        super(FixedPositionalEmbedding, self).__init__()

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        pos_enc = torch.zeros(max_len, 1, d_model)
        pos_enc[:, 0, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 0, 1::2] = torch.cos(position * div_term)

        self.dropout = nn.Dropout(p=0.1)
        self.register_buffer("pos_enc", pos_enc)

    def forward(self, x):
        """
        Forward pass of the FixedPositionalEmbedding module.

        Args:
            x (torch.Tensor): The input embeddings.

        Returns:
            torch.Tensor: The input embeddings with positional encoding added.
        """
        x = x + self.pos_enc[: x.size(0), :]
        return self.dropout(x)


class Embeddings(nn.Module):
    def __init__(self, embed_dim, vocab_size, max_seq_len, p=None):
        super().__init__()
        self.token_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=None,
            max_norm=None,
            norm_type=2.0,
            scale_grad_by_freq=False,  # scale gradient by inverse frequency of words
            sparse=False,
            _weight=None,
        )
        self.position_embeddings = FixedPositionalEmbedding(embed_dim, max_seq_len)
        # layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-10)

    def forward(self, x):
        # batch x max_seq_len x emb_dim
        token_embeddings = self.token_embeddings(x)
        # batch x max_seq_len x emb_dim
        token_embeddings = self.position_embeddings(token_embeddings)
        # add both embeddings
        token_embeddings = self.layer_norm(token_embeddings)
        return token_embeddings


if __name__ == "__main__":
    x = torch.randint(0, 12, (32, 12))  # batch x seq_len should go into forward func
    emb = Embeddings(embed_dim=128, num_tokens=12, max_seq_len=512)
    emb(x)
