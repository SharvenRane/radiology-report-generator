"""A small CNN encoder feeding a GRU decoder for caption generation.

The encoder turns an image into a single context vector. The decoder is a GRU
that conditions on that vector and generates a token sequence. Training uses
teacher forcing: the ground truth previous token is fed at each step. At
inference we run greedy decoding, feeding the decoder its own previous output.
"""

import torch
import torch.nn as nn

from .data import PAD, BOS, EOS


class ImageEncoder(nn.Module):
    """Tiny CNN that maps an image to a fixed-size feature vector."""

    def __init__(self, in_channels: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(32, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [batch, channels, H, W] -> [batch, hidden_dim]
        x = self.features(images)
        x = torch.flatten(x, 1)
        return self.proj(x)


class CaptionDecoder(nn.Module):
    """GRU decoder conditioned on an image context vector.

    The image vector initialises the GRU hidden state. The decoder embeds the
    input tokens, runs the GRU, and projects to vocabulary logits.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

    def forward(
        self, tokens: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        # tokens: [batch, seq], context: [batch, hidden_dim]
        # GRU hidden state expects shape [num_layers, batch, hidden_dim].
        h0 = context.unsqueeze(0)
        emb = self.embedding(tokens)
        output, _ = self.gru(emb, h0)
        return self.out(output)

    def step(
        self, token: torch.Tensor, hidden: torch.Tensor
    ):
        """Single decode step for greedy generation.

        token: [batch, 1] of the previous token id.
        hidden: [1, batch, hidden_dim] GRU hidden state.
        Returns logits [batch, vocab_size] and the new hidden state.
        """
        emb = self.embedding(token)
        output, hidden = self.gru(emb, hidden)
        logits = self.out(output.squeeze(1))
        return logits, hidden


class ReportGenerator(nn.Module):
    """Full encoder plus decoder model."""

    def __init__(
        self,
        vocab_size: int,
        in_channels: int = 1,
        embed_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.encoder = ImageEncoder(in_channels=in_channels, hidden_dim=hidden_dim)
        self.decoder = CaptionDecoder(
            vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim
        )
        self.vocab_size = vocab_size

    def forward(
        self, images: torch.Tensor, captions: torch.Tensor
    ) -> torch.Tensor:
        """Teacher forced forward pass.

        Feeds caption[:, :-1] as decoder input and the model predicts the next
        token at each position. Returns logits aligned with caption[:, 1:].
        """
        context = self.encoder(images)
        decoder_input = captions[:, :-1]
        return self.decoder(decoder_input, context)

    @torch.no_grad()
    def generate(
        self, images: torch.Tensor, max_len: int, bos: int = BOS
    ) -> torch.Tensor:
        """Greedy decoding. Returns generated token ids [batch, max_len].

        Generation always emits exactly max_len tokens. The <bos> token is the
        first input but is not part of the returned sequence, so the output is a
        fixed length sequence of predicted tokens.
        """
        self.eval()
        context = self.encoder(images)
        batch = images.size(0)
        hidden = context.unsqueeze(0)
        token = torch.full((batch, 1), bos, dtype=torch.long, device=images.device)

        generated = []
        for _ in range(max_len):
            logits, hidden = self.decoder.step(token, hidden)
            token = logits.argmax(dim=-1, keepdim=True)
            generated.append(token)
        return torch.cat(generated, dim=1)
