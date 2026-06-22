"""Synthetic image and token-sequence pairs for the report generator.

Each sample is a small fake "radiograph" tensor paired with a short token
sequence that stands in for an impression. The point is to give the model a
learnable mapping: an image whose appearance is controlled by a hidden class,
and a caption whose tokens are a deterministic function of that class. A model
that wires the encoder to the decoder correctly can overfit this and we can
verify the loss falls.
"""

import torch
from torch.utils.data import Dataset

# Reserved token ids.
PAD = 0
BOS = 1
EOS = 2
SPECIAL_TOKENS = {"<pad>": PAD, "<bos>": BOS, "<eos>": EOS}
NUM_SPECIAL = len(SPECIAL_TOKENS)


class SyntheticReportDataset(Dataset):
    """Image to caption pairs driven by a latent class.

    The image for a given class is a fixed random pattern plus a small amount of
    per-sample noise, so images of the same class look alike. The caption for a
    class is a fixed random sequence of vocabulary tokens wrapped in <bos> and
    <eos>. The mapping from image to caption is therefore consistent and
    learnable.
    """

    def __init__(
        self,
        num_samples: int = 64,
        num_classes: int = 4,
        image_size: int = 16,
        channels: int = 1,
        vocab_size: int = 20,
        caption_len: int = 5,
        noise: float = 0.05,
        seed: int = 0,
    ):
        if vocab_size <= NUM_SPECIAL:
            raise ValueError("vocab_size must leave room for content tokens")
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.channels = channels
        self.vocab_size = vocab_size
        self.caption_len = caption_len
        self.noise = noise

        g = torch.Generator().manual_seed(seed)

        # One canonical image prototype per class.
        self.prototypes = torch.randn(
            num_classes, channels, image_size, image_size, generator=g
        )

        # One caption per class: content tokens drawn from the non-special range.
        content = torch.randint(
            NUM_SPECIAL, vocab_size, (num_classes, caption_len), generator=g
        )
        bos = torch.full((num_classes, 1), BOS, dtype=torch.long)
        eos = torch.full((num_classes, 1), EOS, dtype=torch.long)
        # Shape: [num_classes, caption_len + 2] -> <bos> tokens... <eos>
        self.captions = torch.cat([bos, content, eos], dim=1)

        # Assign a class to each sample, round robin so classes are balanced.
        self.labels = torch.arange(num_samples) % num_classes

        # Pre-generate per-sample noise so the dataset is deterministic.
        self.noise_tensor = (
            torch.randn(
                num_samples, channels, image_size, image_size, generator=g
            )
            * noise
        )

    @property
    def seq_len(self) -> int:
        """Length of the full caption including <bos> and <eos>."""
        return self.caption_len + 2

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        label = int(self.labels[idx])
        image = self.prototypes[label] + self.noise_tensor[idx]
        caption = self.captions[label]
        return image, caption
