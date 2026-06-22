import torch

from src.data import SyntheticReportDataset
from src.model import ImageEncoder, CaptionDecoder, ReportGenerator


def test_encoder_output_shape():
    enc = ImageEncoder(in_channels=1, hidden_dim=64)
    images = torch.randn(8, 1, 16, 16)
    out = enc(images)
    assert out.shape == (8, 64)


def test_decoder_output_shape():
    dec = CaptionDecoder(vocab_size=20, embed_dim=32, hidden_dim=64)
    tokens = torch.randint(0, 20, (8, 7))
    context = torch.randn(8, 64)
    logits = dec(tokens, context)
    assert logits.shape == (8, 7, 20)


def test_full_model_forward_shape():
    ds = SyntheticReportDataset(num_samples=8, vocab_size=20, caption_len=5)
    images = torch.stack([ds[i][0] for i in range(8)])
    captions = torch.stack([ds[i][1] for i in range(8)])
    model = ReportGenerator(vocab_size=ds.vocab_size, in_channels=ds.channels)
    logits = model(images, captions)
    # Teacher forcing consumes captions[:, :-1], so the time dimension is
    # seq_len - 1, and the last dim is the vocabulary size.
    assert logits.shape == (8, ds.seq_len - 1, ds.vocab_size)


def test_dataset_consistency_per_class():
    # Same class must yield the same caption across samples.
    ds = SyntheticReportDataset(num_samples=8, num_classes=4)
    _, cap0 = ds[0]
    _, cap4 = ds[4]  # index 4 wraps to class 0
    assert torch.equal(cap0, cap4)
