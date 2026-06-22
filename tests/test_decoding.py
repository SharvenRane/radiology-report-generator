import torch

from src.data import SyntheticReportDataset, BOS
from src.model import ReportGenerator
from src.train import overfit


def test_greedy_decode_length():
    torch.manual_seed(0)
    ds = SyntheticReportDataset(num_samples=8, vocab_size=16, caption_len=4)
    images = torch.stack([ds[i][0] for i in range(4)])
    model = ReportGenerator(vocab_size=ds.vocab_size, in_channels=ds.channels)
    max_len = 6
    seqs = model.generate(images, max_len=max_len, bos=BOS)
    # One row per input image, exactly max_len tokens each.
    assert seqs.shape == (4, max_len)
    assert seqs.dtype == torch.long


def test_greedy_decode_recovers_caption_after_overfit():
    torch.manual_seed(0)
    ds = SyntheticReportDataset(
        num_samples=8, num_classes=2, vocab_size=16, caption_len=4, seed=1
    )
    images = torch.stack([ds[i][0] for i in range(len(ds))])
    captions = torch.stack([ds[i][1] for i in range(len(ds))])
    model = ReportGenerator(vocab_size=ds.vocab_size, in_channels=ds.channels)
    overfit(model, images, captions, steps=300, lr=1e-2)

    # After overfitting, greedy decoding from each image should reproduce the
    # ground truth caption content (the tokens after <bos>).
    target = captions[:, 1:]
    gen = model.generate(images, max_len=target.size(1), bos=BOS)
    assert gen.shape == target.shape
    match_rate = (gen == target).float().mean().item()
    assert match_rate > 0.9
