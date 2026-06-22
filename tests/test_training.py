import torch

from src.data import SyntheticReportDataset
from src.model import ReportGenerator
from src.train import teacher_forcing_loss, overfit


def _tiny_batch(seed=0):
    ds = SyntheticReportDataset(
        num_samples=8,
        num_classes=2,
        image_size=16,
        vocab_size=16,
        caption_len=4,
        seed=seed,
    )
    images = torch.stack([ds[i][0] for i in range(len(ds))])
    captions = torch.stack([ds[i][1] for i in range(len(ds))])
    return ds, images, captions


def test_loss_is_finite_and_positive():
    torch.manual_seed(0)
    ds, images, captions = _tiny_batch()
    model = ReportGenerator(vocab_size=ds.vocab_size, in_channels=ds.channels)
    loss = teacher_forcing_loss(model, images, captions)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_teacher_forcing_loss_decreases_on_overfit():
    torch.manual_seed(0)
    ds, images, captions = _tiny_batch()
    model = ReportGenerator(vocab_size=ds.vocab_size, in_channels=ds.channels)
    losses = overfit(model, images, captions, steps=200, lr=1e-2)
    # The final loss should be well below where it started when overfitting a
    # tiny consistent set.
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.5 * losses[0]
    assert losses[-1] < 0.1
