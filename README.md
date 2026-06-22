# radiology-report-generator

A small image to text model that takes a radiograph style image and generates a
short impression. The architecture is the classic captioning recipe kept
deliberately tiny: a convolutional encoder squeezes the image into one context
vector, and a GRU decoder turns that vector into a sequence of tokens. Training
uses teacher forcing, and inference uses greedy decoding.

This repo is built around synthetic data so it runs anywhere on CPU in a few
seconds with no downloads. The goal is a clean, tested reference for how an
encoder and a decoder are wired together for sequence generation, not a clinical
tool.

## How it works

The encoder is two convolution blocks followed by global average pooling and a
linear projection, giving a single vector per image. The decoder is a GRU whose
initial hidden state is that image vector. At each step the decoder embeds the
previous token, advances the GRU, and projects to vocabulary logits.

During training we feed the ground truth previous token at every position, which
is teacher forcing. The loss is cross entropy over the next token, with padding
positions ignored. At inference the decoder feeds itself: it starts from the
beginning of sequence token and takes its own argmax output as the next input,
emitting a fixed number of tokens.

## Synthetic data

`SyntheticReportDataset` builds image and caption pairs from a hidden class. Each
class owns one image prototype and one fixed caption. A sample is the prototype
plus a little noise, paired with that class caption wrapped in beginning and end
of sequence markers. Because the mapping from image to caption is consistent, a
correctly wired model can learn it, and the loss falling on a tiny set is real
evidence that the gradient path runs end to end.

## Layout

```
src/
  data.py    synthetic dataset and special token ids
  model.py   CNN encoder, GRU decoder, full ReportGenerator
  train.py   teacher forcing loss and a short overfit loop
tests/
  test_shapes.py     forward shapes of encoder, decoder, full model
  test_training.py   loss is finite and falls when overfitting
  test_decoding.py   greedy decode length and caption recovery
```

## Running the tests

```
python -m pytest tests/ -q
```

The suite checks behavior rather than fixed numbers. It confirms the forward
shapes line up, the teacher forcing loss drops sharply when overfitting a tiny
consistent batch, greedy decoding returns a sequence of the requested length,
and after overfitting the decoded tokens match the target caption. On a recent
CPU run all eight tests passed in about three seconds.

## Notes

The model is intentionally small so tests stay fast and deterministic. Swapping
the encoder for a pretrained backbone or the GRU for a transformer decoder would
be the natural next step toward something closer to a real report generator, but
the training and decoding contract stays the same.
