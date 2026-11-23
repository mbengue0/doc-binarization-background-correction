"""Background correction + thresholding experiments for document binarization.

Conventions used everywhere in this package (checked against the official
DIBCO evaluation code, see third_party/DIBCO_metrics.m):
  * grayscale images are uint8 in [0, 255];
  * binary images are uint8 with TEXT = 0 (black) and BACKGROUND = 255 (white);
  * text is assumed darker than the background.
"""
