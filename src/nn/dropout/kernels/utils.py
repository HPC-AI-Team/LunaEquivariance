"""Dropout launch dimensions and batch tiling."""

BLOCK = 256
VBLOCK = 128
FORWARD_BLOCK = 128
BATCH_THREADS = 128

PAIR_LOW_ROWS = 2
BATCH_ROWS = 4

MAX_DIMS = 64
