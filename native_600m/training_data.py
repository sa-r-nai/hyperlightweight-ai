"""Memory-mapped next-token windows and reproducible sample addressing."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path


def sample_index(sample: int, count: int, seed: int) -> int:
    """Permute each epoch without keeping a large Python index list or RNG cursor."""
    if sample < 0 or count <= 0:
        raise ValueError("Invalid sample index or dataset length.")
    epoch, position = divmod(sample, count)
    digest = hashlib.sha256(f"{seed}:{epoch}".encode("ascii")).digest()
    multiplier = int.from_bytes(digest[:8], "big") % count or 1
    while math.gcd(multiplier, count) != 1:
        multiplier += 1
    offset = int.from_bytes(digest[8:16], "big") % count
    return (position * multiplier + offset) % count


class TokenWindows:
    def __init__(self, root: Path, entry: dict, seq_len: int):
        import numpy as np

        if seq_len < 2:
            raise ValueError("seq_len must be at least two.")
        self.seq_len = seq_len
        self.tokens = np.memmap(root / entry["tokens_file"], dtype="<u2", mode="r")
        self.mask = None
        if entry.get("mask_file"):
            self.mask = np.memmap(root / entry["mask_file"], dtype="u1", mode="r")
        block_count = (len(self.tokens) - 1) // seq_len
        if block_count <= 0:
            raise ValueError("Token file is shorter than one complete next-token window.")
        # Only SFT windows with at least one supervised NEXT token are usable.
        self.valid_blocks = None
        if self.mask is not None:
            selected = []
            for block in range(block_count):
                start = block * seq_len
                if self.mask[start + 1:start + seq_len + 1].any():
                    selected.append(block)
            self.valid_blocks = np.asarray(selected, dtype=np.int64)
        self.block_count = block_count if self.valid_blocks is None else len(self.valid_blocks)
        if self.block_count == 0:
            raise ValueError("No complete windows contain supervised assistant tokens.")

    def __len__(self):
        return self.block_count

    def __getitem__(self, index):
        import numpy as np
        import torch

        if not 0 <= index < self.block_count:
            raise IndexError(index)
        block = index if self.valid_blocks is None else int(self.valid_blocks[index])
        start = block * self.seq_len
        values = np.array(self.tokens[start:start + self.seq_len + 1], dtype=np.int64)
        x = torch.from_numpy(values[:-1].copy())
        y = torch.from_numpy(values[1:].copy())
        if self.mask is not None:
            supervised = np.array(self.mask[start + 1:start + self.seq_len + 1], dtype=bool)
            y[torch.from_numpy(~supervised)] = -100
        return x, y

    def batch(self, first_sample: int, batch_size: int, seed: int):
        import torch

        rows = [self[sample_index(first_sample + offset, len(self), seed)] for offset in range(batch_size)]
        return torch.stack([row[0] for row in rows]), torch.stack([row[1] for row in rows])
