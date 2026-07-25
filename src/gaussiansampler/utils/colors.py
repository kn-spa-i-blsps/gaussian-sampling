import numpy as np


def parse_color(s: str):
    val = int(s.lstrip("#").lstrip("0x"), 16)
    return np.array([(val >> 16) & 0xFF, (val >> 8) & 0xFF, val & 0xFF], dtype=np.uint8)
