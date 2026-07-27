import re

import numpy as np


def parse_color(s: str):
    regex = re.compile(r"^(#|0x)([0-9a-fA-F]{6})$")
    if not regex.match(s):
        raise ValueError("Color must be in the format '#RRGGBB' or '0xRRGGBB'")
    try:
        int(s.lstrip("#").lstrip("0x"), 16)
    except ValueError:
        raise ValueError("Color must be a valid hex color")
    val = int(s.lstrip("#").lstrip("0x"), 16)
    return np.array([(val >> 16) & 0xFF, (val >> 8) & 0xFF, val & 0xFF], dtype=np.uint8)
