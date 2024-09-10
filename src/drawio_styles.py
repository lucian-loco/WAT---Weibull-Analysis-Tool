#/usr/bin/env python
from drawio import Style

# Common styles applied to the generated shapes
# Crate outline
crate_outline = Style(
    fillColor='none',
    strokeColor='black',
    strokeWidth=1)

# Horizontal module shapes
shape_h = Style(
    aspect='variable',
    imageAspect=False)

# Vertical module shapes (differ only by the rotation)
shape_v = shape_h.copy()
shape_v.rotation.set(90)