#/usr/bin/env python
from drawio import Style

# Common styles applied to the generated shapes
# Crate outline
crate_outline = Style(
    fillColor='none',
    strokeColor='black',
    strokeWidth=1)

# Slot boxes (when missing shape graphics)
box = Style(
    fillColor='#dddddd',
    horizontal=False,
    direction='west')

# Slot number labels
slot = box.copy()
slot.fillColor.set('none')
slot.strokeColor.set('none')
slot.fontSize.set(7)

# Horizontal module shapes
shape_h = Style(
    aspect='variable',
    imageAspect=False)

# Vertical module shapes (differ only by the rotation;
# shapes in the library are normally horizontal, so we need to rotate them to make them vertical)
shape_v = shape_h.copy()
shape_v.rotation.set(90)