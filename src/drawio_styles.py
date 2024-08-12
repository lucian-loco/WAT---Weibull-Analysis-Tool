#/usr/bin/env python
import drawio

# Common styles applied to the generated shapes
# Crate outline
crate_outline_style = drawio.Style(
    fillColor='none',
    strokeColor='black',
    strokeWidth=1)

# Slot boxes (when missing shape graphics)
box_style = drawio.Style(
    fillColor='#dddddd',
    horizontal=False,
    direction='west')

# Slot number labels
slot_style = box_style.copy()
slot_style.fillColor.set('none')
slot_style.strokeColor.set('none')
slot_style.fontSize.set(7)

# Horizontal module shapes
shape_h_style = drawio.Style(
    aspect='variable',
    imageAspect=False)

# Vertical module shapes (differ only by the rotation;
# shapes in the library are normally horizontal, so we need to rotate them to make them vertical)
shape_v_style = shape_h_style.copy()
shape_v_style.rotation.set(90)