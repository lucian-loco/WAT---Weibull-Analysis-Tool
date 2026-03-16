#!/usr/bin/env python
from dataclasses import dataclass
import layout
import io
import glob
import os
import drawio
import drawio_styles

import logging
logger = logging.getLogger(__name__)


# Constants defining the generated graph geometry
PAGE_WIDTH = 800
PAGE_HEIGHT = 600
TITLE_HEIGHT = 20

DEFAULT_FONT_SIZE = 12


@dataclass
class Module:
    position: str   # EAM position
    description: str
    equipmentCode: str
    slotNumber: int

    # Layout data (position expressed in slots)
    width: layout.Slice = None
    height: layout.Slice = None
    depth: layout.Slice = None


    @property
    def orientation(self):
        if self.width.length > self.height.length:
            return layout.Orientation.HORIZONTAL
        else:
            return layout.Orientation.VERTICAL


@dataclass
class Crate:
    position: str   # EAM position
    fecName: str
    equipmentCode: str
    modules: list[Module]

    width: float = None     # width in meters
    height: float = None    # height in meters
    depth: float = None     # depth in meters
    widthSlots: int = None  # number of uniform slots in the width dimension
    heightSlots: int = None # number of uniform slots in the height dimension
    depthSlots: int = None  # number of uniform slots in the depth dimension


    def slot(self, index):
        """ Returns the module in the specified slot. """
        for m in self.modules:
            if m.slotNumber == index:
                return m

        return None


    def is_front_module(self, module: Module):
        """ Checks if the module is located at the front of the crate. """
        # the lowest depth coordinate indicates the front
        return module.depth.intersects(1)


    def is_back_module(self, module: Module):
        """ Checks if the module is located at the back of the crate. """
        # the highest depth coordinate indicates the back
        return module.depth.intersects(self.depthSlots)


    def is_module_face(self, module: Module, face: layout.Face):
        if face == layout.Face.FRONT:
            return self.is_front_module(module)
        elif face == layout.Face.BACK:
            return self.is_back_module(module)
        else:
            raise ValueError('Invalid face')


    @property
    def x_slot_size(self):
        # Returns size of each uniform slot in X dimension.
        return self.width / self.widthSlots


    @property
    def y_slot_size(self):
        # Returns size of each uniform slot in Y dimension.
        return self.height / self.heightSlots


    @property
    def z_slot_size(self):
        # Returns size of each uniform slot in Z dimension.
        return self.depth / self.depthSlots


    def x_slot_to_coord(self, slot):
        return slot * self.x_slot_size


    def y_slot_to_coord(self, slot):
        return slot * self.y_slot_size


    def z_slot_to_coord(self, slot):
        return slot * self.z_slot_size


def get_crate_data(crate_id, version='TODAY'):
    crate_data = layout.get_crate_data(crate_id)
    crate_dim_data = layout.get_crate_dims(crate_data['crate_equipment_code'])

    crate = Crate(
            position        = crate_data['crate_name'],
            fecName         = crate_data['fec_name'],
            equipmentCode   = crate_data['crate_equipment_code'],

            width           = crate_dim_data['width'],
            height          = crate_dim_data['height'],
            depth           = crate_dim_data['depth'],
            widthSlots      = crate_dim_data['width_dim'],
            heightSlots     = crate_dim_data['height_dim'],
            depthSlots      = crate_dim_data['depth_dim'],

            modules         = [])

    for module_data in layout.get_crate_modules(crate_id, version):
        module = Module(
            position        = module_data['name'],
            description     = module_data['description'],
            equipmentCode   = 'HC' + module_data['equipment_code'],
            slotNumber      = module_data['slot_number'],
            width           = module_data['width'],
            height          = module_data['height'],
            depth           = module_data['depth'])

        crate.modules.append(module)

    return crate


def generate_graph_crate(crate, version='TODAY', face=layout.Face.FRONT, scale=None, max_size=None):
    logger.debug('Generating crate graph for %s', crate)

    if scale is not None and max_size is not None:
        logger.warning('Both scale and max_size parameters are provided, max_size will be ignored')
        max_size = None
    elif scale is None and max_size is None:
        scale = 1.0

    if max_size is not None:
        # Fit the rendered page to a square of side max_size.
        scale = min(max_size / PAGE_WIDTH, max_size / PAGE_HEIGHT)

    if scale is None or scale <= 0:
        raise ValueError('Scale must be greater than 0')

    # Get crate & module data
    crate = get_crate_data(crate, version)

    page_width = PAGE_WIDTH * scale
    page_height = PAGE_HEIGHT * scale
    title_height = TITLE_HEIGHT * scale

    generator.clear_page(page_width, page_height)
    scale_hor = page_width / crate.width
    scale_ver = (page_height - title_height) / crate.height
    layout_scale = min(scale_hor, scale_ver)

    start_x = (page_width - crate.width * layout_scale) / 2  # center the crate layout
    start_y = title_height  # draw the crate layout right below the title

    # Crate name label
    generator.add_box(0, 0, page_width, title_height, text=crate.position.upper(),
                       style=drawio.Style(fontSize=scale * DEFAULT_FONT_SIZE, fontStyle=drawio.FontStyle.BOLD))

    # Draw the crate outline
    generator.add_box(start_x, start_y, crate.width * layout_scale, crate.height * layout_scale, style=drawio_styles.crate_outline)

    # Draw shapes/boxes representing the modules in each slot
    for module in crate.modules:
        if module.width is None or module.height is None or module.depth is None:
            logger.warning('No layout data for module %s', module.position)
            continue

        if not crate.is_module_face(module, face):
            continue        # process modules visible only on the specified face (front/back)


        # Calculate the shape box coordinates
        scaled_width = module.width.length * crate.x_slot_size * layout_scale
        scaled_height = module.height.length * crate.y_slot_size * layout_scale

        pos_x = start_x + (module.width.start - 1) * crate.x_slot_size * layout_scale

        # For the back face, use mirror transformation (revert X position)
        if face == layout.Face.BACK:
            pos_x = page_width - pos_x - scaled_width

        # In LayoutDB Y=0 is at the bottom, in Draw.io Y=0 is at the top, so revert the Y position
        pos_y = start_y + (crate.heightSlots - module.height.end) * crate.y_slot_size * layout_scale


        # Find the appropriate shape, if available
        library_name = generator.find_shape_library(module.equipmentCode)

        # Determine the shape orientation
        try:
            shape = generator.get_shape(library_name[0], module.equipmentCode)
            shape_horizontal = shape.is_horizontal
        except (KeyError, IndexError) as e:
            # No shape available, there will be a box with description
            shape_horizontal = True

        # Determine the module orientation (in LayoutDB)
        module_horizontal = (module.orientation == layout.Orientation.HORIZONTAL)


        # Handle rotation of the module
        if module_horizontal:
            if shape_horizontal:
                shape_style = drawio_styles.shape_h.copy()
                rotate = False
            else:
                shape_style = drawio_styles.shape_v.copy()
                rotate = True
        else:
            if shape_horizontal:
                shape_style = drawio_styles.shape_v.copy()
                rotate = True
            else:
                shape_style = drawio_styles.shape_h
                rotate = False

        shape_style.fontSize.set(scale * DEFAULT_FONT_SIZE)  # scale the font size as well

        if rotate:
            (scaled_width, scaled_height) = (scaled_height, scaled_width)

            # Adjust the position after rotating the module
            # (to understand it better: rotate a shape in draw.io, while watching its X, Y position)
            size_diff = (scaled_height - scaled_width) / 2
            pos_x += size_diff
            pos_y -= size_diff


        if len(library_name) > 0:
            # Shape found in the library, use it
            generator.add_shape(library_name[0], module.equipmentCode,
                    pos_x, pos_y, width=scaled_width, height=scaled_height, style=shape_style)
        else:
            # No shape available in the library, create a simple box with a description instead
            generator.add_box(pos_x, pos_y, scaled_width, scaled_height,
                    text=module.description, style=shape_style)


    # Create a buffer which can be returned by flask
    buffer = io.BytesIO(generator.as_string().encode('utf-8'))
    buffer.seek(0)
    return buffer


def generate_graph_stencil(stencil, scale=None, max_size=None):
    logger.debug('Generating stencil graph for %s', stencil)

    if scale and max_size:
        logger.warning('Both scale and max_size parameters are provided, max_size will be ignored')
        max_size = None
    elif not scale and not max_size:
        scale = 1.0

    try:
        # Find the shape to get its dimensions
        library_name = generator.find_shape_library(stencil)
        shape = generator.get_shape(library_name[0], stencil)

        if max_size:
            # Calculate the scale to fit within the maximum size
            scale = min(max_size / shape.width, max_size / shape.height)

        generator.clear_page(shape.width * scale, shape.height * scale)
        generator.add_shape(library_name[0], stencil, 0, 0, shape.width * scale, shape.height * scale)
    except (KeyError, IndexError) as e:
        # No shape available, create a simple box with a description instead
        generator.clear_page(200, 100)
        generator.add_box(0, 0, 200, 100,
                text=f'Stencil "{stencil}" is not available')

    # Create a buffer which can be returned by flask
    buffer = io.BytesIO(generator.as_string().encode('utf-8'))
    buffer.seek(0)
    return buffer


# Create the generator instance and load the templates once
generator = drawio.Generator()

for filepath in glob.glob(os.path.join('drawio', '*.xml')):
    try:
        generator.load_library(filepath)
    except:
        logger.error(f'Could not load library {filepath}')


if __name__ == '__main__':
    import pprint

    #data = get_crate_layout_data('CVREC')
    #print(data)

    crate = get_crate_data('cfv-774-caos4')
    pprint.pprint(crate)

    #graph = generate_graph('cfv-193-ascool')
    #print(graph.readlines())
