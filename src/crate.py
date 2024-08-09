#!/usr/bin/env python
from dataclasses import dataclass
import ccda
import layout
import io
import glob
import os
import copy
import drawio
import shapelinks

import logging
logger = logging.getLogger(__name__)


# Constants defining the generated graph geometry
PAGE_WIDTH = 800
PAGE_HEIGHT = 600
TITLE_HEIGHT = 20


@dataclass
class Module:
    position: str   # EAM position
    typeName: str
    typeCode: str
    slotNumber: int
    lun: int

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
    name: str
    typeName: str
    typeCode: str
    locationName: str
    rackName: str
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
        # the highest depth coordinate indicates the front
        return module.depth.intersects(self.depthSlots)


    def is_back_module(self, module: Module):
        """ Checks if the module is located at the back of the crate. """
        # the lowest depth coordinate indicates the back
        return module.depth.intersects(1)


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


def make_crate(crate):
    crate_ccde_data = ccda.crate_by_label(crate)

    # Remove 'HC' prefix from the equipment code
    assert(crate_ccde_data['typeCode'].startswith('HC'))
    crate_ccde_data['typeCode'] = crate_ccde_data['typeCode'][2:]

    crate_layout_data = layout.get_crate(crate_ccde_data['typeCode'])

    crate = Crate(
            position     = crate_ccde_data['name'],
            name         = crate_ccde_data['label'],
            typeName     = crate_ccde_data['typeName'],
            typeCode     = crate_ccde_data['typeCode'],
            locationName = crate_ccde_data['locationName'],
            rackName     = crate_ccde_data['rackName'],

            width        = crate_layout_data['width'],
            height       = crate_layout_data['height'],
            depth        = crate_layout_data['depth'],
            widthSlots   = crate_layout_data['width_dim'],
            heightSlots  = crate_layout_data['height_dim'],
            depthSlots   = crate_layout_data['depth_dim'],

            modules      = [])

    for module_ccde_data in crate_ccde_data['modules']:
        # Remove 'HC' prefix from the equipment code
        assert(module_ccde_data['typeCode'].startswith('HC'))
        module_ccde_data['typeCode'] = module_ccde_data['typeCode'][2:]

        module = Module(
            position   = module_ccde_data['name'],
            typeName   = module_ccde_data['typeName'],
            typeCode   = module_ccde_data['typeCode'],
            slotNumber = module_ccde_data['slotNumber'],
            lun        = module_ccde_data['lun'])

        try:
            module_layout_data = layout.get_module(module_ccde_data['name'])
            module.width      = module_layout_data['width']
            module.height     = module_layout_data['height']
            module.depth      = module_layout_data['depth']

            if module_ccde_data['slotNumber'] != module_layout_data['slot_number']:
                logger.warning('Slot number mismatch for {0} (CCDE:{1} / Layout:{2})',
                                    module_ccde_data['name'],
                                    module_ccde_data['slotNumber'],
                                    module_layout_data['slot_number'])
        except ValueError as e:
            logger.warning('Could not find module layout data for %s', module_ccde_data['name'])

        crate.modules.append(module)

    return crate


def generate_graph(crate):
    logger.debug('Generating graph for %s', crate)

    # Styles applied to the generated shapes
    box_style = drawio.Style(horizontal=False, direction='west')    # slot boxes (when missing graphics)
    slot_style = copy.deepcopy(box_style)                           # slot labels
    slot_style.apply(drawio.Style(fillColor='none', strokeColor='none', fontSize=7))
    # Module shapes are normally horizontal, so the vertical ones need a rotation
    shape_h_style = drawio.Style(aspect='variable', imageAspect=False)                                  # horizontal module graphics/shapes
    shape_v_style = drawio.Style(rotation=90, aspect='variable', imageAspect=False)                       # vertical module graphics/shapes

    # Get crate & module data
    crate = make_crate(crate)

    generator.clear_page(PAGE_WIDTH, PAGE_HEIGHT)
    scale_hor = PAGE_WIDTH / crate.width
    scale_ver = (PAGE_HEIGHT - TITLE_HEIGHT) / crate.height
    scale = min(scale_hor, scale_ver)

    START_X = (PAGE_WIDTH - crate.width * scale) / 2  # center the crate layout
    START_Y = TITLE_HEIGHT  # draw the crate layout right below the title

    # Crate name label
    generator.add_box(0, 0, PAGE_WIDTH, TITLE_HEIGHT, text=crate.name.upper())

    # Draw the crate outline
    generator.add_box(START_X, START_Y, crate.width * scale, crate.height * scale)

    # Draw shapes/boxes representing the modules in each slot
    for module in crate.modules:
        if module.width is None or module.height is None or module.depth is None:
            logger.warning('No layout data for module %s', module.position)
            continue

        width = (module.width.length + 1) * crate.x_slot_size * scale
        height = (module.height.length + 1) * crate.y_slot_size * scale
        pos_x = START_X + module.width.start * crate.x_slot_size * scale - width / 2
        pos_y = START_Y + module.height.start * crate.y_slot_size * scale - height / 2

        if module.orientation == layout.Orientation.HORIZONTAL:
            shape_style = shape_h_style
        else:
            shape_style = shape_v_style
            (width, height) = (height, width)
            # Adjust the position after rotating the module
            # (to understand it better: rotate a shape in draw.io, while watching its X, Y position)
            size_diff = (height - width) / 2
            pos_x += size_diff
            pos_x -= height / 2 # yes, height / 2! since all modules are normally horizontal, height is the actual module width
            pos_y -= size_diff

        try:
            link = shapelinks.mapping[module.typeCode]

            generator.add_shape(link.libraryName, link.shapeName,
                    pos_x, pos_y, width=width, height=height, style=shape_style)
        except KeyError:
            # No shape available in the library
            generator.add_box(pos_x, pos_y, width, height,
                    text=module.typeName, style=shape_style)

        # Slot label
        # TODO where to put them? it is not smart to always put them below the crate, not every crate has horizontal slots
        # if module.slotNumber:
        #     generator.add_box(pos_x, START_Y + crate.height * scale, crate.x_slot_size, 50,
        #             text=f'{module.slotNumber}', style=slot_style)

    buffer = io.BytesIO(generator.as_string().encode('utf-8'))
    buffer.seek(0)
    return buffer


# Create the generator instance and load the templates once
generator = drawio.Generator()

for filepath in glob.glob(os.path.join('static/drawio', '*.xml')):
    try:
        generator.load_library(filepath)
    except:
        logger.error(f'Could not load library {filepath}')


if __name__ == '__main__':
    import pprint

    #data = get_crate_layout_data('CVREC')
    #print(data)

    #crate = make_crate('cfv-193-ascool')
    #pprint.pprint(crate)

    graph = generate_graph('cfv-193-ascool')
    print(graph.readlines())