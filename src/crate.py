#!/usr/bin/python3
from dataclasses import dataclass
import ccda
import io
import glob
import os
import copy
import functools
from enum import Enum
import drawio
import db_hitdata
import shapelinks

import logging
logger = logging.getLogger(__name__)


class Orientation(Enum):
    HORIZONTAL = 1
    VERTICAL = 2


class Face(Enum):
    FRONT = 1
    BACK = 2


@dataclass
class Dimension:
    start: float
    end: float


    def __post_init__(self):
        assert(self.start <= self.end)


    @property
    def length(self):
        return self.end - self.start


    def intersects(self, value):
        return self.start <= value and value <= self.end


@dataclass
class Module:
    position: str   # EAM position
    typeName: str
    typeCode: str
    slotNumber: int
    lun: int
    width: Dimension = None
    height: Dimension = None
    depth: Dimension = None


    @property
    def orientation(self):
        if self.width.length > self.height.length:
            return Orientation.HORIZONTAL
        else:
            return Orientation.VERTICAL


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
    widthDim: int = None    # number of uniform slots in the width dimension
    heightDim: int = None   # number of uniform slots in the height dimension
    depthDim: int = None    # number of uniform slots in the depth dimension


    def slot(self, index):
        """ Returns the module in the specified slot. """
        for m in self.modules:
            if m.slotNumber == index:
                return m

        return None


    def is_front_module(self, module: Module):
        """ Checks if the module is located at the front of the crate. """
        # the high depth position indicates the front
        return module.depth.intersects(self.depthDim)


    def is_back_module(self, module: Module):
        """ Checks if the module is located at the back of the crate. """
        # the low depth position indicates the back
        return module.depth.intersects(1)


    def is_module_face(self, module: Module, face: Face):
        if face == Face.FRONT:
            return self.is_front_module(module)
        elif face == Face.BACK:
            return self.is_back_module(module)
        else:
            raise ValueError('Invalid face')


@functools.lru_cache(32)
def get_crate_layout_data(equipment_code):
    cursor = db_hitdata.get_cursor()
    query = f"SELECT WIDTH, HEIGHT, DEPTH, WIDTH_DIM, HEIGHT_DIM, DEPTH_DIM " \
                "FROM crate_dimensions_v WHERE EQUIPMENT_CODE = :equipment_code"
    response = cursor.execute(query, (equipment_code,)).fetchall()

    if len(response) != 1:
        raise ValueError(f'Invalid equipment code: {equipment_code}')

    return {
        'width': response[0][0],
        'height': response[0][1],
        'depth': response[0][2],
        'width_dim': response[0][3],
        'height_dim': response[0][4],
        'depth_dim': response[0][5]
    }


def get_module_layout_data(position):
    cursor = db_hitdata.get_cursor()
    # TODO MODULE_DESCRIPTION? MODULE_EQUIPMENT_CODE?
    query = f"SELECT SLOT_NUMBER, " \
            "H_START_NUM, H_END_NUM, " \
            "W_START_NUM, W_END_NUM, " \
            "D_START_NUM, D_END_NUM " \
            "FROM module_positions_v WHERE MODULE_NAME = :position"
    response = cursor.execute(query, (position,)).fetchall()

    if len(response) != 1:
        raise ValueError(f'Invalid module position: {position}')

    return {
        'slot_number': response[0][0],
        'height': Dimension(response[0][1], response[0][2]),
        'width': Dimension(response[0][3], response[0][4]),
        'depth': Dimension(response[0][5], response[0][6]),
    }


def make_crate(crate):
    crate_ccde_data = ccda.crate_by_label(crate)

    # Remove 'HC' prefix from the equipment code
    assert(crate_ccde_data['typeCode'].startswith('HC'))
    crate_ccde_data['typeCode'] = crate_ccde_data['typeCode'][2:]

    crate_layout_data = get_crate_layout_data(crate_ccde_data['typeCode'])

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
            widthDim     = crate_layout_data['width_dim'],
            heightDim    = crate_layout_data['height_dim'],
            depthDim     = crate_layout_data['depth_dim'],

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
            module_layout_data = get_module_layout_data(module_ccde_data['name'])
            module.width      = module_layout_data['width']
            module.height     = module_layout_data['height']
            module.depth      = module_layout_data['depth']

            if module_ccde_data['slotNumber'] != module_layout_data['slot_number']:
                logger.warning('Slot number mismatch for %s (CCDE:%d / Layout:%d)',
                                module_ccde_data['name'], module_ccde_data['slotNumber'], module_layout_data['slot_number'])
        except ValueError as e:
            logger.warning('Could not find module layout data for %s', module_ccde_data['name'])

        crate.modules.append(module)

    return crate


def generate_graph(crate):
    # TODO do not hardcode slot dimensions
    SLOT_COUNT = 20
    SLOT_WIDTH = 13
    SLOT_HEIGHT = 157
    START_X = 0
    START_Y = 0
    SHAPE_OFFSET_X = -72
    SHAPE_OFFSET_Y = 72

    crate = make_crate(crate)

    generator.clear_page(320, 240)
    generator.add_box(10, 200, 200, 20, text=crate.name)
    box_style = drawio.Style(horizontal=False, direction='west')    # slot boxes (when missing graphics)
    slot_style = copy.deepcopy(box_style)                           # slot labels
    slot_style.apply(drawio.Style(fillColor='none', strokeColor='none', fontSize=7))
    shape_style = drawio.Style(rotation=90)                         # module graphics/shapes

    for slot_index in range(1, SLOT_COUNT + 1):
        module = crate.slot(slot_index)
        pos_x = START_X + slot_index * SLOT_WIDTH
        pos_y = START_Y

        if module is None:
            # Empty slot
            generator.add_box(pos_x, pos_y, SLOT_WIDTH, SLOT_HEIGHT, style=box_style)

        else:
            try:
                link = shapelinks.mapping[module.typeCode]
                generator.add_shape(link.libraryName, link.shapeName,
                        pos_x + SHAPE_OFFSET_X, pos_y + SHAPE_OFFSET_Y, style=shape_style)
            except KeyError:
                # No shape available in the library
                generator.add_box(pos_x, pos_y, SLOT_WIDTH, SLOT_HEIGHT,
                        text=module.typeName, style=box_style)

        # Slot label
        generator.add_box(pos_x, pos_y + SLOT_HEIGHT, SLOT_WIDTH, 30,
                text=f'Slot {slot_index}', style=slot_style)

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

    crate = make_crate('cfc-774-cdv35')
    pprint.pprint(crate)

    #graph = generate_graph('cfv-774-cdv03')
    #print(graph.readlines())