#!/usr/bin/python3
from dataclasses import dataclass
import json
import requests
import io
import glob
import os
import copy
import drawio
import shapelinks


CCDA_API_URL='https://ccda.cern.ch:8900/api'

@dataclass
class Module:
    typeName: str
    typeCode: str
    slotNumber: int
    lun: int


@dataclass
class Crate:
    name: str
    typeName: str
    typeCode: str
    locationName: str
    rackName: str
    modules: list[Module]

    def slot(self, index):
        for m in self.modules:
            if m.slotNumber == index:
                return m

        return None


@dataclass
class CrateLink:
    libraryName: str
    shapeName: str
    slotWidth: int
    firstSlotX: int
    firstSlotY: int


def get_ccde_crate(crate):
    s = requests.session()
    request = s.get(f'{CCDA_API_URL}/crates/search?query=label%3D%3D{crate}', verify=False)
    response = json.loads(request.text)

    if response['totalElements'] <= 0:
        raise RuntimeError('Invalid crate name')

    if response['totalElements'] > 1:
        raise RuntimeError('Too many results')

    crate_data = response['content'][0]
    crate = Crate(
            name         = crate_data['label'],
            typeName     = crate_data['typeName'],
            typeCode     = crate_data['typeCode'],
            locationName = crate_data['locationName'],
            rackName     = crate_data['rackName'],
            modules      = [])

    for module_data in crate_data['modules']:
        crate.modules.append(Module(
            typeName   = module_data['typeName'],
            typeCode   = module_data['typeCode'],
            slotNumber = module_data['slotNumber'],
            lun        = module_data['lun']))

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

    crate = get_ccde_crate(crate)

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

for filepath in glob.glob(os.path.join('../drawio', "*.xml")):
    generator.load_library(filepath)


if __name__ == '__main__':
    #crate = get_ccde_crate('cfv-193-ascool')
    #print(crate)

    graph = generate_graph('cfv-774-cdv03')
    print(graph.readlines())
