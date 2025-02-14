#!/usr/bin/env python
from dataclasses import dataclass
from enum import Enum
import functools
import db_hitdata


class Orientation(Enum):
    HORIZONTAL = 1
    VERTICAL = 2


class Face(Enum):
    FRONT = 1
    BACK = 2

    @staticmethod
    def from_str(value):
        if value.lower() == 'front':
            return Face.FRONT
        elif value.lower() == 'back':
            return Face.BACK
        else:
            raise ValueError(f'Invalid face value: {value}')


@dataclass
class Slice:
    start: int
    end: int


    def __post_init__(self):
        assert(self.start <= self.end)


    @property
    def length(self):
        """ Returns length of the slice (in slots). """
        return self.end - self.start + 1


    def intersects(self, value):
        return self.start <= value and value <= self.end


    def __str__(self):
        return f'{self.start}-{self.end}'


@functools.lru_cache(32)
def get_crate(equipment_code):
    if equipment_code.startswith('HC'):
        # Strip the 'HC' prefix
        equipment_code = equipment_code[2:]

    cursor = db_hitdata.get_cursor()
    query = "SELECT WIDTH, HEIGHT, DEPTH, WIDTH_DIM, HEIGHT_DIM, DEPTH_DIM " \
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


def get_module(position, version='TODAY'):
    cursor = db_hitdata.get_cursor()
    query = "SELECT MODULE_NAME, SLOT_NUMBER, " \
            "H_START_NUM, H_END_NUM, " \
            "W_START_NUM, W_END_NUM, " \
            "D_START_NUM, D_END_NUM " \
            "FROM MODULE_POSITIONS_DAYS_V WHERE MODULE_NAME = :position " \
            "AND VALID_FROM_DAY <= (SELECT MILESTONE_DAY FROM LAYOUT_MILESTONE_DAYS WHERE LABEL = :version) " \
            "AND EXPIRY_DAY > (SELECT MILESTONE_DAY FROM LAYOUT_MILESTONE_DAYS WHERE LABEL = :version)"
    response = cursor.execute(query, (position, version, version)).fetchall()

    if len(response) != 1:
        raise ValueError(f'Invalid module position: {position}')

    return {
        'name': response[0][0],
        'slot_number': response[0][1],
        'height': Slice(response[0][2], response[0][3]),
        'width': Slice(response[0][4], response[0][5]),
        'depth': Slice(response[0][6], response[0][7]),
    }


def get_fec_modules(fec_name, version='TODAY'):
    """ Returns a list of modules installed in a given FEC. """
    cursor = db_hitdata.get_cursor()
    query = "SELECT MODULE_NAME, SLOT_NUMBER, " \
            "MODULE_DESCRIPTION, MODULE_EQUIPMENT_CODE, "\
            "H_START_NUM, H_END_NUM, " \
            "W_START_NUM, W_END_NUM, " \
            "D_START_NUM, D_END_NUM " \
            "FROM MODULE_POSITIONS_DAYS_V WHERE UPPER(FEC_NAME) = :fec_name " \
            "AND VALID_FROM_DAY <= (SELECT MILESTONE_DAY FROM LAYOUT_MILESTONE_DAYS WHERE LABEL = :version) " \
            "AND EXPIRY_DAY > (SELECT MILESTONE_DAY FROM LAYOUT_MILESTONE_DAYS WHERE LABEL = :version)"
    response = cursor.execute(query, (fec_name.upper(), version, version)).fetchall()

    return [{
        'name': row[0],
        'slot_number': row[1],
        'description': row[2],
        'equipment_code': row[3],
        'height': Slice(row[4], row[5]),
        'width': Slice(row[6], row[7]),
        'depth': Slice(row[8], row[9]),
    } for row in response]