from dataclasses import dataclass

@dataclass
class ShapeLink:
    libraryName: str
    shapeName: str


# Map module type name to a Draw.IO shape from a template library
mapping = {
    # Modules VME
    'CTRV_': ShapeLink('Modules VME', 'CTRV'),
    'CVBWA': ShapeLink('Modules VME', 'WR2RF'),
    'CVOIA': ShapeLink('Modules VME', 'VD80'),
    'CVOPE': ShapeLink('Modules VME', 'SIS3300'),
    'CVORA': ShapeLink('Modules VME', 'CVORA'),
    'CVORI': ShapeLink('Modules VME', 'CVORI'),
    'CVUNB': ShapeLink('Modules VME', 'CPU MEN A20'),
    'CVUNC': ShapeLink('Modules VME', 'CPU MEN A25'),
    'CTDAP': ShapeLink('Modules VME', 'Conv-TTL-RS485'),
    'CVORL': ShapeLink('Modules VME', 'CTLEI'),
    #'': ShapeLink('Modules VME', 'VMOD'),
}
