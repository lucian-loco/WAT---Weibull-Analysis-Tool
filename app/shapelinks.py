from dataclasses import dataclass

@dataclass
class ShapeLink:
    libraryName: str
    shapeName: str


# Map module type name to a Draw.IO shape from a template library
mapping = {
    # Modules VME
    'HCCTRV_': ShapeLink('Modules VME', 'CTRV'),
    'HCCVBWA': ShapeLink('Modules VME', 'WR2RF'),
    'HCCVOIA': ShapeLink('Modules VME', 'VD80'),
    'HCCVOPE': ShapeLink('Modules VME', 'SIS3300'),
    'HCCVORA': ShapeLink('Modules VME', 'CVORA'),
    'HCCVORI': ShapeLink('Modules VME', 'CVORI'),
    'HCCVUNB': ShapeLink('Modules VME', 'CPU MEN A20'),
    'HCCVUNC': ShapeLink('Modules VME', 'CPU MEN A25'),
    'HCCTDAP': ShapeLink('Modules VME', 'Conv-TTL-RS485'),
    'HCCVORL': ShapeLink('Modules VME', 'CTLEI'),
    #'': ShapeLink('Modules VME', 'VMOD'),
}
