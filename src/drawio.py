#!/usr/bin/python3
from xml.dom.minidom import Document, parseString
from dataclasses import dataclass
import json
import datetime
import string
import random
import html
import os
import copy


class StyleAttrAbstract:
    def __init__(self, name):
        self._name = name
        self._value = None

    def set(self, value):
        raise NotImplemented

    def get(self):
        return self._value

    def __str__(self):
        return f'{self._name}={self._value}'


class StyleAttrFlag(StyleAttrAbstract):
    def __init__(self, name):
        super(StyleAttrFlag, self).__init__(name)

    def set(self, value):
        self._value = True if value else None

    def __str__(self):
        return f'{self._name}' if self._value else None


class StyleAttrBool(StyleAttrAbstract):
    def __init__(self, name):
        super(StyleAttrBool, self).__init__(name)

    def set(self, value):
        if isinstance(value, str):
            if value.lower() in ('false', 'off', '0'):
                value = False
            elif value.lower() in ('true', 'on', '1'):
                value = True
            else:
                raise RuntimeError('Invalid boolean value for the style attribute')

        self._value = '1' if value else '0'


class StyleAttrInt(StyleAttrAbstract):
    def __init__(self, name):
        super(StyleAttrInt, self).__init__(name)

    def set(self, value):
        #assert(isinstance(value, int)) # TODO validation
        self._value = str(value)


class StyleAttrStr(StyleAttrAbstract):
    def __init__(self, name):
        super(StyleAttrStr, self).__init__(name)

    def set(self, value):
        self._value = str(value)


class StyleAttrEnum(StyleAttrAbstract):
    def __init__(self, name, allowed_values):
        super(StyleAttrEnum, self).__init__(name)
        self._allowed_values = allowed_values

    def set(self, value):
        if value not in self._allowed_values:
            raise RuntimeError('Invalid value for the style attribute')

        self._value = str(value)


class StyleAttrColor(StyleAttrAbstract):
    _HTML_COLORS = ("black", "navy", "darkblue", "mediumblue", "blue",
            "darkgreen", "green", "teal", "darkcyan", "deepskyblue",
            "darkturquoise", "mediumspringgreen", "lime", "springgreen",
            "aqua", "cyan", "midnightblue", "dodgerblue", "lightseagreen",
            "forestgreen", "seagreen", "darkslategray", "darkslategrey",
            "limegreen", "mediumseagreen", "turquoise", "royalblue",
            "steelblue", "darkslateblue", "mediumturquoise", "indigo",
            "darkolivegreen", "cadetblue", "cornflowerblue", "rebeccapurple",
            "mediumaquamarine", "dimgray", "dimgrey", "slateblue", "olivedrab",
            "slategray", "slategrey", "lightslategray", "lightslategrey",
            "mediumslateblue", "lawngreen", "chartreuse", "aquamarine",
            "maroon", "purple", "olive", "gray", "grey", "skyblue",
            "lightskyblue", "blueviolet", "darkred", "darkmagenta",
            "saddlebrown", "darkseagreen", "lightgreen", "mediumpurple",
            "darkviolet", "palegreen", "darkorchid", "yellowgreen", "sienna",
            "brown", "darkgray", "darkgrey", "lightblue", "greenyellow",
            "paleturquoise", "lightsteelblue", "powderblue", "firebrick",
            "darkgoldenrod", "mediumorchid", "rosybrown", "darkkhaki",
            "silver", "mediumvioletred", "indianred", "peru", "chocolate",
            "tan", "lightgray", "lightgrey", "thistle", "orchid", "goldenrod",
            "palevioletred", "crimson", "gainsboro", "plum", "burlywood",
            "lightcyan", "lavender", "darksalmon", "violet", "palegoldenrod",
            "lightcoral", "khaki", "aliceblue", "honeydew", "azure",
            "sandybrown", "wheat", "beige", "whitesmoke", "mintcream",
            "ghostwhite", "salmon", "antiquewhite", "linen",
            "lightgoldenrodyellow", "oldlace", "red", "fuchsia", "magenta",
            "deeppink", "orangered", "tomato", "hotpink", "coral",
            "darkorange", "lightsalmon", "orange", "lightpink", "pink", "gold",
            "peachpuff", "navajowhite", "moccasin", "bisque", "mistyrose",
            "blanchedalmond", "papayawhip", "lavenderblush", "seashell",
            "cornsilk", "lemonchiffon", "floralwhite", "snow", "yellow",
            "lightyellow", "ivory", "white")
    _KEYWORDS = ("default", "none", "swimlane", "inherit", "indicated")

    def __init__(self, name):
        super(StyleAttrColor, self).__init__(name)

    def set(self, value):
        if not value:
            raise RuntimeError('Invalid color value')

        value = str(value).lower()

        if value[0] == "#":      #RRGGBB color code
            if len(value) <= 1 or len(value) > 7:
                raise RuntimeError('Invalid color code')

            for c in value[1:]:
                if c not in '0123456789abcdef':
                    raise RuntimeError('Invalid color code')

        elif value not in self._HTML_COLORS and value not in self._KEYWORDS:
            raise RuntimeError('Invalid color value')

        self._value = str(value)


class Style:
    __valid_attributes = {
        'align': StyleAttrEnum('align', ('left', 'center', 'right')),
        'aspect': StyleAttrEnum('aspect', ('fixed', 'variable')),
        'direction': StyleAttrEnum('direction', ('north', 'south', 'east', 'west')),
        'fillColor': StyleAttrColor('fillColor'),
        'fontSize': StyleAttrInt('fontSize'),
        'horizontal': StyleAttrBool('horizontal'),
        'html': StyleAttrBool('html'),
        'image': StyleAttrStr('image'),
        'imageAspect': StyleAttrBool('imageAspect'),
        'labelBackground': StyleAttrColor('labelBackgroundColor'),
        'points': StyleAttrStr('points'),
        'rotation': StyleAttrInt('rotation'),
        'rounded': StyleAttrBool('rounded'),
        'shape': StyleAttrEnum('shape', ('image',)),
        'strokeColor': StyleAttrColor('strokeColor'),
        'text': StyleAttrFlag('text'),
        'verticalAlign': StyleAttrEnum('verticalAlign', ('top', 'middle', 'bottom')),
        'verticalLabelPosition': StyleAttrEnum('verticalLabelPosition', ('top', 'middle', 'bottom')),
        'whiteSpace': StyleAttrEnum('whiteSpace', ('wrap',)),
    }


    @staticmethod
    def __attribute_factory(name):
        if name not in Style.__valid_attributes:
            raise KeyError('Invalid style attribute')

        return copy.deepcopy(Style.__valid_attributes[name])


    def __init__(self, *args, **kwargs):
        self._attributes = dict()

        for k, v in kwargs.items():
            new_attr = Style.__attribute_factory(k)
            new_attr.set(v)
            self._attributes[k] = new_attr


    @staticmethod
    def from_style_str(style_str):
        new_style = Style()

        for k_v in style_str.split(';'):
            if not k_v:
                continue

            k_v_splitted = k_v.split('=')
            key = k_v_splitted[0]
            value = ''.join(k_v_splitted[1:])
            new_style.__getattr__(key).set(value)

        return new_style


    def apply(self, other):
            """
            Applies the attributes from another object to this object.
            If attributes exist in other object, but not in self - they are copied.
            If attributes exist in both self and other objects - other object overrides self.

            Args:
                other: Another object containing attributes to be applied.

            Returns:
                None
            """
            for name, attr in other._attributes.items():
                value = attr.get()

                if value is not None:
                    self.__getattr__(name).set(value)


    def copy(self):
        """ Returns a deep copy of the object. """
        return copy.deepcopy(self)


    def __copy__(self):
        new_obj = type(self)()
        new_obj._attributes.update(self._attributes)
        return new_obj


    def __deepcopy__(self, memo):
        new_obj = type(self)()
        memo[id(self)] = new_obj
        for k, v in self.__dict__.items():
            setattr(new_obj, k, copy.deepcopy(v, memo))
        return new_obj


    def __getattr__(self, item):
        if item not in self._attributes.keys() and item in Style.__valid_attributes:
            self._attributes[item] = Style.__attribute_factory(item)

        if item in self._attributes.keys():
            return self._attributes[item]

        raise AttributeError(item)


    def __str__(self):
        style = ''

        for attr in self._attributes.values():
            if attr.get() is None:
                continue

            style += f'{str(attr)};'

        return style


class Shape:
    def __init__(self, shape_dict):
        self.title = shape_dict['title'] if 'title' in shape_dict else 'Untitled'
        self.width = shape_dict['w']
        self.height = shape_dict['h']

        # TODO: handle other attributes specified in shape_dict?
        if 'xml' in shape_dict:
            self.style = Shape._xml_style(shape_dict)
        elif 'data' in shape_dict:
            self.style = Shape._data_style(shape_dict)
        else:
            raise RuntimeError('No graphic data in Shape')


    @staticmethod
    def _xml_style(shape_dict):
        xml_raw_data = html.unescape(shape_dict['xml'])

        xml_tree = parseString(xml_raw_data)
        # Reach the last mxCell, which actually describes the shape
        mxcell = xml_tree.firstChild.firstChild.childNodes[-1]
        return Style.from_style_str(mxcell.getAttribute('style'))


    @staticmethod
    def _data_style(shape_dict):
        # TODO all these are necessary?
        style = Style(shape='image',
            verticalLabelPosition='bottom',
            labelBackgroundColor='default',
            verticalAlign='top')
            #aspect='fixed',
            #imageAspect=False)

        # ';base64' must be removed for some reason when a shape is instantiated
        image = shape_dict['data'].replace(
                    'data:image/svg+xml;base64,',
                    'data:image/svg+xml,')

        if 'style' in shape_dict:
            image += ';' + shape_dict['style']

        style.image = image
        return style


class Library:
    def __init__(self, filename):
        """Loads all shapes from a template library"""
        self._shapes = {}

        with open(filename, 'r') as file:
            contents = file.read()

        # Strip <mxlibrary> tags, the rest is JSON
        contents = contents.replace('<mxlibrary>', '').replace('</mxlibrary>', '')

        for shape_data in json.loads(contents):
            if 'title' not in shape_data:
                shape_data['title'] = self._generate_title()

            new_shape = Shape(shape_data)
            self._shapes[new_shape.title] = new_shape


    def has_shape(self, name):
        return name in self._shapes.keys()


    def _make_style(self, name):
        """Generates 'style' attribute describing the shape. It contains the graphics too."""

        # TODO all these are necessary?
        style = Style(shape='image',
            verticalLabelPosition='bottom',
            labelBackgroundColor='default',
            verticalAlign='top',
            aspect='fixed',
            imageAspect=False)

        image = self._shapes[name]['data']

        if 'style' in self._shapes[name]:
            image += ';' + self._shapes[name]['style']

        style.image = image
        # TODO: handle other attributes specified in the library?

        return style


    def _generate_title(self):
        """Generates unique titles for shapes which do not specify them."""
        index = 0

        while True:
            title_candidate = f'Untitled-{index}'

            if title_candidate not in self._entries.keys():
                return title_candidate

            index += 1


class Generator:
    def __init__(self):
        self._libraries = {}
        self.clear_page(800, 600)


    def clear_page(self, width, height):
        self._page_width = width
        self._page_height = height
        self._uids = []
        self._document = Document()
        self._top = None    # top of the document tree
        self._root = None   # root container for graphics (mxCells)
        self._create_doc_structure()


    def save(self, filename):
        with open(filename, 'w') as xml_file:
            xml_file.write(self.as_string())


    def as_string(self):
        return self._top.toprettyxml(indent='  ')


    def load_library(self, filename):
        name = os.path.splitext(os.path.basename(filename))[0]  # extract the base file name (without extension)
        assert(name not in self._libraries.keys())
        self._libraries[name] = Library(filename)


    def get_shape(self, library_name, shape_name):
        library = self._libraries[library_name]
        return library._shapes[shape_name]


    def add_shape(self, library_name, shape_name, x, y, width=None, height=None, style=None):
        shape = self.get_shape(library_name, shape_name)

        if width is None:
            width = shape.width
        if height is None:
            height = shape.height

        geometry = self._create_geometry(x, y, width, height)
        shape_style = copy.deepcopy(shape.style)

        if style:
            shape_style.apply(style)

        self._create_cell(style=shape_style, geometry=geometry)


    def add_box(self, x, y, width, height, text='', style=Style()):
        geometry = self._create_geometry(x, y, width, height)
        self._create_cell(value=text, style=style, geometry=geometry)


    def _create_doc_structure(self):
        # Construct the base structure
        self._top = self._create_header()
        self._document.appendChild(self._top)

        diagram = self._create_diagram()
        self._top.appendChild(diagram)

        model = self._create_graphmodel()
        diagram.appendChild(model)

        # All graphical elements (mxCells) are under 'root'
        self._root = self._document.createElement('root')
        model.appendChild(self._root)

        # There are always two mxCells like this:
        mx1 = self._document.createElement('mxCell')
        mx1.setAttribute('id', '0')
        self._root.appendChild(mx1)

        mx2 = self._document.createElement('mxCell')
        mx2.setAttribute('id', '1')
        mx2.setAttribute('parent', '0')
        self._root.appendChild(mx2)


    def _create_header(self):
        header = self._document.createElement('mxfile')
        # TODO check which ones are really necessary
        header.setAttribute('host', 'Electron')
        timestamp = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        header.setAttribute('modified', timestamp)
        header.setAttribute('agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) draw.io/24.2.5 Chrome/120.0.6099.109 Electron/28.1.0 Safari/537.36')
        header.setAttribute('etag', self._generate_uid(20))
        header.setAttribute('version', '24.2.5')
        header.setAttribute('type', 'device')
        return header


    def _create_diagram(self, page_nr=1):
        diagram = self._document.createElement('diagram')
        diagram.setAttribute('name', f'Page-{page_nr}')
        diagram.setAttribute('id', self._generate_uid(20))
        return diagram


    def _create_graphmodel(self):
        graphmodel = self._document.createElement('mxGraphModel')
        graphmodel.setAttribute('dx', '64')
        graphmodel.setAttribute('dy', '64')
        graphmodel.setAttribute('gridSize', '10')
        graphmodel.setAttribute('pageScale', '1')
        graphmodel.setAttribute('pageWidth', str(self._page_width))
        graphmodel.setAttribute('pageHeight', str(self._page_height))

        # Options enabled
        for attr in ('grid', 'guides', 'tooltips', 'connect', 'arrows', 'fold', 'page'):
            graphmodel.setAttribute(attr, '1')

        # Options disabled
        for attr in ('math', 'shadow'):
            graphmodel.setAttribute(attr, '0')

        return graphmodel


    def _create_cell(self, parent=None, value='', vertex='1', style=Style(), geometry=None):
        if parent is None:  # if no parent specified, use the main cell
            parent = '1'

        cell = self._document.createElement('mxCell')
        cell.setAttribute('parent', str(parent))
        cell.setAttribute('id', self._generate_uid(14))
        cell.setAttribute('value', str(value))
        cell.setAttribute('vertex', str(vertex))
        cell.setAttribute('style', str(style))

        if geometry is not None:
            cell.appendChild(geometry)

        self._root.appendChild(cell)


    def _create_geometry(self, x, y, width, height):
        for i in (x, y, width, height):
            assert(isinstance(i, int) or isinstance(i, float))

        element = self._document.createElement('mxGeometry')
        element.setAttribute('x', str(x))
        element.setAttribute('y', str(y))
        element.setAttribute('width', str(width))
        element.setAttribute('height', str(height))
        element.setAttribute('as', 'geometry')
        return element


    def _generate_uid(self, length):
        """Generates a random string of the given length consisting of upper and lower case letters."""
        while True:
            uid = ''.join(random.choice(string.ascii_letters) for _ in range(length))

            if uid in self._uids:
                continue

            self._uids.append(uid)
            return uid


if __name__ == '__main__':
    generator = Generator()
    generator.load_library('../drawio/WFIP.xml')

    # Create a box with text inside, using custom style
    s = Style(whiteSpace='wrap', align='center')
    s.rounded.set(False)
    s.html.set(True)
    generator.add_box(10, 20, 100, 300, text='Test', style=s)

    # Create a modification of the previous style
    new_s = s.copy()
    new_s.rounded.set(True)
    new_s.fillColor.set('red')
    generator.add_box(110, 20, 100, 300, text='Test 2', style=new_s)

    s.apply(new_s)
    assert(s.fillColor.get() == 'red')

    # Append some shape from the template library
    generator.add_shape('WFIP', 'TAP BOX', 10, 10, style=Style(rotation=45))
    generator.add_shape('WFIP', 'TAP BOX', 20, 20, style=Style(rotation=-90))
    generator.add_shape('WFIP', 'TAP BOX', 30, 30)

    generator.save('test.drawio')