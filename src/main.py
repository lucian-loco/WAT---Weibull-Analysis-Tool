#!/usr/bin/env python
import weibull
import crate
import layout
import os
import urllib.parse
import functools
from flask import Flask, render_template, request, send_file, url_for
from markupsafe import escape

app = Flask(__name__)


# TODO not very elegant, think of a better way
@functools.cache
def get_drawio_lib_urls():
    """ Generate a list of draw.io libraries URL, encoded for use in draw.io editor """
    drawio_lib_urls = []

    try:
        drawio_lib_path = 'drawio'
        base_url = os.environ['DRAWIO_LIBS_URL']

        for filename in os.listdir(drawio_lib_path):
            if os.path.isfile(os.path.join(drawio_lib_path, filename)):
                url = f'{base_url}/{filename}'
                encoded = urllib.parse.quote(url, safe='')
                drawio_lib_urls.append(encoded)
    except:
        pass    # no draw.io libraries, it is a pity, but not a showstopper

    return drawio_lib_urls


@app.route('/')
def route_main():
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
   return app.send_static_file('images/favicon.ico')


@app.route('/weibull', methods=['GET'])
def route_weibull():
    try:
        part = request.args.get('part')
        graph = weibull.generate_graph(part)
    except RuntimeError as e:
        return 'Weibull plot cannot be generated: ' + str(e), 400

    return send_file(graph, mimetype='image/png')


@app.route('/crate/new')
def route_crate_new():
    drawio_libs = get_drawio_lib_urls()
    return render_template('drawio.html', drawio_libs=drawio_libs)


def make_crate_graph(args):
    crate_name = args.get('name', None)
    crate_id = args.get('id', None)

    if crate_name is None and crate_id is None:
        return None

    if crate_name is not None and crate_id is not None:
        raise RuntimeError('Only one of crate name or ID can be provided')

    if crate_name is not None:
        crate_id = layout.crate_name_to_crate_id(crate_name)

    version = args.get('version', 'TODAY')
    face = layout.Face.from_str(args.get('face', 'front'))
    return crate.generate_graph(crate_id, version, face)


def get_crate_name(args):
    crate_name = args.get('name', None)
    crate_id = args.get('id', None)

    if crate_name is None and crate_id is None:
        return None

    if crate_name is not None and crate_id is not None:
        raise RuntimeError('Only one of crate name or ID can be provided')

    if crate_name is not None:
        return crate_name

    version = args.get('version', 'TODAY')
    return layout.crate_id_to_crate_name(crate_id, version)


@app.route('/crate/edit', methods=['GET'])
def route_crate_edit():
    drawio_libs = get_drawio_lib_urls()

    try:
        graph = make_crate_graph(request.args)

        if graph is None:   # no crate name/ID specified, show an empty draw.io editor
            return render_template('drawio.html', drawio_libs=drawio_libs)

        document = graph.getvalue().decode('utf-8')
    except RuntimeError as e:
        return 'Crate layout cannot be generated: ' + str(e), 400

    return render_template('drawio.html', document_data=document, drawio_libs=drawio_libs)


@app.route('/crate/get', methods=['GET'])
def route_crate_get():
    try:
        graph = make_crate_graph(request.args)

        if graph is None:
            raise RuntimeError('Crate name or ID must be provided')
    except RuntimeError as e:
        return 'Crate layout cannot be generated: ' + str(e), 400

    return send_file(graph, mimetype='text/drawio',
                     as_attachment=True, download_name=f'{crate_name}.drawio')


@app.route('/whiterabbit')
def route_wr_main():
    return render_template('whiterabbit/main.html')


@app.route('/whiterabbit/network', methods=['GET'])
def route_wr_network():
    source = request.args.get('source', 'ptp')
    return render_template('whiterabbit/network.html', connectivity_source=source)


@app.route('/whiterabbit/switch/<switch_name>')
def route_wr_switch(switch_name):
    return render_template('whiterabbit/switch.html', switch_name=escape(switch_name))


@app.route('/whiterabbit/fiber/<int:source>/<int:destination>')
def route_wr_fiber(source, destination):
    return render_template('whiterabbit/fiber.html', source=source, destination=destination)


@app.route('/whiterabbit/connections/<source>')
def route_wr_connections(source):
    return send_file(f'whiterabbit/data/connectivity_{source}.json')


if __name__ == '__main__':
    app.run(debug=True, port=8888, host='0.0.0.0', threaded=True)
