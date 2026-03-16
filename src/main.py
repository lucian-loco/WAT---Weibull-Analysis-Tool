#!/usr/bin/env python
import weibull
import crate
import layout
import drawio
import os
import urllib.parse
import functools
import io
import requests
from flask import Flask, render_template, request, send_file, url_for, redirect
from markupsafe import escape

app = Flask(__name__)
drawio_export_server = os.environ.get('DRAWIO_EXPORT_URL', '')

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


def get_mimetype(format):
    if format == 'drawio':
        return 'text/drawio'
    elif format == 'pdf':
        return 'application/pdf'
    elif format == 'png':
        return 'image/png'
    else:
        raise RuntimeError('Unsupported format: ' + format)


def export_drawio(document, format):
    if format not in ('pdf', 'png', 'drawio'):
        raise RuntimeError('Unsupported format: ' + format)

    if format == 'drawio':  # nothing to do, just return the document as is
        return io.BytesIO(document.encode('utf-8'))

    if drawio_export_server == '':
        raise RuntimeError('DRAWIO_EXPORT_URL is not configured (required for exporting to pdf/png/svg)')

    response = requests.post(
        drawio_export_server,
        data = {'format': format, 'xml': document}
    )

    response.raise_for_status()
    return io.BytesIO(response.content)


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


@app.route('/weibull_new', methods=['GET', 'POST'])
def route_weibull_new():
    part = request.args.get('part')

    if not part:
        return 'Parameter "part" not valid or is missing', 400

    errors = {}

    if request.method == 'POST':
        sb, err = weibull.validate_sort_by(request.form.get('sort_by', ''))
        if err: errors['sort_by'] = err

        ci, err = weibull.validate_ci(request.form.get('ci', ''))
        if err: errors['ci'] = err

        if not errors:
            try:
                graph = weibull.generate_graph_new(part=part, sort_by=sb, ci=ci)
            except RuntimeError as e:
                return 'Weibull plot cannot be generated: ' + str(e), 400

            return send_file(graph, mimetype='image/png')

    return render_template('weibull_form.html', part=part, errors=errors, defaults={'sort_by': 'BIC', 'ci': 0.95})


@app.route('/crate/new')
def route_crate_new():
    drawio_libs = get_drawio_lib_urls()
    return render_template('drawio.html', drawio_libs=drawio_libs)


def make_crate_graph(args, scale=None, max_size=None):
    crate_name = args.get('name', None)
    crate_id = args.get('id', None)

    if crate_name is None and crate_id is None:
        return None

    if crate_name is not None and crate_id is not None:
        raise RuntimeError('Either crate name or crate ID should be provided, not both')

    if crate_name is not None:
        crate_id = layout.crate_name_to_crate_id(crate_name)

    version = args.get('version', 'TODAY')
    face = layout.Face.from_str(args.get('face', 'front'))
    return crate.generate_graph_crate(crate_id, version, face, scale, max_size)


def get_crate_name(args):
    crate_name = args.get('name', None)
    crate_id = args.get('id', None)

    if crate_name is None and crate_id is None:
        return None

    if crate_name is not None and crate_id is not None:
        raise RuntimeError('Either crate name or crate ID should be provided, not both')

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
    except (RuntimeError, ValueError) as e:
        return 'Crate layout cannot be generated: ' + str(e), 400

    return render_template('drawio.html', document_data=document, drawio_libs=drawio_libs)


@app.route('/crate/get', methods=['GET'])
def route_crate_get():
    try:
        # crate_name = get_crate_name(request.args)
        scale = request.args.get('scale', None, type=float)
        max_size = request.args.get('maxsize', None, type=float)
        graph = make_crate_graph(request.args, scale, max_size)
        format = request.args.get('format', 'drawio')

        if graph is None:
            raise RuntimeError('Crate name (position) or ID must be provided')

        document = graph.getvalue().decode('utf-8')

    except (RuntimeError, ValueError) as e:
        return 'Crate layout cannot be generated: ' + str(e), 400

    return send_file(export_drawio(document, format), mimetype=get_mimetype(format))


@app.route('/crate/stencil/get', methods=['GET'])
def route_crate_stencil_get():
    try:
        format = 'png'
        part = request.args.get('part')
        scale = request.args.get('scale', None, type=float)
        max_size = request.args.get('maxsize', None, type=float)
        graph = crate.generate_graph_stencil(part, scale, max_size)

        if graph is None:   # no crate name/ID specified, show an empty draw.io editor
            return render_template('drawio.html', drawio_libs=())

        document = graph.getvalue().decode('utf-8')

    except (RuntimeError, ValueError) as e:
        return 'Stencil cannot be shown: ' + str(e), 400

    return send_file(export_drawio(document, format), mimetype=get_mimetype(format))


@app.route('/whiterabbit')
def route_wr_main():
    return redirect("https://mondi.app.cern.ch/whiterabbit", code=302)


@app.route('/whiterabbit/network', methods=['GET'])
def route_wr_network():
    return redirect("https://mondi.app.cern.ch/whiterabbit/network", code=302)


if __name__ == '__main__':
    app.run(debug=True, port=8888, host='0.0.0.0', threaded=True)
