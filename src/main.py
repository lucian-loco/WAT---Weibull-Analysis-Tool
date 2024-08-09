#!/usr/bin/env python
import weibull
import crate
import os
import urllib.parse
import functools
from flask import Flask, render_template, request, send_file, url_for

app = Flask(__name__)


# TODO not very elegant, think of a better way
@functools.cache
def get_drawio_lib_urls():
    """ Generate a list of draw.io libraries URL, encoded for use in draw.io editor """
    drawio_lib_urls = []
    drawio_lib_path = 'static/drawio'
    base_url = url_for('static', filename=f'drawio', _external=True)

    for filename in os.listdir(drawio_lib_path):
        if os.path.isfile(os.path.join(drawio_lib_path, filename)):
            url = f'{base_url}/{filename}'
            encoded = urllib.parse.quote(url, safe='')
            drawio_lib_urls.append(encoded)

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


@app.route('/crate/edit', methods=['GET'])
def route_crate_edit():
    try:
        crate_name = request.args.get('name')
        graph = crate.generate_graph(crate_name)
        drawio_libs = get_drawio_lib_urls()
    except RuntimeError as e:
        return 'Crate layout cannot be generated: ' + str(e), 400

    document = graph.getvalue().decode('utf-8')
    return render_template('drawio.html', document_data=document, drawio_libs=drawio_libs)


@app.route('/crate/get', methods=['GET'])
def route_crate_get():
    try:
        crate_name = request.args.get('name')
        graph = crate.generate_graph(crate_name)
    except RuntimeError as e:
        return 'Crate layout cannot be generated: ' + str(e), 400

    return send_file(graph, mimetype='text/drawio',
                     as_attachment=True, download_name=f'{crate_name}.drawio')


if __name__ == '__main__':
    app.run(debug=True, port=8888, host='0.0.0.0', threaded=True)