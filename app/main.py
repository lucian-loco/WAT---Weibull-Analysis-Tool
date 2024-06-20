#!/usr/bin/python3
import weibull
import crate
from flask import Flask, send_file, request

app = Flask(__name__)


@app.route('/weibull', methods=['GET'])
def route_weibull():
    try:
        part = request.args.get('part')
        graph = weibull.generate_graph(part)
    except RuntimeError as e:
        return 'Weibull plot cannot be generated: ' + str(e), 400

    return send_file(graph, mimetype='image/png')


@app.route('/crate', methods=['GET'])
def route_crate():
    try:
        crate_name = request.args.get('name')
        graph = crate.generate_graph(crate_name)
    except RuntimeError as e:
        return 'Crate layout cannot be generated: ' + str(e), 400

    return send_file(graph, mimetype='text/drawio', download_name=f'{crate_name}.drawio')


if __name__ == '__main__':
    app.run(debug=True, port=8888, host='0.0.0.0', threaded=True)
