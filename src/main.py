#!/usr/bin/env python
import utils
import crate
import layout
import base64
import drawio
import os
import urllib.parse
import functools
import io
import requests
from datetime import date, timedelta
from flask import Flask, render_template, request, send_file, url_for, redirect
from apscheduler.schedulers.background import BackgroundScheduler
from weibull_forecast import forecast_part_direct_delta
from data_weibull import refresh_cache, weibull_cache_enabled
from weibull_evaluation import compare_best_distribution
from reliability_confluence_summary import reliability_summary_table
from weibull import generate_graph, refresh_analysis_cache, refresh_forecast_cache, get_analysis_cache, weibull_fit_best
import atexit
from markupsafe import escape

app = Flask(__name__)
drawio_export_server = os.environ.get('DRAWIO_EXPORT_URL', '')
build_date = os.environ.get('APP_BUILD_DATE', 'unknown')
commit_hash = os.environ.get('APP_GIT_COMMIT', 'unknown')


def refresh_all():
    """Full refresh chain: data → model selection → forecast of expected failures"""
    refresh_cache()             # 1. Pull from DB
    refresh_analysis_cache()    # 2. Model selection with CV (default)
    refresh_forecast_cache()    # 3. Expected failure forecasts
    # try:
    #     reliability_summary_table()
    # except RuntimeError as e:
    #     return 'Reliability table could not be generated in confluence: ' + str(e), 400


if weibull_cache_enabled:
    refresh_all()

    scheduler = BackgroundScheduler()
    scheduler.add_job(refresh_all, 'cron', hour=1, minute=0)
    scheduler.start()

    atexit.register(lambda: scheduler.shutdown())


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
    return render_template('index.html', build_date=build_date, commit_hash=commit_hash)


@app.route('/favicon.ico')
def favicon():
   return app.send_static_file('images/favicon.ico')


@app.route('/weibull', methods=['GET', 'POST'])
def route_weibull():
    part = request.args.get('part')
    plot_type = request.args.get('plot_type')
    edit = request.args.get('edit', '0') == '1'

    if not part:
        return 'Parameter "part" not valid or is missing', 400

    if plot_type not in ('CDF', 'SF'):
        return 'Parameter "plot_type" must be "CDF" or "SF"', 400

    errors = {}
    defaults = {'sort_by': 'CV', 'ci': 0.95}
    return_sf = (plot_type == 'SF')

    if request.method == 'POST':
        sb, err = utils.validate_sort_by(request.form.get('sort_by', ''))
        if err:
            errors['sort_by'] = err

        ci, err = utils.validate_ci(request.form.get('ci', ''))
        if err:
            errors['ci'] = err

        defaults = {'sort_by': request.form.get('sort_by', 'CV'),
                    'ci': request.form.get('ci', '0.95')
        }

        if errors:
            return render_template('weibull_form.html', part=part, plot_type=plot_type, errors=errors, defaults=defaults)

        try:
            graph = generate_graph(part=part, sort_by=sb, ci=ci, return_sf=return_sf)
        except RuntimeError as e:
            return 'Weibull plot cannot be generated: ' + str(e), 400

        image_b64 = base64.b64encode(graph.getvalue()).decode('ascii')

        return render_template('weibull_results.html', part=part, plot_type=plot_type, sort_by=sb, ci=ci, image_b64=image_b64)

    if edit:
        return render_template('weibull_form.html', part=part, plot_type=plot_type, errors=errors, defaults=defaults)

    try:
        graph = generate_graph(part=part, sort_by='CV', ci=0.95, return_sf=return_sf)
    except RuntimeError as e:
        return 'Weibull plot cannot be generated: ' + str(e), 400

    image_b64 = base64.b64encode(graph.getvalue()).decode('ascii')

    return render_template('weibull_results.html', part=part, plot_type=plot_type, sort_by='CV', ci=0.95, image_b64=image_b64)


@app.route('/forecast_fixed', methods=['GET', 'POST'])
def route_forecast_fixed():
    part = request.args.get('part')
    edit = request.args.get('edit', '0') == '1'

    if not part:
        return 'Parameter "part" not valid or is missing', 400

    errors = {}
    defaults = {'fc': '365, 730, 1095, 1825', 'ci': 0.95, 'sort_by': 'CV'}

    def compute_forecast(sb):
        analysis_cache = get_analysis_cache()
        # As long as sort_by=='CV' the cache is valid to use even for the weibull_form
        using_cached_analysis = (sb == 'CV')

        if using_cached_analysis and analysis_cache and part in analysis_cache:
            cached = analysis_cache[part]
            best_model = cached['best_model']
            wb_data_fit_all = cached['fit_table']
            cv_used = cached['cv_used']
        else:
            sort_for_fit = sb if sb != 'CV' else 'BIC'
            wb_data_fit_all, _, data, fit_status = weibull_fit_best(part=part, sort_by=sort_for_fit)
            best_model, cv_used = compare_best_distribution(df=wb_data_fit_all, sort_by=sb, part=part, data=data, ic_fallback='BIC', delta=0.466, fit_status=fit_status)

        return wb_data_fit_all, best_model, cv_used

    if request.method == 'POST':
        sb, err = utils.validate_sort_by(request.form.get('sort_by', ''))
        if err: errors['sort_by'] = err

        fc_values, err = utils.validate_fc(request.form.get('fc', ''))
        if err: errors['fc'] = err

        ci, err = utils.validate_ci(request.form.get('ci', ''))
        if err: errors['ci'] = err

        defaults = {
            'fc': request.form.get('fc', defaults['fc']),
            'ci': request.form.get('ci', defaults['ci']),
            'sort_by': request.form.get('sort_by', defaults['sort_by']),
        }

        if errors:
            return render_template('forecast_form_fixed.html', part=part, errors=errors, defaults=defaults, today_iso=date.today().isoformat())

        try:
            wb_data_fit_all, best_model, cv_used = compute_forecast(sb)
# ToDo: Maybe use here the cached forecast if possible too
            forecast = forecast_part_direct_delta(part=part, deltas=fc_values, fit_table=wb_data_fit_all, best_model=best_model, CI=ci)
            selection_used = 'CV' if cv_used else 'BIC'
        except RuntimeError as e:
            return 'Forecast cannot be calculated: ' + str(e), 400

        return render_template('forecast_results_fixed.html', output=forecast, ci=ci, sort_by=selection_used, today=date.today(), timedelta=timedelta, part=part)

    if edit:
        return render_template('forecast_form_fixed.html', part=part, errors=errors, defaults=defaults, today_iso=date.today().isoformat())

    try:
        fc_default_values, err = utils.validate_fc(defaults['fc'])
        if err:
            return 'Default forecast values invalid: ' + err, 400
        wb_data_fit_all, best_model, cv_used = compute_forecast(defaults['sort_by'])
# ToDo: Maybe use here the cached forecast if possible too
        forecast = forecast_part_direct_delta(part=part, deltas=fc_default_values, fit_table=wb_data_fit_all, best_model=best_model, CI=defaults['ci'])
        selection_used = 'CV' if cv_used else 'BIC'
    except RuntimeError as e:
        return 'Forecast cannot be calculated: ' + str(e), 400

    return render_template('forecast_results_fixed.html', output=forecast, ci=defaults['ci'], sort_by=selection_used, today=date.today(), timedelta=timedelta, part=part)


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
    # If weibull_cache_enabled is used then use_reloader is deactivated otherwise it would load the cache everytime again something is changed
    app.run(debug=True, port=8888, host='0.0.0.0', threaded=True, use_reloader=not weibull_cache_enabled)
