#!/usr/bin/env python
import utils
import base64
import os
from datetime import date, timedelta
from flask import Flask, render_template, request, send_file, url_for, redirect
from apscheduler.schedulers.background import BackgroundScheduler
from weibull_forecast import forecast_part_direct_delta
from data_weibull import refresh_cache, weibull_cache_enabled
from weibull_evaluation import compare_best_distribution
from reliability_confluence_summary import reliability_summary_table
from weibull import generate_graph, refresh_analysis_cache, refresh_forecast_cache, get_analysis_cache, weibull_fit_best
import atexit



app = Flask(__name__)
drawio_export_server = os.environ.get('DRAWIO_EXPORT_URL', '')
build_date = os.environ.get('APP_BUILD_DATE', 'unknown')
commit_hash = os.environ.get('APP_GIT_COMMIT', 'unknown')


def refresh_all():
    """
    Run the full data-to-forecast refresh chain, in dependency order:
    raw data cache → model-selection analysis cache → expected-failures forecast cache.

    Steps
    -----
    1. `refresh_cache()` — pulls the latest Weibull failure/suspension data from the database into the in-memory data cache.
    2. `refresh_analysis_cache()` — re-fits and selects the best Weibull model per part (cross-validation by default),
       using the freshly refreshed data cache.
    3. `refresh_forecast_cache()` — computes expected-failure forecasts per part, using the freshly refreshed analysis cache.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Updates the module-level caches in `data_weibull.py` and `weibull.py` as a side effect.

    Notes
    -----
    Called once synchronously at app startup (when caching is enabled) and then scheduled to run daily
    at 01:00 Europe/Zurich-server time via APScheduler. A commented-out block would additionally publish the
    Confluence reliability summary table after each refresh, but this is currently disabled.
    """
    refresh_cache()             # 1. Pull from DB
    refresh_analysis_cache()    # 2. Model selection with CV (default)
    refresh_forecast_cache()    # 3. Expected failure forecasts
    try:
        reliability_summary_table()
    except RuntimeError as e:
        return 'Reliability table could not be generated in confluence: ' + str(e), 400


if weibull_cache_enabled:
    refresh_all()

    scheduler = BackgroundScheduler()
    scheduler.add_job(refresh_all, 'cron', hour=1, minute=0)
    scheduler.start()

    atexit.register(lambda: scheduler.shutdown())


@app.route('/')
def route_main():
    return render_template('index.html', build_date=build_date, commit_hash=commit_hash)


@app.route('/favicon.ico')
def favicon():
   return app.send_static_file('images/favicon.ico')


@app.route('/weibull', methods=['GET', 'POST'])
def route_weibull():
    """
    Flask view for generating and displaying a Weibull probability (CDF) or reliability (SF) plot for a specific part.

    Query parameters
    ----------------
    part : str, required
        Part identifier to analyze. Returns HTTP 400 if missing.
    plot_type : str, required
        Must be 'CDF' (failure probability plot) or 'SF' (survival/reliability plot). Returns HTTP 400 if neither.
    edit : str, optional
        If '1', renders the editable parameter form (`weibull_form.html`) instead of immediately generating a plot with defaults.

    Form fields (POST only)
    ------------------------
    sort_by : str
        Model-selection method; validated via `utils.validate_sort_by`.
    ci : str
        Confidence level; validated via `utils.validate_ci`.

    Behavior
    --------
    - GET without `edit`: generates and displays the plot immediately using default parameters (sort_by='CV', ci=0.95),
      via `weibull.generate_graph`.
    - GET with `edit=1`: renders the parameter-editing form pre-filled with default values, without generating a plot.
    - POST: validates submitted `sort_by`/`ci` values; on validation failure, re-renders the form with inline error
      messages; on success, generates the plot with the user-chosen parameters and renders the results page.

    Returns
    -------
    flask.Response
        Rendered `weibull_form.html` (on missing/invalid input awaiting correction) or `weibull_results.html`
        (with a base64-encoded PNG embedded as `image_b64`) on success. Returns a plain-text error with HTTP 400
        if `part`/`plot_type` are invalid or if plot generation raises a `RuntimeError`.
    """
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


@app.route('/forecast', methods=['GET', 'POST'])
def route_forecast():
    """
    Flask view for computing and displaying the expected-number-of-failures forecast (with confidence bounds) for a
    specific part, over one or more future time horizons.

    Query parameters
    ----------------
    part : str, required
        Part identifier to forecast for. Returns HTTP 400 if missing.
    edit : str, optional
        If '1', renders the editable parameter form (`forecast_form.html`) instead of immediately computing a forecast
        with defaults.

    Form fields (POST only)
    ------------------------
    sort_by : str
        Model-selection method; validated via `utils.validate_sort_by`.
    fc : str
        Comma-separated forecast horizons in days; validated via `utils.validate_fc`.
    ci : str
        Confidence level; validated via `utils.validate_ci`.

    Behavior
    --------
    - GET without `edit`: computes and displays the forecast immediately using default parameters
      (fc='365, 730, 1095, 1825', ci=0.95, sort_by='CV').
    - GET with `edit=1`: renders the parameter-editing form pre-filled with default values, without computing a forecast.
    - POST: validates submitted `sort_by`/`fc`/`ci` values; on validation failure, re-renders the form with inline
      error messages; on success, computes the forecast with the user-chosen parameters and renders the results page.

    In both the GET-default and POST-success paths, the best-fit model is resolved via the nested `compute_forecast`
    helper (reusing the pre-computed analysis cache when `sort_by == 'CV'` and the part is cached, otherwise re-fitting
    on the fly), and the forecast itself is computed via `weibull_forecast.forecast_part_direct_delta`.

    Returns
    -------
    flask.Response
        Rendered `forecast_form.html` (on missing/invalid input awaiting correction) or `forecast_results.html`
        (with the forecast output, confidence level, selection method used, and current date) on success.
        Returns a plain-text error with HTTP 400 if `part` is missing, if the default forecast horizons fail validation,
        or if forecast computation raises a `RuntimeError`.
    """
    part = request.args.get('part')
    edit = request.args.get('edit', '0') == '1'

    if not part:
        return 'Parameter "part" not valid or is missing', 400

    errors = {}
    defaults = {'fc': '365, 730, 1095, 1825', 'ci': 0.95, 'sort_by': 'CV'}

    def compute_forecast(sb):
        """
        Resolve the best-fit Weibull model and its goodness-of-fit table for the current part, reusing the pre-computed
        analysis cache when possible.

        Parameters
        ----------
        sb : str
            The model-selection method to use ('AICc', 'BIC', or 'CV').

        Returns
        -------
        tuple
            (wb_data_fit_all, best_model, cv_used):
            - wb_data_fit_all : pandas.DataFrame, goodness-of-fit results table for the part (from cache or freshly
                                computed via `weibull.weibull_fit_best`).
            - best_model : str, the selected best-fit distribution name.
            - cv_used : bool, whether the selection was made via cross-validation (True) or an information-criterion
                        fallback (False).

        Notes
        -----
        The pre-computed analysis cache (`weibull.get_analysis_cache()`) is only considered valid/reused
        when `sb == 'CV'` and the part is present in it; otherwise, the model is refit from scratch (with `sb` mapped to
        'BIC' internally if it was 'CV' but no cache was available) and re-evaluated via
        `weibull_evaluation.compare_best_distribution`. This is a closure over `part` from the enclosing
        `route_forecast` view function.
        """
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
            return render_template('forecast_form.html', part=part, errors=errors, defaults=defaults, today_iso=date.today().isoformat())

        try:
            wb_data_fit_all, best_model, cv_used = compute_forecast(sb)
# ToDo: Maybe use here the cached forecast if possible too
            forecast = forecast_part_direct_delta(part=part, deltas=fc_values, fit_table=wb_data_fit_all, best_model=best_model, CI=ci)
            selection_used = 'CV' if cv_used else 'BIC'
        except RuntimeError as e:
            return 'Forecast cannot be calculated: ' + str(e), 400

        return render_template('forecast_results.html', output=forecast, ci=ci, sort_by=selection_used, today=date.today(), timedelta=timedelta, part=part)

    if edit:
        return render_template('forecast_form.html', part=part, errors=errors, defaults=defaults, today_iso=date.today().isoformat())

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

    return render_template('forecast_results.html', output=forecast, ci=defaults['ci'], sort_by=selection_used, today=date.today(), timedelta=timedelta, part=part)



if __name__ == '__main__':
    # If weibull_cache_enabled is used then use_reloader is deactivated otherwise it would load the cache everytime again something is changed
    app.run(debug=True, port=8888, host='0.0.0.0', threaded=True, use_reloader=not weibull_cache_enabled)
