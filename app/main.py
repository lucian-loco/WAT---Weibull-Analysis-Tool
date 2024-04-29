#!/usr/bin/python3
from predictr import Analysis
from flask import Flask, send_file, request, g, current_app
import oracledb
import io
import os

app = Flask(__name__)


def get_db_cursor():
    if 'db' not in g:
        conn_params = {
            'user':         os.environ['DB_USER'],
            'password':     os.environ['DB_PASS'],
            'host':         os.environ['DB_HOST'],
            'port':         os.environ['DB_PORT'],
            'service_name': os.environ['DB_SERV'],
        }

        g.db = oracledb.connect(**conn_params)

    return g.db.cursor()


def get_weibull_data(part):
    failures = []
    suspensions = []

    # columns available in Weibull_data view: PART,ASSET_ID,RUNNING_TIME,STATUS
    with get_db_cursor() as cursor:
        sql_query = "SELECT RUNNING_TIME, STATUS FROM Weibull_data WHERE PART = :part_id"
        result = cursor.execute(sql_query, (part,))

        for row in result:
            if row[1] == 'S':
                suspensions.append(int(row[0]))
            elif row[1] == 'F':
                failures.append(int(row[0]))
            else:
                raise RuntimeError('Unknown status "{0}"'.format(row[1]))


    if not failures and not suspensions:
        raise RuntimeError('No data for "{0}"'.format(part))

    return {'failures': failures, 'suspensions': suspensions}



@app.route('/weibull', methods=['GET'])
def graph():
    part = request.args.get('part')

    if not part:
        return 'Invalid request ("part" not specified)', 400


    try:
        data = get_weibull_data(part)
    except RuntimeError as e:
        return str(e), 400


    # Prepare the response
    buffer = io.BytesIO()   # buffer to keep the plot in RAM (instead of a file)

    # Weibull Analysis
    # see https://tvtoglu.github.io/predictr/classes/#default-arguments-and-values for more parameters
    x = Analysis(df=data['failures'], ds=data['suspensions'],
            show=False, save=True,
            fig_size=(9.5, 6),    # (8, 6) -> 800x600
            unit='days',
            plot_title='Weibull Probability Plot for {0}'.format(part),
            path=buffer)
    x.mrr()


    # Send the plot image
    buffer.seek(0)
    return send_file(buffer, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True, port=8888, host='0.0.0.0', threaded=True)
