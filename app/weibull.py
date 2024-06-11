#!/usr/bin/python3
from predictr import Analysis
import database
import io

def get_data(part):
    failures = []
    suspensions = []

    # Columns available in Weibull_data view: PART,ASSET_ID,RUNNING_TIME,STATUS
    with database.get_cursor() as cursor:
        sql_query = "SELECT RUNNING_TIME, STATUS FROM Weibull_data WHERE PART = :part_id"
        result = cursor.execute(sql_query, (part,))

        for row in result:
            if row[1] == 'S':
                suspensions.append(int(row[0]))
            elif row[1] == 'F':
                failures.append(int(row[0]))
            else:
                raise RuntimeError('Unknown status "{0}"'.format(row[1]))


    if not failures or not suspensions:
        raise RuntimeError('No failure/suspension data for "{0}"'.format(part))

    return {'failures': failures, 'suspensions': suspensions}


def generate_graph(part):
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    data = get_data(part)

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
    return buffer
