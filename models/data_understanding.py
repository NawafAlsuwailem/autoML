import seaborn as sns
from io import StringIO
import numpy as np
import plotly
import json
import plotly.express as px
import pandas as pd


def get_data_shape(data):
    return data.shape


def get_data_describe(data):
    return data.describe()


def get_data_info(data):
    return data.info()


def get_null_sum(data):
    return data.isnull().sum()


def get_heatmap(data):
    return sns.heatmap(data)


def process_content_info(content: pd.DataFrame):
    content_info = StringIO()
    content.info(buf=content_info, null_counts=False)
    str_ = content_info.getvalue()

    lines = str_.split("\n")
    table = StringIO("\n".join(lines[3:-3]))
    datatypes = pd.read_table(table, delim_whitespace=True,
                   names=["Feature", "dtype"])
    datatypes = pd.DataFrame(datatypes)
    null = np.array(content.isnull().sum())
    datatypes['null'] = null

    return datatypes


def get_feature_stats(dataframe):
    list_of_charts = []
    columns = dataframe.keys()
    try:
        for col in columns:
            fig = None
            column = dataframe[col]
            if np.issubdtype(column.dtype, np.object) and dataframe.groupby(column).count().shape[0] <= 5:
                fig = px.histogram(dataframe, x=column, color=column)
            elif np.issubdtype(column.dtype, np.number):
                if dataframe.groupby(column).count().shape[0] <= 2:
                    fig = px.histogram(dataframe, x=column, color=column)
                elif column.min() == 0 and\
                        column.quantile(0) == 0 and\
                        column.quantile(0.5) == 0 and dataframe.groupby(column).count().shape[0] <= 5:
                    fig = px.histogram(dataframe, x=column, color=column)
                else:
                    fig = px.box(dataframe, y=column)
            else:
                pass

            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            list_of_charts.append([col, graphJSON])
    except:
        pass
    list_of_charts = pd.DataFrame(list_of_charts, columns=["feature", "chart"])
    # print(list_of_charts)
    return list_of_charts

