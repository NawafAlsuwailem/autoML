import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from generic.service import *
from preprocessing import *
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly
import json
import plotly.express as px


def get_heatmap(dataframe):
    # print(data frame['gender'].values < 0)
    corr = dataframe.corr()
    value_list = corr.values.tolist()
    x = y = corr.columns.tolist()
    fig = ff.create_annotated_heatmap(value_list, x=x, y=y, colorscale='Viridis')
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 8
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


def get_scatter_matrix(dataframe, target_feature):
    fig = px.scatter_matrix(dataframe, color=target_feature)
    fig.update_layout(
        font=dict(
            family="Courier New, monospace",
            size=2,
            color="#7f7f7f"
        )
    )
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


def get_parallel_categories(dataframe, target_feature):
    try:
        fig = px.parallel_categories(dataframe, color=target_feature,
                                      color_continuous_scale=px.colors.sequential.Inferno)
    except:
        fig = px.parallel_categories(dataframe)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


def get_data_dist(dataframe, target_feature):
    list_of_charts = []
    columns = dataframe.keys()
    try:
        for col in columns:
            fig = None
            column = dataframe[col]
            fig = px.histogram(dataframe, x=column, y=target_feature, color=target_feature,
                               marginal="box",
                               hover_data=dataframe.columns)

            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            list_of_charts.append([col, graphJSON])
    except:
        pass
    list_of_charts = pd.DataFrame(list_of_charts, columns=["feature", "chart"])
    return list_of_charts


def get_feature_importance(dataframe, target_feature):
    X = define_independent_variables(dataframe, target_feature)
    y = define_dependant_variable(dataframe, target_feature)

    forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    feature_by_importance = []

    for i in indices:
        feature_by_importance.append(X.iloc[:, [i]].keys()[0])


    feature_impor_details = pd.DataFrame(data=[importances, std], index=["importance", "std"],
                                         columns=X.keys())

    fig = px.bar(feature_impor_details, y=importances, x=X.keys(), color=feature_by_importance)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    feature_by_importance = feature_by_importance[:3]

    return feature_impor_details, feature_by_importance, graphJSON
