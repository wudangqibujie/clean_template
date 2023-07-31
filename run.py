# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument("--model", "-m", type=str, default="BPR", help="name of models")
# parser.add_argument(
#     "--dataset", "-d", type=str, default="ml-100k", help="name of datasets"
# )
# parser.add_argument("--config_files", type=str, default=None, help="config files")
# parser.add_argument(
#     "--nproc", type=int, default=1, help="the number of process in this group"
# )
#
# args, _ = parser.parse_known_args()
# print(args)
#


# See official docs at https://dash.plotly.com
# pip install dash pandas
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')
app = Dash(__name__, external_stylesheets=[r"./assets/style.css"])
app.layout = html.Div([
    html.H1(children='Title of Dash App', style={'textAlign': 'center'}),
    dcc.Dropdown(df.country.unique(), 'Canada', id='dropdown-selection'),
    dcc.Graph(id='graph-content')
])
@callback(
    Output('graph-content', 'figure'),
    Input('dropdown-selection', 'value')
)
def update_graph(value):
    dff = df[df.country == value]
    return px.line(dff, x='year', y='pop')


if __name__ == '__main__':
    app.run(debug=True)
