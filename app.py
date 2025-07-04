import dash
import dash_mantine_components as dmc

from dash import Dash, html, dcc
from database.init_db import init_db

init_db()

app = Dash(__name__, use_pages=True)

app.layout = dmc.MantineProvider(
    children=[
        html.Div([
            dash.page_container
        ])
    ]
)

if __name__ == '__main__':
    app.run(debug=True)
