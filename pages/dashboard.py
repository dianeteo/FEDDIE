import dash
from dash import html
import dash_mantine_components as dmc

dash.register_page(__name__, path="/dashboard", name="Dashboard")

layout = html.Div(
    id="dashboard-page",
    children=[
        dmc.Title("Dashboard", order=2),
        html.Div(id="dashboard-content")
    ]
)
