import dash
from dash import html
import dash_mantine_components as dmc

dash.register_page(__name__, path="/macroeconomic", name="Macroeconomic", order=2)

layout = html.Div(
    id="macro-container",
    children=[
        dmc.Title("Macroeconomic Dashboard", order=3),
        html.Div("Coming soon…")
    ],
)
