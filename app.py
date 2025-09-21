import dash
from dash import Dash, html, dcc, Input, Output
import dash_mantine_components as dmc
from database.init_db import init_db

init_db()

app = Dash(__name__, use_pages=True)
server = app.server  # (optional) if you deploy later

BASE_LINK_STYLE = {
    "textDecoration": "none",
    "fontWeight": 500,
    "color": "#062840",
    "padding": "6px 10px",
}

def build_navbar():
    return html.Div(
        id="navbar",
        children=[
            # Left side: logo + title
            html.Div(
                id="navbar-left",
                children=[
                    html.Img(src="/assets/FEDDIE_LOGO.png", id="navbar-logo"),
                    html.Span("FEDDIE", id="navbar-title"),
                ],
            ),
            # Right side: nav links
            html.Div(
                id="navbar-links",
                children=[
                    dcc.Link("Home", href="/", id="nav-home", className="nav-link"),
                    dcc.Link("Fed Sentiment Dashboard", href="/dashboard", id="nav-dashboard", className="nav-link"),
                    dcc.Link("Macroeconomic Dashboard", href="/macroeconomic", id="nav-macro", className="nav-link"),
                ],
            ),
        ],
    )

app.layout = dmc.MantineProvider(
    children=[
        dcc.Location(id="url", refresh=False),

        # Outer flex column container that fills the screen
        html.Div(
            style={
                "display": "flex",
                "flexDirection": "column",
                "height": "100vh",   # full viewport
            },
            children=[
                build_navbar(),
                html.Div(
                    id="page-container",
                    children=dash.page_container,
                    style={
                        "flex": 1,              # take remaining space
                        "overflow": "auto",     # scroll inside if content is taller
                        "display": "flex",
                        "flexDirection": "column",
                    },
                ),
            ],
        ),
    ]
)

# --- Active link highlighting (underline current page) ---
@app.callback(
    Output("nav-home", "style"),
    Output("nav-dashboard", "style"),
    Output("nav-macro", "style"),
    Input("url", "pathname"),
)
def highlight_active(pathname: str):
    def style(is_active: bool):
        s = dict(BASE_LINK_STYLE)
        if is_active:
            s.update({"borderBottom": "2px solid #062840"})
        return s

    return (
        style(pathname == "/"),
        style(pathname == "/dashboard"),
        style(pathname == "/macroeconomic"),
    )

if __name__ == "__main__":
    app.run(debug=True)
