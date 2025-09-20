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
    return dmc.Paper(
        withBorder=True,
        shadow="xs",
        p="sm",
        style={"height": "60px", "display": "flex", "alignItems": "center"},
        children=dmc.Group(
            gap="lg",
            children=[
                dcc.Link("Home", href="/", id="nav-home", style=BASE_LINK_STYLE),
                dcc.Link("Fed Sentiment Dashboard", href="/dashboard", id="nav-dashboard", style=BASE_LINK_STYLE),
                dcc.Link("Macroeconomic Dashboard", href="/macroeconomic", id="nav-macro", style=BASE_LINK_STYLE),
            ],
        ),
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
                        "justifyContent": "center",  # optional, to vertically center
                        "alignItems": "center",      # optional, to horizontally center
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
