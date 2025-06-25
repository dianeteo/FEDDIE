
import torch
import dash_mantine_components as dmc

from dash import Dash, html, callback, Input, State, Output
from transformers import AutoTokenizer, RobertaForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("./models/finetuned_roberta_model_pre_overfit_epoch_8")
model = RobertaForSequenceClassification.from_pretrained(
    "./models/finetuned_roberta_model_pre_overfit_epoch_8", torch_dtype=torch.float16, device_map="cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

id2label = {0: "HAWKISH", 1: "DOVISH", 2: "NEUTRAL"}

app = Dash(
    __name__,
    title="FEDDIE"
)

# Requires Dash 2.17.0 or later
app.layout = dmc.MantineProvider(
    children=[
        html.Div(
            id = "landing-container",
            children=[
                html.Div(
                    id = "logo-description-button-container",
                    children=[
                        # Left column — Logo
                        html.Div(
                            id = "logo-div",
                            children=html.Img(
                                id = "logo-img",
                                src="assets/FEDDIE_LOGO.png",  # replace with your actual logo path
                            ),
                        ),
                        # Right column — Description + Button
                        html.Div(
                            id = "description-button-div",
                            children=[
                                dmc.Title(
                                    "Welcome to FEDDIE",
                                    id = "welcome-title",
                                    order=3,
                                ),
                                dmc.Text(
                                    "FEDDIE analyses the Federal Market Open Committee's documents and news articles and generates a sentiment score denoting whether the Fed's stance is Hawkish/Dovish.",
                                    id = "description-text",
                                    size="lg",
                                ),
                                dmc.Button(
                                    "Get Started",
                                    id="get-started-button",
                                    color="#062840",
                                    variant="filled",
                                    size="lg",
                                    radius="xl",
                                ),
                            ],
                        ),
                    ],
                )
            ],
        )
    ]
)



if __name__ == '__main__':
    app.run(debug=True)


