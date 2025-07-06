import os
import dash
import torch
from dash import html
import dash_mantine_components as dmc
from transformers import AutoTokenizer, RobertaForSequenceClassification

dash.register_page(__name__, path="/dashboard", name="Dashboard")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load RoBERTa model from local path ===
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/finetuned_roberta_model_pre_overfit_epoch_8"))

roberta_tokenizer_pre_overfit = AutoTokenizer.from_pretrained(model_dir)
roberta_model_pre_overfit = RobertaForSequenceClassification.from_pretrained(model_dir, num_labels=3)

roberta_model_pre_overfit = roberta_model_pre_overfit.to(torch.float32).to(device)
roberta_model_pre_overfit.eval()

layout = html.Div(
    id="dashboard-container",
    children=[
        html.Div(
            id="retrieval-statistics-index-summary",
            style={
                "display": "flex",
                "flexDirection": "row",
                "height": "50vh",     # Top half of the screen
                "width": "100vw",     # Full screen width
                "margin": "0",
                "padding": "0",
                "gap": "10px"
            },
            children=[
                # LEFT COLUMN: 3 stacked divs
                html.Div(
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "flex": "1",
                        "gap": "10px"
                    },
                    children=[
                        html.Div("FOMC Docs", id="num-fomc-documents", style={
                            "background": "#E5D8C1",
                            "flex": "1",
                            "padding": "10px"
                        }),
                        html.Div("CNBC Articles", id="num-cnbc-articles", style={
                            "background": "#D1C0AA",
                            "flex": "1",
                            "padding": "10px"
                        }),
                        html.Div("Sentences", id="num-sentences", style={
                            "background": "#BDA893",
                            "flex": "1",
                            "padding": "10px"
                        })
                    ]
                ),
                # RIGHT COLUMN: 1 full-height div
                html.Div("Index Summary", id="index-summary", style={
                    "background": "#A9907D",
                    "flex": "3",
                    "padding": "10px"
                })
            ]
        )
    ]
)