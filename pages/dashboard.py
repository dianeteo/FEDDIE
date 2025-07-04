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
    id="dashboard-page",
    children=[
        dmc.Title("Dashboard", order=2),
        html.Div(id="dashboard-content")
    ]
)