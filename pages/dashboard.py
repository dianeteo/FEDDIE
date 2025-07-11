import os
import dash
import sqlite3
import torch

from tqdm import tqdm
from dash import html, dcc, callback, Input, Output

import dash_mantine_components as dmc

from transformers import AutoTokenizer, RobertaForSequenceClassification

from database.init_db import get_db_connection

dash.register_page(__name__, path="/dashboard", name="Dashboard")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load RoBERTa model from local path ===
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/finetuned_roberta_model_2_pre_overfit_epoch_10_safetensors"))

roberta_tokenizer_pre_overfit = AutoTokenizer.from_pretrained(model_dir)
roberta_model_pre_overfit = RobertaForSequenceClassification.from_pretrained(model_dir, num_labels=3)

roberta_model_pre_overfit = roberta_model_pre_overfit.to(torch.float32).to(device)
roberta_model_pre_overfit.eval()

layout = html.Div(
    id="dashboard-container",
    children=[
        html.Div(
            id="retrieval-statistics-index-summary",
            children=[
                # TOP ROW: 2 side-by-side divs
                html.Div(
                    id="top-horizontal-divs",
                    children=[
                        html.Div(
                            id="num-fomc-docs",
                            children=[
                                html.Div(id="fomc-docs-count", className="fomc-docs-count"),
                                html.Div(
                                    id="fomc-docs-info",
                                    children=[
                                        html.Div("FOMC Documents", className="fomc-docs-title"),
                                        html.Div(id="fomc-docs-links", className="fomc-docs-links")
                                    ]
                                ),
                            ]
                        ),
                        html.Div(
                            id="num-cnbc-articles",
                            children=[
                                html.Div(id="cnbc-articles-count", className="cnbc-articles-count"),
                                html.Div(
                                    id="cnbc-articles-info",
                                    children=[
                                        html.Div("CNBC Articles", className="cnbc-articles-title"),
                                        html.Div(id="cnbc-articles-links", className="cnbc-articles-links")
                                    ]
                                )
                            ]
                        )
                    ]
                ),
                # BOTTOM ROW: Index summary
                html.Div("Index Summary", id="right-index-summary")
            ]
        ),
        dcc.Location(id="url", refresh=False)
    ]
)


@callback(
    Output("fomc-docs-count", "children"),
    Output("fomc-docs-links", "children"),
    Output("cnbc-articles-count", "children"),
    Output("cnbc-articles-links", "children"),
    Input("url", "pathname"),
    prevent_initial_call=True
)
def load_retrieval_stats(pathname):
    if pathname != "/dashboard":
        raise dash.exceptions.PreventUpdate

    conn = get_db_connection()
    cursor = conn.cursor()

    # === FOMC DOCUMENTS ===
    cursor.execute("SELECT date, type, url FROM fomc_documents ORDER BY date DESC")
    fomc_rows = cursor.fetchall()
    fomc_count = len(fomc_rows)

    # Mapping internal types to display labels
    type_labels = {
        "statement": "Statement",
        "minutes": "Meeting Minutes",
        "press_conference": "Press Conference Transcript"
    }

    fomc_links = [
        html.A(
            f"{date} {type_labels.get(doc_type, doc_type.title())}",
            href=url,
            target="_blank"
        )
        for date, doc_type, url in fomc_rows
    ]

    # === CNBC ARTICLES ===
    cursor.execute("SELECT title, url FROM cnbc_articles ORDER BY date DESC")
    cnbc_rows = cursor.fetchall()
    cnbc_count = len(cnbc_rows)
    cnbc_links = [
        html.A(title, href=url, target="_blank", style={"display": "block", "marginBottom": "4px"})
        for title, url in cnbc_rows
    ]
    
    conn.close()

    return fomc_count, fomc_links, cnbc_count, cnbc_links


@callback(
    Input("url", "pathname"),
    prevent_initial_call=True
)
def load_retrieval_stats(pathname):
    if pathname != "/dashboard":
        raise dash.exceptions.PreventUpdate

    conn = get_db_connection()
    cursor = conn.cursor()
    
    # === GENERATING SENTIMENTS FOR EACH SENTENCE ===
    cursor.execute("SELECT id, sentence FROM sentences WHERE sentiment IS NULL")
    sentences_rows = cursor.fetchall()
    
    # === Score each sentence and update DB ===
    for row_id, sentence in tqdm(sentences_rows, desc="Scoring Sentences"):
        print(f"🔍 Scoring Sentence ID {row_id}: {sentence[:80]}...")  # Show first 80 chars

        inputs = roberta_tokenizer_pre_overfit(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = roberta_model_pre_overfit(**inputs)
            logits = outputs.logits
            sentiment_score = logits.argmax(dim=-1).item()

        print(f"✅ Predicted Sentiment Score: {sentiment_score}\nLogits: {logits.cpu().numpy()}")

        cursor.execute("UPDATE sentences SET sentiment = ? WHERE id = ?", (sentiment_score, row_id))

    conn.commit()
    conn.close()
    
    return 