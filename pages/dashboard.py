import os
import dash
import sqlite3
import torch

from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
from dash import html, dcc, callback, Input, Output
from openai import OpenAI

import dash_mantine_components as dmc
import dash_daq as daq

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

load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
                html.Div(
                    id="index-summary",
                    children=[

                        # SENTIMENT SCORE PANEL (index + slider)
                        html.Div(
                            id="sentiment-score-panel",
                            children=[
                                html.Div(
                                    id="sentiment-index-value",
                                    className="sentiment-index-value",
                                ),
                                dmc.Slider(
                                    id="sentiment-index-slider",
                                    min=0,
                                    max=100,
                                    value=50,
                                    disabled=True,
                                    step=1,
                                    size="lg",
                                    marks=[
                                        {"value": 0, "label": "Dovish (-1.0)"},
                                        {"value": 50, "label": "Neutral (0.0)"},
                                        {"value": 100, "label": "Hawkish (1.0)"},
                                    ]
                                )
                            ]
                        ),

                        # SENTIMENT INFO PANEL
                        html.Div(
                            id="sentiment-index-info",
                            style={"width": "100%"},
                            children=[
                                html.Div(
                                    className="sentiment-index-title-with-button",
                                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                                    children=[
                                        html.Div("Sentiment Index & Summary", className="sentiment-index-title"),
                                        dmc.Button(
                                            "Generate Summary",
                                            id="generate-summary-button",
                                            size="xs",
                                            radius="xl",
                                            variant="outline",
                                        )
                                    ],
                                ),
                                html.Div(id="sentiment-index-breakdown", className="sentiment-index-breakdown", style={"marginTop": "0.5rem"}),
                                html.Div(id="sentiment-index-summary", className="sentiment-breakdown-summary", style={"marginTop": "0.5rem"})
                            ]
                        )
                    ],
                )
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


@callback(
    Output("sentiment-index-value", "children"),
    Output("sentiment-index-breakdown", "children"),
    Output("sentiment-index-slider", "value"),
    Output("sentiment-index-slider", "styles"),
    Input("url", "pathname")
)
def update_sentiment_index(pathname):
    if pathname != "/dashboard":
        raise dash.exceptions.PreventUpdate

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT sentiment FROM sentences WHERE sentiment IS NOT NULL")
    sentiments = [row[0] for row in cursor.fetchall()]
    conn.close()

    num_hawkish = sum(1 for s in sentiments if s == 0)
    num_dovish = sum(1 for s in sentiments if s == 1)
    num_neutral = sum(1 for s in sentiments if s == 2)
    total = len(sentiments)

    sentiment_index = (num_hawkish - num_dovish) / total if total > 0 else 0.0
    index_display = f"{sentiment_index:.2f}"
    breakdown = (
        f"Hawkish: {num_hawkish}  |  "
        f"Dovish: {num_dovish}  |  "
        f"Neutral: {num_neutral}  |  "
        f"Total: {total}"
    )
    
    def get_gradient_stop_color(value):
        """Returns the interpolated color at a given slider percentage [0-100]"""
        if value <= 50:
            ratio = value / 50
            r, g, b = 255, int(255 * ratio), 0
        else:
            ratio = (value - 50) / 50
            r = int(255 * (1 - ratio))
            g = int(255 * (1 - 0.5 * ratio))  # yellow to green (255→128)
            b = 0
        return f"rgb({r},{g},{b})"

    slider_val = int((sentiment_index + 1) * 50)  # Keep this for color calculation
    end_color = get_gradient_stop_color(slider_val)
    
    gradient = f"linear-gradient(90deg, red 0%, {end_color} 100%)"

    slider_styles = {
        "bar": {"background": gradient, "height": "8px"},
        "track": {"background": "#e0e0e0", "height": "8px"},
        "thumb": {
            "border": "2px solid white",
            "boxShadow": "0 0 0 1px rgba(0, 0, 0, 0.1)",
            "width": "16px",
            "height": "16px",
            "backgroundColor": "#fff"
        }
    }

    return index_display, breakdown, slider_val, slider_styles


@callback(
    Output("sentiment-index-summary", "children", allow_duplicate=True),
    Input("url", "pathname"),
    prevent_initial_call=True
)
def load_latest_summary(pathname):
    if pathname != "/dashboard":
        raise dash.exceptions.PreventUpdate

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT summary, generated_timestamp
        FROM summary
        ORDER BY generated_timestamp DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()

    if row:
        summary, timestamp = row
        return f"🕒 {timestamp[:19]} UTC\n\n{summary}"
    else:
        return "ℹ️ No summary has been generated yet. Click 'Generate Summary' to create one."
    

@callback(
    Output("sentiment-index-summary", "children", allow_duplicate=True),
    Input("generate-summary-button", "n_clicks"),
    prevent_initial_call=True
)
def generate_summary(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    conn = get_db_connection()
    cursor = conn.cursor()

    # === Fetch all labeled sentences ===
    cursor.execute("SELECT sentence, sentiment FROM sentences WHERE sentiment IS NOT NULL")
    rows = cursor.fetchall()

    if not rows:
        conn.close()
        return "No labeled sentences available for summary."

    # === Build sentence lines ===
    label_map = {0: "Hawkish", 1: "Dovish", 2: "Neutral"}

    num_hawkish = 0
    num_dovish = 0
    sentence_lines = []

    for i, (sentence, label_id) in enumerate(rows):
        label = label_map.get(label_id, "Unknown")
        if label_id == 0:
            num_hawkish += 1
        elif label_id == 1:
            num_dovish += 1
        sentence_lines.append(f"[{i+1}] \"{sentence}\" → Label: {label}")

    total = len(rows)
    index = (num_hawkish - num_dovish) / total if total > 0 else 0
    sentences_text_block = "\n".join(sentence_lines)

    # === Compose GPT prompt ===
    user_message = f"""
Given the following sentences and their sentiment classifications, summarise the overall monetary policy stance of the Fed.

The index is calculated as: (number of hawkish sentences - number of dovish sentences) / (total number of sentences).
A positive index (> 0) indicates an overall hawkish stance.
A negative index (< 0) indicates an overall dovish stance.
An index close to 0 indicates a neutral stance.

Please consider both the index and the provided sentences in your reasoning.

Index value: {index:.2f}

Sentences:
{sentences_text_block}

Summary:
    """

    # === Generate Summary via GPT ===
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a monetary policy expert."},
                {"role": "user", "content": user_message}
            ],
            max_tokens=500,
            temperature=0
        )
        summary = response.choices[0].message.content
    except Exception as e:
        summary = f"❌ Error generating summary: {str(e)}"
        conn.close()
        return summary

    # === Save to DB ===
    timestamp = datetime.utcnow().isoformat()
    cursor.execute("INSERT INTO summary (summary, generated_timestamp) VALUES (?, ?)", (summary, timestamp))
    conn.commit()
    conn.close()

    return f"🆕 Generated: {timestamp[:19]} UTC\n\n{summary}"