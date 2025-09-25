import os
import dash
import sqlite3
import torch

from datetime import date, datetime, timedelta
from dotenv import load_dotenv
from tqdm import tqdm
from dash import html, dcc, callback, Input, Output, get_asset_url
from openai import OpenAI

import dash_mantine_components as dmc
import dash_daq as daq
from dash_iconify import DashIconify

from transformers import AutoTokenizer, RobertaForSequenceClassification

from database.init_db import get_db_connection

dash.register_page(__name__, path="/dashboard", name="Dashboard", order=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load RoBERTa model from local path ===
model_dir = os.path.abspath(os.path.join(os.path.dirname(
    __file__), "../models/finetuned_roberta_model_macro_f1_minority_recall_4_best_val_loss"))

roberta_tokenizer_pre_overfit = AutoTokenizer.from_pretrained(model_dir)
roberta_model_pre_overfit = RobertaForSequenceClassification.from_pretrained(
    model_dir, num_labels=3, use_safetensors=False, weights_only=False
)

roberta_model_pre_overfit = roberta_model_pre_overfit.to(
    torch.float32).to(device)
roberta_model_pre_overfit.eval()

load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

layout = html.Div(
    id="dashboard-container",
    children=[
        html.Div(
            className="sentiment-hero",
            children=[
                dmc.Title("Fed Sentiment Dashboard", order=2),
                html.Div(
                    className="sentiment-actions",
                    children=[
                        dmc.Button(
                            "Get latest data",
                            id="btn-fetch-sentiment",
                            size="sm",
                            variant="light",
                            color="#062840",
                            leftSection=DashIconify(icon="mdi:download"),
                        ),
                        dmc.Badge("UPDATED", id="sentiment-status-badge", color="GREEN", variant="light"),
                        dmc.LoadingOverlay(
                            id="sentiment-loader"
                        )
                    ],
                ),
            ],
        ),
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
                                html.Div(id="fomc-docs-count",
                                         className="fomc-docs-count"),
                                html.Div(
                                    id="fomc-docs-info",
                                    children=[
                                        html.Div("FOMC Documents",
                                                 className="fomc-docs-title"),
                                        html.Div(id="fomc-docs-links",
                                                 className="fomc-docs-links")
                                    ]
                                ),
                            ]
                        ),
                        html.Div(
                            id="num-cnbc-articles",
                            children=[
                                html.Div(id="cnbc-articles-count",
                                         className="cnbc-articles-count"),
                                html.Div(
                                    id="cnbc-articles-info",
                                    children=[
                                        html.Div(
                                            "CNBC Articles", className="cnbc-articles-title"),
                                        html.Div(id="cnbc-articles-links",
                                                 className="cnbc-articles-links")
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
                                html.Div(id="stance-and-score",
                                         children=[
                                             html.Div(
                                                 id="sentiment-stance-box",
                                                 className="sentiment-stance-box"
                                             ),
                                         ]
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
                                        {"value": 50,
                                            "label": "Neutral (0.0)"},
                                        {"value": 100,
                                            "label": "Hawkish (1.0)"},
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
                                    style={
                                        "display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                                    children=[
                                        html.Div("Sentiment Index & Summary",
                                                 className="sentiment-index-title"),
                                        dmc.Button(
                                            "Generate Summary",
                                            id="generate-summary-button",
                                            size="xs",
                                            radius="xl",
                                            variant="outline",
                                        )
                                    ],
                                ),
                                html.Div(id="sentiment-index-breakdown",
                                         className="sentiment-index-breakdown", style={"marginTop": "0.5rem"}),
                                html.Div(
                                    id="sentiment-index-summary", className="sentiment-breakdown-summary", style={"marginTop": "0.5rem"})
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

    today = date.today()

    # === define month window ===
    month_start = today.replace(day=1)
    if today.month == 12:
        month_end = date(today.year + 1, 1, 1)
    else:
        month_end = date(today.year, today.month + 1, 1)

    # === FOMC DOCUMENTS: only this month ===
    cursor.execute(
        """
        SELECT date, type, url 
        FROM fomc_documents 
        WHERE date >= ? AND date < ?
        ORDER BY date DESC
        """,
        (month_start.isoformat(), month_end.isoformat())
    )
    fomc_rows = cursor.fetchall()
    fomc_count = len(fomc_rows)

    type_labels = {
        "statement": "Statement",
        "minutes": "Meeting Minutes",
        "press_conference": "Press Conference Transcript"
    }

    fomc_links = [
        html.A(
            f"{doc_date} {type_labels.get(doc_type, doc_type.title())}",
            href=url,
            target="_blank"
        )
        for doc_date, doc_type, url in fomc_rows
    ]

    # === CNBC ARTICLES: only past 7 days ===
    one_week_ago = today - timedelta(days=7)
    cursor.execute(
        """
        SELECT title, url, date 
        FROM cnbc_articles 
        WHERE date >= ?
        ORDER BY date DESC
        """,
        (one_week_ago.isoformat(),)
    )
    cnbc_rows = cursor.fetchall()
    cnbc_count = len(cnbc_rows)
    cnbc_links = [
        html.A(
            title, href=url, target="_blank",
            style={"display": "block", "marginBottom": "4px"}
        )
        for title, url, _ in cnbc_rows
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
    cursor.execute(
        "SELECT id, sentence FROM sentences WHERE sentiment IS NULL")
    sentences_rows = cursor.fetchall()

    # === Score each sentence and update DB ===
    for row_id, sentence in tqdm(sentences_rows, desc="Scoring Sentences"):
        # Show first 80 chars
        print(f"🔍 Scoring Sentence ID {row_id}: {sentence[:80]}...")

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

        print(
            f"✅ Predicted Sentiment Score: {sentiment_score}\nLogits: {logits.cpu().numpy()}")

        cursor.execute(
            "UPDATE sentences SET sentiment = ? WHERE id = ?", (sentiment_score, row_id))

    conn.commit()
    conn.close()

    return


@callback(
    Output("sentiment-stance-box", "children"),
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
    cursor.execute(
        "SELECT sentiment FROM sentences WHERE sentiment IS NOT NULL")
    sentiments = [row[0] for row in cursor.fetchall()]
    conn.close()

    num_hawkish = sum(1 for s in sentiments if s == 0)
    num_dovish = sum(1 for s in sentiments if s == 1)
    num_neutral = sum(1 for s in sentiments if s == 2)
    total = len(sentiments)

    sentiment_index = (num_hawkish - num_dovish) / total if total > 0 else 0.0
    
    # Stance and logo selection
    if sentiment_index > 0:
        stance_text = f"Hawkish ({sentiment_index:.2f})"
        logo_file = "HAWKISH_LOGO.png"
    elif sentiment_index < 0:
        stance_text = f"Dovish ({sentiment_index:.2f})"
        logo_file = "DOVISH_LOGO.png"
    else:
        stance_text = f"Neutral ({sentiment_index:.2f})"
        logo_file = "FEDDIE_LOGO.png"
        
    stance_children = html.Div(
        className="stance-row",
        children=[
            html.Span(stance_text),
            html.Img(
                src=get_asset_url(logo_file),
                alt=f"{stance_text} logo",
                className="stance-logo",
                draggable="false"
            ),
        ],
    )
    
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

    # Keep this for color calculation
    slider_val = int((sentiment_index + 1) * 50)
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

    return stance_children, breakdown, slider_val, slider_styles


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

    cursor.execute(
        "SELECT sentence, sentiment FROM sentences WHERE sentiment IS NOT NULL")
    rows = cursor.fetchall()
    if not rows:
        conn.close()
        return "No labeled sentences available for summary."

    label_map = {0: "Hawkish", 1: "Dovish", 2: "Neutral"}
    num_hawkish = sum(1 for _, l in rows if l == 0)
    num_dovish = sum(1 for _, l in rows if l == 1)
    total = len(rows)
    index = (num_hawkish - num_dovish) / total if total else 0.0

    # === Map–Reduce ===
    BATCH_SIZE = 300
    batch_summaries = []
    for i in range(0, total, BATCH_SIZE):
        subset = rows[i:i+BATCH_SIZE]
        lines = [f'"{s}" → {label_map.get(l,"Unknown")}' for s, l in subset]
        batch_prompt = (
            "You are a monetary policy expert.\n"
            f"Global Index: {index:.2f}\n\n"
            "Summarise stance (≤120 words) and give counts hawkish/dovish/neutral.\n"
            "Sentences:\n" + "\n".join(lines)
        )
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": batch_prompt}],
            max_tokens=250, temperature=0, timeout=30,
        )
        batch_summaries.append(r.choices[0].message.content)

    final_prompt = (
        "You are a monetary policy expert.\n"
        f"Global Index: {index:.2f}\n\n"
        "You are given batch summaries (each includes counts). "
        "Produce ONE final coherent summary (≤200 words).\n\n"
        + "\n".join(f"[Batch {i+1}] {s}" for i,
                    s in enumerate(batch_summaries))
    )

    try:
        final_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": final_prompt}],
            max_tokens=400, temperature=0, timeout=45,
        )
        summary = final_resp.choices[0].message.content
    except Exception as e:
        conn.close()
        return f"❌ Error generating summary: {e}"

    timestamp = datetime.utcnow().isoformat()
    cursor.execute(
        "INSERT INTO summary (summary, generated_timestamp) VALUES (?, ?)",
        (summary, timestamp)
    )
    conn.commit()
    conn.close()

    return f"🆕 Generated: {timestamp[:19]} UTC\n\n{summary}"