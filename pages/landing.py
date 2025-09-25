import dash
import dash_mantine_components as dmc
import re
import requests
import sqlite3
import time
import re

from datetime import date, timedelta
from dash import html, dcc, callback, Input, Output, State
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PyPDF2 import PdfReader
from io import BytesIO
from datetime import datetime, timedelta

from database.init_db import get_db_connection, insert_fomc_document, insert_cnbc_article

dash.register_page(__name__, path="/", name="Home", order=0)

layout = html.Div(
    id="landing-container",
    children=[
        dcc.Location(id="url", refresh=False),
        html.Div(
            id="logo-description-button-container",
            children=[
                html.Div(
                    id="logo-div",
                    children=html.Img(
                        id="logo-img",
                        src="assets/FEDDIE_LOGO.png",
                    ),
                ),
                html.Div(
                    id="description-button-div",
                    children=[
                        dmc.Title("Welcome to FEDDIE",
                                  id="welcome-title", order=3),
                        dmc.Text(
                            "FEDDIE analyses the Federal Market Open Committee's documents and news articles and generates a Hawkish/Dovish sentiment score.",
                            id="description-text",
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
                        dcc.Store(id='sentences-store'),
                        dcc.Store(id='bool-trigger-scraping'), # STORES BOOLEAN
                        dcc.Store(id='fomc-documents-retrieved'), # STORES BOOLEAN 
                        dcc.Store(id='cnbc-articles-retrieved'), # STORES BOOLEAN 
                        dcc.Store(id="loading-progress-bar-status", data=0),
                        dcc.Location(id="redirect", refresh=True),
                        html.Div(
                            id='loading-div',
                            children=[
                                dmc.Progress(id="loading-progress-bar", value=0, color="#062840", size="sm"),
                                html.Div(id="loading-progress-text"),
                            ]
                        )
                    ],
                ),
            ],
        )
    ]
)


# === Regex Setup ===
sentence_pattern = re.compile(r'(?<=[.!?]) +')
split_tokens = ["but", "however", "even though", "although", "while", ";"]
split_pattern = re.compile(
    r"\b(" + "|".join(map(re.escape, split_tokens)) + r")\b|;")

keywords = set(map(str.lower, [
    "inflation expectation", "interest rate", "bank rate", "fund rate", "price",
    "economic activity", "inflation", "employment",
    "anchor", "cut", "subdue", "decline", "decrease", "reduce", "low", "drop", "fall",
    "fell", "decelerate", "slow", "pause", "pausing", "stable", "non-accelerating",
    "downward", "tighten",
    "unemployment", "growth", "exchange rate", "productivity", "deficit", "demand",
    "job market", "monetary policy",
    "ease", "easing", "rise", "rising", "increase", "expand", "improve", "strong",
    "upward", "raise", "high", "rapid"
]))

junk_phrases = [
    "cookie", "cookies", "terms of use", "privacy policy", "ads and content",
    "by using this site", "subscribe", "sign up", "CNBC", "NBCUniversal", "copyright",
    "click", "browser", "advertise with us"
]

def start_chrome_driver():
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--log-level-3')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    return webdriver.Chrome(options=options)


@callback(
    Output("bool-trigger-scraping", "data", allow_duplicate=True),
    Output("fomc-documents-retrieved", "data", allow_duplicate=True),
    Output("cnbc-articles-retrieved", "data", allow_duplicate=True),
    Output("sentences-store", "data", allow_duplicate=True),
    Output("loading-progress-bar-status", "data", allow_duplicate=True),
    Output("loading-progress-text", "children", allow_duplicate=True),
    Input("url", "pathname"),
    prevent_initial_call=True
)
def reset_stores_on_landing(pathname):
    if pathname == "/":
        return False, False, False, {}, 0, None
    raise dash.exceptions.PreventUpdate


@callback(
    Output("loading-progress-text", "children", allow_duplicate=True),
    Input("get-started-button", "n_clicks"),
    prevent_initial_call=True
)
def get_started(n_clicks):
    return "Loading FOMC documents..."

@callback(
    Output("bool-trigger-scraping", "data"),
    Output("fomc-documents-retrieved", "data"),
    Output("loading-progress-text", "children"),
    Input("get-started-button", "n_clicks"),
    prevent_initial_call=True
)
def scrape_fomc(n_clicks):
    print("🚀 Starting FOMC + CNBC scrape and sentence processing...")

    driver = None
    try:
        driver = start_chrome_driver()

        today = datetime.today()
        driver.get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
        time.sleep(2)

        meeting_blocks = driver.find_elements(By.CSS_SELECTOR, ".fomc-meeting")
        latest_meeting = None
        latest_date = None

        for block in meeting_blocks:
            try:
                month_text = block.find_element(By.CLASS_NAME, "fomc-meeting__month").text.strip()
                day_text = block.find_element(By.CLASS_NAME, "fomc-meeting__date").text.strip()
                first_day = int(re.findall(r"\d+", day_text)[0])
                year_match = re.search(r"20\d{2}", block.get_attribute("innerHTML"))
                year = int(year_match.group()) if year_match else today.year
                month_num = time.strptime(month_text, '%B').tm_mon
                date_obj = datetime(year, month_num, first_day)
                if date_obj <= today and (latest_date is None or date_obj > latest_date):
                    latest_meeting = block
                    latest_date = date_obj
            except:
                continue

        if latest_meeting and latest_date:
            date_str = latest_date.strftime("%Y-%m-%d")
            full_page_soup = BeautifulSoup(driver.page_source, "html.parser")

            # scrape ONLY this month's FOMC *minutes* 
            from urllib.parse import urljoin

            # define current month window [start, end)
            month_start = today.replace(day=1).date()
            month_end = (month_start.replace(year=month_start.year + 1, month=1, day=1)
                        if month_start.month == 12
                        else month_start.replace(month=month_start.month + 1, day=1))

            # find all minutes links on the calendar page
            minutes_links = []
            for a in full_page_soup.find_all("a", href=True):
                href = a["href"]
                m = re.search(r"fomcminutes(20\d{6})\.htm", href)  # captures YYYYMMDD
                if not m:
                    continue
                ymd = m.group(1)
                try:
                    dt = datetime.strptime(ymd, "%Y%m%d").date()
                except ValueError:
                    continue

                # keep only links whose date is within this month
                if not (month_start <= dt < month_end):
                    continue

                abs_url = href if href.startswith("http") else urljoin("https://www.federalreserve.gov", href)
                minutes_links.append((abs_url, dt))

            # de-dupe by date (multiple anchors can point to same doc)
            seen = set()
            filtered = []
            for url, dt in minutes_links:
                if dt in seen:
                    continue
                seen.add(dt)
                filtered.append((url, dt))

            # fetch each minutes page and insert
            for url, dt in filtered:
                try:
                    driver.get(url)
                    time.sleep(2)
                    doc_soup = BeautifulSoup(driver.page_source, "html.parser")
                    content = doc_soup.get_text(separator="\n", strip=True)

                    insert_fomc_document(
                        url=url,
                        date=dt.strftime("%Y-%m-%d"),
                        type="minutes",
                        content=content
                    )
                    print(f"✅ Inserted minutes {dt} -> {url}")
                except Exception as e:
                    print(f"⚠️ Failed to fetch/insert {url}: {e}")
        else:
            print("⚠️ No latest meeting block found; proceeding anyway to signal next stage.")

        fomc_token = {"done": True, "t": time.time()}
        return True, fomc_token, "Loading CNBC articles..."

    except Exception as e:
        print(f"❌ FOMC scrape error: {e}")
        return False, False, f"FOMC scrape failed: {e}"
    finally:
        try:
            if driver:
                driver.quit()
        except Exception:
            pass


@callback(
    Output("cnbc-articles-retrieved", "data", allow_duplicate=True),
    Output("loading-progress-bar-status", "data", allow_duplicate=True),
    Output("loading-progress-text", "children", allow_duplicate=True),
    Input("fomc-documents-retrieved", "data"),
    State("bool-trigger-scraping", "data"),
    prevent_initial_call=True
)
def scrape_cnbc(trigger, scrape_bool):
    if not trigger or not scrape_bool:
        raise dash.exceptions.PreventUpdate
    
    driver = start_chrome_driver()
    
    today = datetime.today()
    one_week_ago = today - timedelta(days=7)
    
    # === CNBC SCRAPER ===
    driver.get("https://www.cnbc.com/federal-reserve/")
    time.sleep(5)
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    for card in soup.find_all("div", class_="Card-card"):
        title_tag = card.find("a", class_="Card-title")
        date_tag = card.find("span", class_="Card-time")
        if not title_tag or not date_tag:
            continue

        try:
            clean_date = date_tag.text.strip().replace('st', '').replace(
                'nd', '').replace('rd', '').replace('th', '')
            article_date = datetime.strptime(clean_date, "%a, %b %d %Y")
            if article_date < one_week_ago:
                continue

            article_url = title_tag["href"]
            driver.get(article_url)
            time.sleep(2)

            article_soup = BeautifulSoup(driver.page_source, 'html.parser')
            summary = article_soup.find_all('li')
            paragraphs = article_soup.find_all('p')

            content_parts = [title_tag.text.strip()]
            if summary:
                content_parts.append("Summary:")
                content_parts.extend(line.get_text(strip=True)
                                     for line in summary if line.get_text(strip=True))
            if paragraphs:
                content_parts.append("Body:")
                content_parts.extend(p.get_text(strip=True)
                                     for p in paragraphs if p.get_text(strip=True))

            content = '\n'.join(content_parts)
            insert_cnbc_article(title=title_tag.text.strip(
            ), url=article_url, date=article_date.strftime("%Y-%m-%d"), content=content)

        except Exception as e:
            print(f"⚠️ CNBC article scrape error: {e}")

    driver.quit()
    
    return True, 66, "Processing sentences..."


def _month_bounds(d: date):
    month_start = d.replace(day=1)
    # first day of next month
    if month_start.month == 12:
        month_end = month_start.replace(year=month_start.year + 1, month=1, day=1)
    else:
        month_end = month_start.replace(month=month_start.month + 1, day=1)
    return month_start, month_end


@callback(
    Output("sentences-store", "data", allow_duplicate=True),
    Output("loading-progress-bar-status", "data", allow_duplicate=True),
    Output("loading-progress-text", "children", allow_duplicate=True),
    Output("redirect", "href"),
    Input("cnbc-articles-retrieved", "data"),
    State("bool-trigger-scraping", "data"),
    prevent_initial_call=True
)
def process_sentences(trigger, scrape_bool):
    if not trigger or not scrape_bool:
        raise dash.exceptions.PreventUpdate

    print("⌛ Processing sentences...")

    # === Step 1: Fetch raw content (filter to THIS MONTH'S FOMC MINUTES) ===
    conn = get_db_connection()
    cursor = conn.cursor()

    # Compute month window in ISO strings (assumes your 'date' column is ISO 'YYYY-MM-DD')
    today = date.today()
    month_start, month_end = _month_bounds(today)
    ms_str = month_start.isoformat()      # e.g. '2025-09-01'
    me_str = month_end.isoformat()        # e.g. '2025-10-01'

    # Only minutes, only this month
    cursor.execute("""
        SELECT url, content, date
        FROM fomc_documents
        WHERE type = 'minutes' AND date >= ? AND date < ?
        ORDER BY date DESC
    """, (ms_str, me_str))
    fomc_data = [{"url": r[0], "content": r[1], "type": "fomc", "date": r[2]} for r in cursor.fetchall()]

    # If you STILL want CNBC context (optional), you can either keep it unrestricted,
    # or also filter CNBC to this month. To keep it unrestricted, leave as-is:
    cursor.execute("SELECT url, content, date FROM cnbc_articles")
    # If you prefer CNBC only this month, use the filtered version instead:
    # cursor.execute("SELECT url, content, date FROM cnbc_articles WHERE date >= ? AND date < ?", (ms_str, me_str))

    cnbc_data = [{"url": r[0], "content": r[1], "type": "cnbc", "date": r[2]} for r in cursor.fetchall()]

    conn.close()

    all_data = fomc_data + cnbc_data
    sentences_to_insert = []

    for item in all_data:
        content = item.get("content", "")
        url = item.get("url", "unknown_source")
        source_type = item.get("type", "unknown_type")
        dt = item.get("date", None)

        if not content.strip():
            continue

        sentences = sentence_pattern.split(content)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            parts = split_pattern.split(sentence)
            parts = [p.strip() for p in parts if p and not re.match(split_pattern, p)]

            for part in parts:
                if len(part.split()) < 3 or part.count('\n') > 3 or len(re.findall(r'[.!?]', part)) < 1:
                    continue

                part_lower = part.lower()
                if any(jp in part_lower for jp in junk_phrases):
                    continue
                if any(re.search(rf"\b{re.escape(k)}\b", part_lower) for k in keywords):
                    sentences_to_insert.append((part, None, source_type, url, dt))

    print(f"✅ Prepared {len(sentences_to_insert)} sentences for insertion.")

    # === Step 2: Insert ===
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT OR IGNORE INTO sentences (sentence, sentiment, source_type, url, date)
        VALUES (?, ?, ?, ?, ?)
    """, sentences_to_insert)
    conn.commit()
    conn.close()

    print("✅ Sentences inserted into the database.")
    return {}, 100, "Processing sentences...", "/dashboard"


@callback(
    Output("loading-progress-bar", "value"),
    Input("loading-progress-bar-status", "data"),
    prevent_initial_call=True
)
def update_progress_bar(progress):
    return progress