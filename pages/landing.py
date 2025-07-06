import dash
import dash_mantine_components as dmc
import re
import requests
import sqlite3
import time
import re

from dash import html, dcc, ctx, callback, Input, Output, State
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PyPDF2 import PdfReader
from io import BytesIO
from datetime import datetime, timedelta

from database.init_db import get_db_connection, insert_fomc_document, insert_cnbc_article

dash.register_page(__name__, path="/", name="Home")

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
                        dcc.Store('bool-trigger-scraping'), # STORES BOOLEAN
                        dcc.Store(id='fomc-documents-retrieved'), # STORES BOOLEAN 
                        dcc.Store(id='cnbc-articles-retrieved'), # STORES BOOLEAN 
                        dcc.Store(id='sentences-store'),
                        dcc.Location(id="redirect", refresh=True),
                        dcc.Store(id="loading-progress-bar-status", data=0),
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
    Output("bool-trigger-scraping", "data", allow_duplicate=True),
    Output("fomc-documents-retrieved", "data"),
    Output("loading-progress-bar-status", "data", allow_duplicate=True),
    Output("loading-progress-text", "children", allow_duplicate=True),
    Input("get-started-button", "n_clicks"),
    prevent_initial_call=True
)
def scrape_fomc(n_clicks):
    print("🚀 Starting FOMC + CNBC scrape and sentence processing...")

    # === Setup Chrome headless ===
    driver = start_chrome_driver()

    today = datetime.today()

    # === FOMC DOCUMENT SCRAPER ===
    driver.get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
    time.sleep(2)

    meeting_blocks = driver.find_elements(By.CSS_SELECTOR, ".fomc-meeting")
    latest_meeting = None
    latest_date = None

    for block in meeting_blocks:
        try:
            month_text = block.find_element(
                By.CLASS_NAME, "fomc-meeting__month").text.strip()
            day_text = block.find_element(
                By.CLASS_NAME, "fomc-meeting__date").text.strip()
            first_day = int(re.findall(r"\d+", day_text)[0])
            year_match = re.search(
                r"20\d{2}", block.get_attribute("innerHTML"))
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

        def scrape_and_insert(doc_type, pattern, is_pdf=False, nested_pdf=False):
            if nested_pdf:
                pressconf_link = full_page_soup.find(
                    "a", href=re.compile(pattern))
                if not pressconf_link:
                    print(f"❌ Could not find link to {doc_type} HTML page")
                    return

                pressconf_url = urljoin(
                    "https://www.federalreserve.gov", pressconf_link["href"])
                try:
                    response = requests.get(pressconf_url)
                    inner_soup = BeautifulSoup(response.text, "html.parser")
                    pdf_link = inner_soup.find(
                        "a", href=re.compile(r"FOMCpresconf20\d{6}\.pdf"))
                    if not pdf_link:
                        print(f"❌ No {doc_type} PDF found in linked page")
                        return

                    pdf_url = urljoin(
                        "https://www.federalreserve.gov", pdf_link["href"])
                    response = requests.get(pdf_url)
                    reader = PdfReader(BytesIO(response.content))
                    content = "\n".join(page.extract_text()
                                        or "" for page in reader.pages)
                    insert_fomc_document(
                        date_str, doc_type, "PDF", pdf_url, content)
                except Exception as e:
                    print(f"⚠️ Error while processing nested {doc_type}: {e}")
                return

            link = full_page_soup.find("a", href=re.compile(pattern))
            if not link:
                print(f"❌ Could not find direct link for {doc_type}")
                return

            url = urljoin("https://www.federalreserve.gov", link["href"])
            try:
                response = requests.get(url)
                if is_pdf:
                    reader = PdfReader(BytesIO(response.content))
                    content = "\n".join(page.extract_text()
                                        or "" for page in reader.pages)
                else:
                    content = BeautifulSoup(response.text, "html.parser").get_text(
                        separator="\n", strip=True)
                insert_fomc_document(date_str, doc_type,
                                     "PDF" if is_pdf else "HTML", url, content)
            except Exception as e:
                print(f"⚠️ Error while processing {doc_type}: {e}")

        scrape_and_insert("statement", r"monetary20\d{6}a\.htm")
        scrape_and_insert("minutes", r"fomcminutes20\d{6}\.htm")
        scrape_and_insert(
            "press_conference", r"fomcpresconf20\d{6}\.htm", is_pdf=True, nested_pdf=True)
        
        driver.quit()
        
        return True, True, 33, "Loading FOMC documents..."


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
    
    return True, 66, "Loading CNBC articles..."


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
    if not trigger or trigger is None or not scrape_bool:
        raise dash.exceptions.PreventUpdate
    
    # === PROCESS SENTENCES ===
    print("⌛ Processing sentences...")
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT url, content FROM fomc_documents")
    fomc_data = [{"url": row[0], "content": row[1], "type": "fomc"}
                 for row in cursor.fetchall()]

    cursor.execute("SELECT url, content FROM cnbc_articles")
    cnbc_data = [{"url": row[0], "content": row[1], "type": "cnbc"}
                 for row in cursor.fetchall()]

    conn.close()

    all_data = fomc_data + cnbc_data
    filtered_sentences_by_url = {}

    for item in all_data:
        content = item.get("content", "")
        url = item.get("url", "unknown_source")
        source_type = item.get("type", "unknown_type")

        if not content.strip():
            continue

        sentences = sentence_pattern.split(content)
        valid_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            parts = split_pattern.split(sentence)
            parts = [part.strip()
                     for part in parts if part and not re.match(split_pattern, part)]

            for part in parts:
                if len(part.split()) < 3 or part.count('\n') > 3 or len(re.findall(r'[.!?]', part)) < 1:
                    continue

                part_lower = part.lower()

                if any(junk_phrase in part_lower for junk_phrase in junk_phrases):
                    continue

                if any(re.search(rf"\b{re.escape(keyword)}\b", part_lower) for keyword in keywords):
                    valid_sentences.append(part)

        if valid_sentences:
            filtered_sentences_by_url[url] = {
                "type": source_type,
                "sentences": valid_sentences
            }

    print(
        f"✅ Sentence processing complete: {len(filtered_sentences_by_url)} articles with valid content.")

    return filtered_sentences_by_url, 100, "Processing sentences...", "/dashboard"


@callback(
    Output("loading-progress-bar", "value"),
    Input("loading-progress-bar-status", "data"),
    prevent_initial_call=True
)
def update_progress_bar(progress):
    return progress


# @callback(
#     Output("sentences-store", "data"),
#     Output("redirect", "href"),
#     Input("get-started-button", "n_clicks"),
#     prevent_initial_call=True
# )
# def retrieve_data(n_clicks):
#     print("🚀 Starting FOMC + CNBC scrape and sentence processing...")

#     # === Setup Chrome headless ===
#     options = Options()
#     options.add_argument('--headless=new')
#     options.add_argument('--log-level=3')
#     options.add_argument('--no-sandbox')
#     options.add_argument('--disable-dev-shm-usage')
#     driver = webdriver.Chrome(options=options)

#     today = datetime.today()
#     one_week_ago = today - timedelta(days=7)

#     # === FOMC DOCUMENT SCRAPER ===
#     driver.get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
#     time.sleep(2)

#     meeting_blocks = driver.find_elements(By.CSS_SELECTOR, ".fomc-meeting")
#     latest_meeting = None
#     latest_date = None

#     for block in meeting_blocks:
#         try:
#             month_text = block.find_element(
#                 By.CLASS_NAME, "fomc-meeting__month").text.strip()
#             day_text = block.find_element(
#                 By.CLASS_NAME, "fomc-meeting__date").text.strip()
#             first_day = int(re.findall(r"\d+", day_text)[0])
#             year_match = re.search(
#                 r"20\d{2}", block.get_attribute("innerHTML"))
#             year = int(year_match.group()) if year_match else today.year
#             month_num = time.strptime(month_text, '%B').tm_mon
#             date_obj = datetime(year, month_num, first_day)

#             if date_obj <= today and (latest_date is None or date_obj > latest_date):
#                 latest_meeting = block
#                 latest_date = date_obj
#         except:
#             continue

#     if latest_meeting and latest_date:
#         date_str = latest_date.strftime("%Y-%m-%d")
#         full_page_soup = BeautifulSoup(driver.page_source, "html.parser")

#         def scrape_and_insert(doc_type, pattern, is_pdf=False, nested_pdf=False):
#             if nested_pdf:
#                 pressconf_link = full_page_soup.find(
#                     "a", href=re.compile(pattern))
#                 if not pressconf_link:
#                     print(f"❌ Could not find link to {doc_type} HTML page")
#                     return

#                 pressconf_url = urljoin(
#                     "https://www.federalreserve.gov", pressconf_link["href"])
#                 try:
#                     response = requests.get(pressconf_url)
#                     inner_soup = BeautifulSoup(response.text, "html.parser")
#                     pdf_link = inner_soup.find(
#                         "a", href=re.compile(r"FOMCpresconf20\d{6}\.pdf"))
#                     if not pdf_link:
#                         print(f"❌ No {doc_type} PDF found in linked page")
#                         return

#                     pdf_url = urljoin(
#                         "https://www.federalreserve.gov", pdf_link["href"])
#                     response = requests.get(pdf_url)
#                     reader = PdfReader(BytesIO(response.content))
#                     content = "\n".join(page.extract_text()
#                                         or "" for page in reader.pages)
#                     insert_fomc_document(
#                         date_str, doc_type, "PDF", pdf_url, content)
#                 except Exception as e:
#                     print(f"⚠️ Error while processing nested {doc_type}: {e}")
#                 return

#             link = full_page_soup.find("a", href=re.compile(pattern))
#             if not link:
#                 print(f"❌ Could not find direct link for {doc_type}")
#                 return

#             url = urljoin("https://www.federalreserve.gov", link["href"])
#             try:
#                 response = requests.get(url)
#                 if is_pdf:
#                     reader = PdfReader(BytesIO(response.content))
#                     content = "\n".join(page.extract_text()
#                                         or "" for page in reader.pages)
#                 else:
#                     content = BeautifulSoup(response.text, "html.parser").get_text(
#                         separator="\n", strip=True)
#                 insert_fomc_document(date_str, doc_type,
#                                      "PDF" if is_pdf else "HTML", url, content)
#             except Exception as e:
#                 print(f"⚠️ Error while processing {doc_type}: {e}")

#         scrape_and_insert("statement", r"monetary20\d{6}a\.htm")
#         scrape_and_insert("minutes", r"fomcminutes20\d{6}\.htm")
#         scrape_and_insert(
#             "press_conference", r"fomcpresconf20\d{6}\.htm", is_pdf=True, nested_pdf=True)

#     # === CNBC SCRAPER ===
#     driver.get("https://www.cnbc.com/federal-reserve/")
#     time.sleep(5)
#     soup = BeautifulSoup(driver.page_source, 'html.parser')

#     for card in soup.find_all("div", class_="Card-card"):
#         title_tag = card.find("a", class_="Card-title")
#         date_tag = card.find("span", class_="Card-time")
#         if not title_tag or not date_tag:
#             continue

#         try:
#             clean_date = date_tag.text.strip().replace('st', '').replace(
#                 'nd', '').replace('rd', '').replace('th', '')
#             article_date = datetime.strptime(clean_date, "%a, %b %d %Y")
#             if article_date < one_week_ago:
#                 continue

#             article_url = title_tag["href"]
#             driver.get(article_url)
#             time.sleep(2)

#             article_soup = BeautifulSoup(driver.page_source, 'html.parser')
#             summary = article_soup.find_all('li')
#             paragraphs = article_soup.find_all('p')

#             content_parts = [title_tag.text.strip()]
#             if summary:
#                 content_parts.append("Summary:")
#                 content_parts.extend(line.get_text(strip=True)
#                                      for line in summary if line.get_text(strip=True))
#             if paragraphs:
#                 content_parts.append("Body:")
#                 content_parts.extend(p.get_text(strip=True)
#                                      for p in paragraphs if p.get_text(strip=True))

#             content = '\n'.join(content_parts)
#             insert_cnbc_article(title=title_tag.text.strip(
#             ), url=article_url, date=article_date.strftime("%Y-%m-%d"), content=content)

#         except Exception as e:
#             print(f"⚠️ CNBC article scrape error: {e}")

#     driver.quit()

#     # === PROCESS SENTENCES ===
#     print("⌛ Processing sentences...")
#     conn = get_db_connection()
#     cursor = conn.cursor()

#     cursor.execute("SELECT url, content FROM fomc_documents")
#     fomc_data = [{"url": row[0], "content": row[1], "type": "fomc"}
#                  for row in cursor.fetchall()]

#     cursor.execute("SELECT url, content FROM cnbc_articles")
#     cnbc_data = [{"url": row[0], "content": row[1], "type": "cnbc"}
#                  for row in cursor.fetchall()]

#     conn.close()

#     all_data = fomc_data + cnbc_data
#     filtered_sentences_by_url = {}

#     for item in all_data:
#         content = item.get("content", "")
#         url = item.get("url", "unknown_source")
#         source_type = item.get("type", "unknown_type")

#         if not content.strip():
#             continue

#         sentences = sentence_pattern.split(content)
#         valid_sentences = []

#         for sentence in sentences:
#             sentence = sentence.strip()
#             if not sentence:
#                 continue

#             parts = split_pattern.split(sentence)
#             parts = [part.strip()
#                      for part in parts if part and not re.match(split_pattern, part)]

#             for part in parts:
#                 if len(part.split()) < 3 or part.count('\n') > 3 or len(re.findall(r'[.!?]', part)) < 1:
#                     continue

#                 part_lower = part.lower()

#                 if any(junk_phrase in part_lower for junk_phrase in junk_phrases):
#                     continue

#                 if any(re.search(rf"\b{re.escape(keyword)}\b", part_lower) for keyword in keywords):
#                     valid_sentences.append(part)

#         if valid_sentences:
#             filtered_sentences_by_url[url] = {
#                 "type": source_type,
#                 "sentences": valid_sentences
#             }

#     print(
#         f"✅ Sentence processing complete: {len(filtered_sentences_by_url)} articles with valid content.")

#     return filtered_sentences_by_url, "/dashboard"
