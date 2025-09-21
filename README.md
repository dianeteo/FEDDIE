# FEDDIE: Macroeconomic Sentiment & Data Dashboard  

![FEDDIE Logo](assets/feddie_logo.png)  

**FEDDIE** (Federal Reserve Economic Data & Documents Intelligence Engine) is a full-stack system that integrates:  

- **Macroeconomic data ingestion & visualization** (FRED, BLS, Yahoo Finance, etc.)  
- **NLP-based sentiment analysis** on FOMC minutes, statements, and CNBC news articles  
- **Fine-tuned LLMs** for classification of monetary policy stance (hawkish vs dovish)  
- **Interactive dashboard** for visual exploration of macroeconomic indicators and sentiment trends  

The project demonstrates how **economic research and ML engineering** can be combined to monitor policy communication and market reaction in real time.  

---

## 📑 Features  

- Scrapes and stores **FOMC documents** and **financial news articles** in SQLite  
- Fine-tunes transformer LLMs (RoBERTa-large, Gemma-2, etc.) on hawkish/dovish classification task  
- Supports **sequential scraping pipelines** (FOMC → CNBC → Sentence extraction)  
- Exposes a **REST API** for sentiment scoring and retrieval  
- Dash-based **Macroeconomic Dashboard** with interactive controls (date range, frequency, units, etc.)  

---

## ⚙️ Technical Architecture  

```mermaid
flowchart TD
    A[Data Sources: FRED, FOMC, CNBC] -->|scraping| B[SQLite Database]
    B -->|ETL| C[Sentence Processor]
    C --> D[Fine-tuned LLM Classifier]
    D -->|scores| E[Sentiment Store]
    E --> F[REST API]
    F --> G[Dash Frontend]
    H[Users] --> G
