# FEDDIE: Macroeconomic Sentiment & Data Dashboard  

<p align="center">
  <img src="https://github.com/user-attachments/assets/4e83f6e9-b3ff-43fb-a278-19b327872248" 
       alt="FEDDIE_LOGO" 
       width="500" 
       height="500">
</p>

<br>

**FEDDIE** is a full-stack system that integrates:  

- **Macroeconomic data ingestion & visualization** (FRED)  
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
                       
- **Backend:** Python, Dash, Flask API  
- **Database:** SQLite (local)
- **Model Training:** PyTorch, Hugging Face Transformers  
- **Deployment:** Dockerized API + Dash frontend
  
---

## 🧠 Model Training  

### Dataset  
- **FOMC Minutes/Statements** (primary source of Fed communication)  
- **CNBC News Articles** (proxy for market/media interpretation)  
- Sentences are labeled **hawkish/dovish/neutral**.  

### Fine-tuning Steps  

1. Preprocess sentences (tokenization, truncation to 512 tokens).  
2. Split into train/val/test sets.  
3. Fine-tune transformer with **class-balanced loss** to address imbalance.  
4. Optimize for **macro-F1** and **minority recall** (important for minority hawkish and dovish class).  

```
python train.py \
    --model roberta-large \
    --train_data data/train.csv \
    --val_data data/val.csv \
    --epochs 5 \
    --batch_size 16 \
    --lr 2e-5 \
    --save_dir models/feddie_roberta
```

### Referenced Methodology  
Fine-tuning approach adapted from:
> Shapiro, A. & Wilson, B. (2022). Hawkish or Dovish? Detecting Monetary Policy Stance in Central Bank Communication Using Transformers. Journal of Financial Econometrics.

---

## 📊 Economic Context

Why this matters:
- Fed communication strongly influences market expectations and asset pricing.
- Detecting shifts in tone (hawkish vs dovish) provides early insight into monetary policy stance.
- This project showcases how econometric reasoning (sentiment as a proxy for stance) and ML techniques (transformer fine-tuning) can work together.
