import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "fed_data.db")


def init_db():
    print(f"📍 DB_PATH: {DB_PATH}")
    if os.path.exists(DB_PATH):
        print("✅ Database already exists. Skipping creation.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # FOMC documents table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fomc_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            type TEXT,
            source_type TEXT,
            url TEXT,
            content TEXT,
            UNIQUE(date, type)
        )
    """)

    # CNBC news articles table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cnbc_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            url TEXT,
            date TEXT,
            content TEXT,
            UNIQUE(title)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE sentences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sentence TEXT NOT NULL UNIQUE,
            sentiment REAL,
            source_type TEXT CHECK(source_type IN ('fomc', 'cnbc')),
            url TEXT,
            date TEXT
        )
    """)

    conn.commit()
    conn.close()
    print("✅ Database and tables created.")


def get_db_connection():
    return sqlite3.connect(DB_PATH)


def insert_fomc_document(date, doc_type, source_type, url, content):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT 1 FROM fomc_documents WHERE date = ? AND type = ?
    """, (date, doc_type))

    if cursor.fetchone():
        print(
            f"⚠️ Skipped FOMC document ({doc_type}) for {date} (already exists)")
        conn.close()
        return

    cursor.execute("""
        INSERT INTO fomc_documents (date, type, source_type, url, content)
        VALUES (?, ?, ?, ?, ?)
    """, (date, doc_type, source_type, url, content))

    print(f"✅ Inserted {doc_type}: {url}")
    conn.commit()
    conn.close()


def insert_cnbc_article(title, url, date, content):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT 1 FROM cnbc_articles WHERE title = ?", (title,))
    
    if cursor.fetchone():
        print(f"⚠️ Skipped CNBC article '{title}' (already exists)")
        conn.close()
        return

    cursor.execute("""
        INSERT INTO cnbc_articles (title, url, date, content)
        VALUES (?, ?, ?, ?)
    """, (title, url, date, content))
    
    print(f"✅ Inserted CNBC: {title}")
    conn.commit()
    conn.close()
