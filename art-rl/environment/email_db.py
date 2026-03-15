"""Email database operations using SQLite with FTS5 for full-text search."""

import os
import sqlite3
from datetime import datetime
from typing import List, Optional

from datasets import Features, Sequence, Value, load_dataset
from tqdm import tqdm

from environment.models import Email, SearchResult

# Database configuration
DEFAULT_DB_PATH = "./enron_emails.db"
EMAIL_DATASET_REPO_ID = "corbt/enron-emails"

# Module-level connection cache
_db_connections: dict[str, sqlite3.Connection] = {}


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Get or create a database connection.

    Uses a module-level cache to reuse connections within the same process.
    """
    if db_path not in _db_connections:
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"Database not found at {db_path}. "
                "Run the data preparation pipeline first."
            )
        _db_connections[db_path] = sqlite3.connect(
            db_path, check_same_thread=False
        )
    return _db_connections[db_path]


def create_email_database(
    db_path: str = DEFAULT_DB_PATH,
    max_body_length: int = 5000,
    max_recipients: int = 30,
) -> str:
    """Create the email database from Hugging Face dataset.

    Filters out emails that are too long or have too many recipients,
    and deduplicates based on (subject, body, from_address).

    Returns:
        Path to the created database.
    """
    print("Creating email database from Hugging Face dataset...")
    print("This may take several minutes for the full Enron dataset...")

    # Database schema with FTS5 for full-text search
    sql_create_tables = """
    DROP TABLE IF EXISTS recipients;
    DROP TABLE IF EXISTS emails_fts;
    DROP TABLE IF EXISTS emails;

    CREATE TABLE emails (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id TEXT UNIQUE,
        subject TEXT,
        from_address TEXT,
        date TEXT,
        body TEXT,
        file_name TEXT
    );

    CREATE TABLE recipients (
        email_id TEXT,
        recipient_address TEXT,
        recipient_type TEXT
    );
    """

    sql_create_indexes = """
    CREATE INDEX idx_emails_from ON emails(from_address);
    CREATE INDEX idx_emails_date ON emails(date);
    CREATE INDEX idx_emails_message_id ON emails(message_id);
    CREATE INDEX idx_recipients_address ON recipients(recipient_address);
    CREATE INDEX idx_recipients_type ON recipients(recipient_type);
    CREATE INDEX idx_recipients_email_id ON recipients(email_id);
    CREATE INDEX idx_recipients_address_email
        ON recipients(recipient_address, email_id);

    CREATE VIRTUAL TABLE emails_fts USING fts5(
        subject,
        body,
        content='emails',
        content_rowid='id'
    );

    CREATE TRIGGER emails_ai AFTER INSERT ON emails BEGIN
        INSERT INTO emails_fts (rowid, subject, body)
        VALUES (new.id, new.subject, new.body);
    END;

    CREATE TRIGGER emails_ad AFTER DELETE ON emails BEGIN
        DELETE FROM emails_fts WHERE rowid=old.id;
    END;

    CREATE TRIGGER emails_au AFTER UPDATE ON emails BEGIN
        UPDATE emails_fts SET subject=new.subject, body=new.body
        WHERE rowid=old.id;
    END;
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executescript(sql_create_tables)
    conn.commit()

    # Load dataset with expected schema
    print("Loading email dataset from Hugging Face...")
    expected_features = Features(
        {
            "message_id": Value("string"),
            "subject": Value("string"),
            "from": Value("string"),
            "to": Sequence(Value("string")),
            "cc": Sequence(Value("string")),
            "bcc": Sequence(Value("string")),
            "date": Value("timestamp[us]"),
            "body": Value("string"),
            "file_name": Value("string"),
        }
    )

    dataset = load_dataset(
        EMAIL_DATASET_REPO_ID, features=expected_features, split="train"
    )
    print(f"Dataset contains {len(dataset)} total emails")

    # Populate database with filtering and deduplication
    print("Populating database...")
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = MEMORY;")
    conn.execute("BEGIN TRANSACTION;")

    record_count = 0
    skipped_count = 0
    duplicate_count = 0
    processed_emails: set[tuple] = set()

    for email_data in tqdm(dataset, desc="Inserting emails"):
        message_id = email_data["message_id"]
        subject = email_data["subject"]
        from_address = email_data["from"]
        date_obj: datetime = email_data["date"]
        body = email_data["body"]
        file_name = email_data["file_name"]
        to_list = [str(addr) for addr in email_data["to"] if addr]
        cc_list = [str(addr) for addr in email_data["cc"] if addr]
        bcc_list = [str(addr) for addr in email_data["bcc"] if addr]

        total_recipients = len(to_list) + len(cc_list) + len(bcc_list)

        # Filter out very long emails and those with too many recipients
        if len(body) > max_body_length:
            skipped_count += 1
            continue

        if total_recipients > max_recipients:
            skipped_count += 1
            continue

        # Deduplication check
        email_key = (subject, body, from_address)
        if email_key in processed_emails:
            duplicate_count += 1
            continue
        processed_emails.add(email_key)

        date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute(
            """
            INSERT INTO emails
                (message_id, subject, from_address, date, body, file_name)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (message_id, subject, from_address, date_str, body, file_name),
        )

        # Insert recipients
        recipient_data = []
        for addr in to_list:
            recipient_data.append((message_id, addr, "to"))
        for addr in cc_list:
            recipient_data.append((message_id, addr, "cc"))
        for addr in bcc_list:
            recipient_data.append((message_id, addr, "bcc"))

        if recipient_data:
            cursor.executemany(
                """
                INSERT INTO recipients
                    (email_id, recipient_address, recipient_type)
                VALUES (?, ?, ?)
                """,
                recipient_data,
            )

        record_count += 1

    conn.commit()

    # Create indexes and FTS triggers
    print("Creating indexes and FTS...")
    cursor.executescript(sql_create_indexes)
    cursor.execute('INSERT INTO emails_fts(emails_fts) VALUES("rebuild")')
    conn.commit()

    print(f"Successfully created database with {record_count} emails.")
    print(f"Skipped {skipped_count} emails due to length/recipient limits.")
    print(f"Skipped {duplicate_count} duplicate emails.")

    # Cache the connection
    _db_connections[db_path] = conn

    return db_path


def search_emails(
    inbox: str,
    keywords: List[str],
    db_path: str = DEFAULT_DB_PATH,
    from_addr: Optional[str] = None,
    to_addr: Optional[str] = None,
    sent_after: Optional[str] = None,
    sent_before: Optional[str] = None,
    max_results: int = 10,
) -> List[SearchResult]:
    """Search the email database based on keywords and filters.

    Args:
        inbox: Email address of the inbox to search.
        keywords: List of keywords to search for (AND logic).
        db_path: Path to the SQLite database.
        from_addr: Optional filter for sender address.
        to_addr: Optional filter for recipient address.
        sent_after: Optional date filter (YYYY-MM-DD).
        sent_before: Optional date filter (YYYY-MM-DD).
        max_results: Maximum number of results to return.

    Returns:
        List of SearchResult objects with message_id and snippet.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    where_clauses: List[str] = []
    params: List[str | int] = []

    if not keywords:
        raise ValueError("No keywords provided for search.")

    if max_results > 10:
        raise ValueError("max_results must be less than or equal to 10.")

    # FTS5 query - escape quotes for safety
    fts_query = " ".join(
        f'"{k.replace(chr(34), chr(34)+chr(34))}"' for k in keywords
    )
    where_clauses.append("fts.emails_fts MATCH ?")
    params.append(fts_query)

    # Inbox filter - emails sent by or received by this address
    where_clauses.append(
        """
        (e.from_address = ? OR EXISTS (
            SELECT 1 FROM recipients r_inbox
            WHERE r_inbox.recipient_address = ?
              AND r_inbox.email_id = e.message_id
        ))
        """
    )
    params.extend([inbox, inbox])

    if from_addr:
        where_clauses.append("e.from_address = ?")
        params.append(from_addr)

    if to_addr:
        where_clauses.append(
            """
            EXISTS (
                SELECT 1 FROM recipients r_to
                WHERE r_to.recipient_address = ?
                  AND r_to.email_id = e.message_id
            )
            """
        )
        params.append(to_addr)

    if sent_after:
        where_clauses.append("e.date >= ?")
        params.append(f"{sent_after} 00:00:00")

    if sent_before:
        where_clauses.append("e.date < ?")
        params.append(f"{sent_before} 00:00:00")

    sql = f"""
        SELECT
            e.message_id,
            snippet(emails_fts, -1, '<b>', '</b>', ' ... ', 15) as snippet
        FROM
            emails e JOIN emails_fts fts ON e.id = fts.rowid
        WHERE
            {" AND ".join(where_clauses)}
        ORDER BY
            e.date DESC
        LIMIT ?;
    """
    params.append(max_results)

    cursor.execute(sql, params)
    results = cursor.fetchall()

    return [SearchResult(message_id=row[0], snippet=row[1]) for row in results]


def read_email(
    message_id: str,
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[Email]:
    """Retrieve a single email by its message_id.

    Args:
        message_id: The unique message ID of the email.
        db_path: Path to the SQLite database.

    Returns:
        Email object if found, None otherwise.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT message_id, date, subject, from_address, body, file_name
        FROM emails WHERE message_id = ?
        """,
        (message_id,),
    )
    email_row = cursor.fetchone()

    if not email_row:
        return None

    msg_id, date, subject, from_addr, body, file_name = email_row

    # Get recipients
    cursor.execute(
        "SELECT recipient_address, recipient_type FROM recipients "
        "WHERE email_id = ?",
        (message_id,),
    )
    recipient_rows = cursor.fetchall()

    to_addresses = []
    cc_addresses = []
    bcc_addresses = []

    for addr, type_val in recipient_rows:
        if type_val.lower() == "to":
            to_addresses.append(addr)
        elif type_val.lower() == "cc":
            cc_addresses.append(addr)
        elif type_val.lower() == "bcc":
            bcc_addresses.append(addr)

    return Email(
        message_id=msg_id,
        date=date,
        subject=subject,
        from_address=from_addr,
        to_addresses=to_addresses,
        cc_addresses=cc_addresses,
        bcc_addresses=bcc_addresses,
        body=body,
        file_name=file_name,
    )
