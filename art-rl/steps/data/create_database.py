"""Step to create the SQLite email database with FTS5 search."""

import os
import sqlite3
from datetime import datetime
from typing import Annotated

from datasets import Dataset
from tqdm import tqdm
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def create_database(
    raw_emails: Dataset,
    db_path: str = "./enron_emails.db",
    max_body_length: int = 5000,
    max_recipients: int = 30,
) -> Annotated[str, "db_path"]:
    """Create a SQLite database with FTS5 full-text search from raw emails.

    This step processes the raw email dataset and creates an optimized
    SQLite database with:
    - Emails table with core email fields
    - Recipients table for to/cc/bcc addresses
    - FTS5 virtual table for fast full-text search
    - Indexes for common query patterns

    Emails are filtered to exclude:
    - Very long emails (> max_body_length characters)
    - Emails with too many recipients (> max_recipients)
    - Duplicate emails (same subject, body, and sender)

    Args:
        raw_emails: The raw email dataset from Hugging Face.
        db_path: Path where the SQLite database will be created.
        max_body_length: Maximum allowed email body length.
        max_recipients: Maximum allowed total recipients.

    Returns:
        Path to the created database file.
    """
    logger.info(f"Creating email database at {db_path}...")

    # Remove existing database if present
    if os.path.exists(db_path):
        os.remove(db_path)

    # Database schema
    sql_create_tables = """
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

    # Optimize for bulk inserts
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = MEMORY;")
    conn.execute("BEGIN TRANSACTION;")

    record_count = 0
    skipped_count = 0
    duplicate_count = 0
    processed_emails: set[tuple] = set()

    for email_data in tqdm(raw_emails, desc="Inserting emails"):
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

        # Apply filters
        if len(body) > max_body_length:
            skipped_count += 1
            continue

        if total_recipients > max_recipients:
            skipped_count += 1
            continue

        # Deduplication
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

    # Create indexes and FTS
    logger.info("Creating indexes and FTS virtual table...")
    cursor.executescript(sql_create_indexes)
    cursor.execute('INSERT INTO emails_fts(emails_fts) VALUES("rebuild")')
    conn.commit()
    conn.close()

    logger.info(f"Created database with {record_count} emails")
    logger.info(f"Skipped {skipped_count} emails (length/recipient limits)")
    logger.info(f"Skipped {duplicate_count} duplicate emails")

    return db_path
