"""
Preprocessing Script - Clean Description Column

This script removes unwanted phrases from job descriptions.
Creates a 'cleaned_description' column in the SQLite database.

Usage:
    python preprocessing/clean_description.py
"""

import sqlite3
import re
import sys
from datetime import datetime

# Database path
DB_PATH = 'jobs_database.db'

# List of unwanted phrases to remove
UNWANTED_PHRASES = [
    "job description", "job title", "role description", "about the job",
    "about the role", "about us", "about the opportunity", "requirements",
    "job requirements", "role requirements", "your role", "your job",
    "offer", "employment offer", "your profile", "responsibilities",
    "job responsibilities", "role responsibilities", "overview", "position overview", 
    "who are we?", "who we are", "who are we ?"
]

# Precompile regex patterns for efficiency
PATTERNS = [re.compile(rf"^{phrase}[:\s]*", re.IGNORECASE) for phrase in UNWANTED_PHRASES]


def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def clean_description(text):
    """
    Clean the Description column by removing unwanted phrases at the beginning:
    - "job description:", "about the role:", etc.
    - Removes these phrases only if they appear at the start of the text
    """
    if isinstance(text, str) and text.strip():
        for pattern in PATTERNS:
            text = pattern.sub("", text).strip()  # Remove matched phrase and trim spaces
        return text if text else None
    return None


def check_and_create_column(conn, table_name, column_name, column_type='TEXT'):
    """
    Check if a column exists in the table, and create it if it doesn't.
    """
    cursor = conn.cursor()
    
    # Check if column exists
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    
    if column_name not in columns:
        print(f"Creating column '{column_name}' in table '{table_name}'...")
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
        conn.commit()
        print(f"✓ Column '{column_name}' created successfully")
        return True
    else:
        print(f"✓ Column '{column_name}' already exists")
        return False


def process_description_column():
    """
    Process the Description column and populate cleaned_description column.
    Only updates rows where:
    - Original Description has content
    - cleaned_description is NULL or empty
    """
    print("\n" + "="*60)
    print("PROCESSING DESCRIPTION COLUMN")
    print("="*60)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check/create column
    check_and_create_column(conn, 'jobs', 'cleaned_description', 'TEXT')
    
    # Get rows that need processing
    cursor.execute("""
        SELECT id, description 
        FROM jobs 
        WHERE description IS NOT NULL 
        AND description != ''
        AND (cleaned_description IS NULL OR cleaned_description = '')
    """)
    
    rows_to_process = cursor.fetchall()
    total = len(rows_to_process)
    
    if total == 0:
        print("✓ No rows need processing (all are already cleaned)")
        conn.close()
        return
    
    print(f"\nFound {total} rows to process...")
    
    # Process in batches
    batch_size = 1000
    processed = 0
    updated = 0
    
    for i in range(0, total, batch_size):
        batch = rows_to_process[i:i+batch_size]
        
        for row in batch:
            job_id = row[0]
            description_value = row[1]
            
            # Clean the description
            cleaned = clean_description(description_value)
            
            if cleaned:  # Only update if we got a valid result
                cursor.execute("""
                    UPDATE jobs 
                    SET cleaned_description = ? 
                    WHERE id = ?
                """, (cleaned, job_id))
                updated += 1
            
            processed += 1
            
            # Show progress
            if processed % 100 == 0:
                print(f"  Processed: {processed}/{total} ({processed/total*100:.1f}%)")
        
        # Commit batch
        conn.commit()
    
    print(f"\n✓ Processing complete!")
    print(f"  - Rows processed: {processed}")
    print(f"  - Rows updated: {updated}")
    print(f"  - Rows skipped: {processed - updated}")
    
    # Show sample results
    print("\nSample results (first 3):")
    cursor.execute("""
        SELECT description, cleaned_description 
        FROM jobs 
        WHERE cleaned_description IS NOT NULL 
        LIMIT 3
    """)
    
    for idx, row in enumerate(cursor.fetchall(), 1):
        print(f"\n{idx}. Original (first 150 chars):")
        print(f"   {row[0][:150]}...")
        print(f"   Cleaned (first 150 chars):")
        print(f"   {row[1][:150]}...")
    
    conn.close()


def verify_results():
    """
    Verify the cleaning results with statistics.
    """
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Total rows
    cursor.execute("SELECT COUNT(*) FROM jobs")
    total_jobs = cursor.fetchone()[0]
    
    # Rows with description data
    cursor.execute("SELECT COUNT(*) FROM jobs WHERE description IS NOT NULL AND description != ''")
    with_descriptions = cursor.fetchone()[0]
    
    # Rows with cleaned descriptions
    cursor.execute("SELECT COUNT(*) FROM jobs WHERE cleaned_description IS NOT NULL AND cleaned_description != ''")
    with_cleaned = cursor.fetchone()[0]
    
    print(f"\nStatistics:")
    print(f"  Total jobs in database: {total_jobs:,}")
    print(f"  Jobs with description data: {with_descriptions:,} ({with_descriptions/total_jobs*100:.1f}%)")
    print(f"  Jobs with cleaned descriptions: {with_cleaned:,} ({with_cleaned/total_jobs*100:.1f}%)")
    
    # Check for any remaining unprocessed rows
    cursor.execute("""
        SELECT COUNT(*) 
        FROM jobs 
        WHERE description IS NOT NULL 
        AND description != ''
        AND (cleaned_description IS NULL OR cleaned_description = '')
    """)
    unprocessed = cursor.fetchone()[0]
    
    if unprocessed > 0:
        print(f"\n⚠ Warning: {unprocessed} rows still need processing")
    else:
        print(f"\n✓ All rows with description data have been processed!")
    
    # Show examples of phrases removed
    print("\n" + "="*60)
    print("EXAMPLES OF CLEANED PHRASES")
    print("="*60)
    
    cursor.execute("""
        SELECT description, cleaned_description 
        FROM jobs 
        WHERE description != cleaned_description
        AND cleaned_description IS NOT NULL
        LIMIT 5
    """)
    
    examples = cursor.fetchall()
    if examples:
        print("\nShowing jobs where unwanted phrases were removed:\n")
        for idx, row in enumerate(examples, 1):
            orig_start = row[0][:100]
            clean_start = row[1][:100]
            print(f"{idx}. BEFORE: {orig_start}...")
            print(f"   AFTER:  {clean_start}...")
            print()
    else:
        print("\nNo examples found (descriptions may not start with unwanted phrases)")
    
    conn.close()


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("  JOBS DATABASE - DESCRIPTION PREPROCESSING")
    print("="*60)
    print(f"\nDatabase: {DB_PATH}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Unwanted phrases to remove: {len(UNWANTED_PHRASES)}\n")
    
    try:
        # Process description column
        process_description_column()
        
        # Verify results
        verify_results()
        
        print("\n" + "="*60)
        print("✓ PREPROCESSING COMPLETED SUCCESSFULLY")
        print("="*60 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())