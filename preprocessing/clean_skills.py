"""
Preprocessing Script for Jobs Database

This script applies cleaning functions to the SQLite database.
It creates new cleaned columns and populates them safely.

Usage:
    python preprocessing/clean_skills.py
"""

import sqlite3
import re
import sys
from datetime import datetime

# Database path
DB_PATH = 'jobs_database.db'


def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def clean_skills(skill_str):
    """
    Clean the Skill column by removing:
    - "Skills: " prefix
    - "X of Y skills match your profile..." pattern
    - ", +X more" suffix
    """
    if isinstance(skill_str, str) and skill_str.strip():
        # Remove "Skills: " if it appears at the beginning
        skill_str = re.sub(r"^Skills:\s*", "", skill_str).strip()
        # Remove "X of Y skills match your profile - you may be ..." pattern
        skill_str = re.sub(r"\d+\s+of\s+\d+\s+skills match your profile - you may be.*", "", skill_str, flags=re.IGNORECASE).strip()
        # Remove ", +X more" where X is any number
        skill_str = re.sub(r",\s*\+\d+\s+more", "", skill_str).strip()
        return skill_str
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


def process_skills_column():
    """
    Process the Skill column and populate cleaned_skills column.
    Only updates rows where:
    - Original Skill has content
    - cleaned_skills is NULL or empty
    """
    print("\n" + "="*60)
    print("PROCESSING SKILLS COLUMN")
    print("="*60)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check/create column
    check_and_create_column(conn, 'jobs', 'cleaned_skills', 'TEXT')
    
    # Get rows that need processing
    cursor.execute("""
        SELECT id, skill 
        FROM jobs 
        WHERE skill IS NOT NULL 
        AND skill != ''
        AND (cleaned_skills IS NULL OR cleaned_skills = '')
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
            skill_value = row[1]
            
            # Clean the skill
            cleaned = clean_skills(skill_value)
            
            if cleaned:  # Only update if we got a valid result
                cursor.execute("""
                    UPDATE jobs 
                    SET cleaned_skills = ? 
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
    print("\nSample results (first 5):")
    cursor.execute("""
        SELECT skill, cleaned_skills 
        FROM jobs 
        WHERE cleaned_skills IS NOT NULL 
        LIMIT 5
    """)
    
    for idx, row in enumerate(cursor.fetchall(), 1):
        print(f"\n{idx}. Original:")
        print(f"   {row[0][:100]}...")
        print(f"   Cleaned:")
        print(f"   {row[1][:100]}...")
    
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
    
    # Rows with skill data
    cursor.execute("SELECT COUNT(*) FROM jobs WHERE skill IS NOT NULL AND skill != ''")
    with_skills = cursor.fetchone()[0]
    
    # Rows with cleaned skills
    cursor.execute("SELECT COUNT(*) FROM jobs WHERE cleaned_skills IS NOT NULL AND cleaned_skills != ''")
    with_cleaned = cursor.fetchone()[0]
    
    print(f"\nStatistics:")
    print(f"  Total jobs in database: {total_jobs:,}")
    print(f"  Jobs with skill data: {with_skills:,} ({with_skills/total_jobs*100:.1f}%)")
    print(f"  Jobs with cleaned skills: {with_cleaned:,} ({with_cleaned/total_jobs*100:.1f}%)")
    
    # Check for any remaining unprocessed rows
    cursor.execute("""
        SELECT COUNT(*) 
        FROM jobs 
        WHERE skill IS NOT NULL 
        AND skill != ''
        AND (cleaned_skills IS NULL OR cleaned_skills = '')
    """)
    unprocessed = cursor.fetchone()[0]
    
    if unprocessed > 0:
        print(f"\n⚠ Warning: {unprocessed} rows still need processing")
    else:
        print(f"\n✓ All rows with skill data have been processed!")
    
    conn.close()


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("  JOBS DATABASE - SKILLS PREPROCESSING")
    print("="*60)
    print(f"\nDatabase: {DB_PATH}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Process skills column
        process_skills_column()
        
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