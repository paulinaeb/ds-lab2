"""
Preprocessing Script - Extract Country from Location

This script extracts country information from the location field.
Creates a 'country' column in the SQLite database.

Usage:
    python preprocessing/extract_country.py
"""

import sqlite3
import re
import sys
from datetime import datetime

# Database path
DB_PATH = 'jobs_database.db'

# Metropolitan area to country mapping
METRO_TO_COUNTRY = {
    # Spain
    "Greater Madrid Metropolitan Area": "Spain",
    "Greater Barcelona Metropolitan Area": "Spain",
    "Greater Sevilla Metropolitan Area": "Spain",
    "Greater Zaragoza Metropolitan Area": "Spain",
    
    # France
    "Greater Paris Metropolitan Region": "France",
    "Greater Lille Metropolitan Area": "France",
    "Greater Marseille Metropolitan Area": "France",
    "Greater Toulon Metropolitan Area": "France",
    
    # United Kingdom & Ireland
    "Greater London": "United Kingdom",
    "Cork Metropolitan Area": "Ireland",
    
    # Netherlands
    "Greater Amsterdam": "Netherlands",
    "Amsterdam Area": "Netherlands",
    "Greater Groningen Area": "Netherlands",
    "Rotterdam and The Hague": "Netherlands",
    "Arnhem-Nijmegen Region": "Netherlands",
    
    # Germany
    "Greater Berlin Metropolitan Area": "Germany",
    "Greater Munich Metropolitan Area": "Germany",
    "Frankfurt Rhine-Main Metropolitan Area": "Germany",
    "Stuttgart Region": "Germany",
    "Cologne Bonn Region": "Germany",
    "Greater Kempten Area": "Germany",
    
    # Italy
    "Greater Milan Metropolitan Area": "Italy",
    "Greater Rome Metropolitan Area": "Italy",
    
    # Portugal
    "Lisbon Metropolitan Area": "Portugal",
    "Porto Metropolitan Area": "Portugal",
    
    # Belgium
    "Brussels Metropolitan Area": "Belgium",
    
    # Romania
    "Bucharest Metropolitan Area": "Romania",
    
    # Finland
    "Tampere Metropolitan Area": "Finland",
    
    # Poland
    "Gdansk Metropolitan Area": "Poland",
    "Wroclaw Metropolitan Area": "Poland",
    "Warsaw Metropolitan Area": "Poland",
    
    # Hungary
    "Budapest Metropolitan Area": "Hungary",
    
    # Sweden
    "Greater Stockholm Metropolitan Area": "Sweden",
    
    # Denmark
    "Copenhagen Metropolitan Area": "Denmark",
    
    # Estonia
    "Tallinn Metropolitan Area": "Estonia",
    
    # Czech Republic
    "Greater Hradec Kralove Area": "Czechia",
    
    # Slovakia
    "Bratislava Metropolitan Area": "Slovakia"
}

# Patterns to exclude (not actual countries)
EXCLUDE_PATTERNS = [
    "EMEA",
    "European Union",
    "Europe",
    "Remote",
    "Hybrid",
    "On-site"
]


def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def extract_country(location_str):
    """
    Extract country from location string.
    
    Logic:
    1. First, check if it's a known metropolitan area
    2. If location contains comma, take text after last comma
    3. Otherwise, use the location as-is
    4. Exclude non-country patterns (EMEA, European Union)
    """
    if not isinstance(location_str, str) or not location_str.strip():
        return None
    
    location_str = location_str.strip()
    
    # Check if it's a known metropolitan area
    for metro, country in METRO_TO_COUNTRY.items():
        if metro.lower() in location_str.lower():
            return country
    
    # Check if we should exclude this location
    for pattern in EXCLUDE_PATTERNS:
        if pattern.lower() in location_str.lower():
            return None
    
    # Extract country (text after last comma, or entire string)
    if "," in location_str:
        country = location_str.split(",")[-1].strip()
    else:
        country = location_str.strip()
    
    # Final validation - return None if empty or too short
    if len(country) < 2:
        return None
    
    return country


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


def process_location_column():
    """
    Process the Location column and populate country column.
    Only updates rows where:
    - Original Location has content
    - country is NULL or empty
    """
    print("\n" + "="*60)
    print("PROCESSING LOCATION COLUMN - EXTRACTING COUNTRY")
    print("="*60)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check/create column
    check_and_create_column(conn, 'jobs', 'country', 'TEXT')
    
    # Get rows that need processing
    cursor.execute("""
        SELECT id, location 
        FROM jobs 
        WHERE location IS NOT NULL 
        AND location != ''
        AND (country IS NULL OR country = '')
    """)
    
    rows_to_process = cursor.fetchall()
    total = len(rows_to_process)
    
    if total == 0:
        print("✓ No rows need processing (all countries already extracted)")
        conn.close()
        return
    
    print(f"\nFound {total} rows to process...")
    
    # Process in batches
    batch_size = 1000
    processed = 0
    updated = 0
    excluded = 0
    
    for i in range(0, total, batch_size):
        batch = rows_to_process[i:i+batch_size]
        
        for row in batch:
            job_id = row[0]
            location_value = row[1]
            
            # Extract country
            country = extract_country(location_value)
            
            if country:  # Only update if we got a valid result
                cursor.execute("""
                    UPDATE jobs 
                    SET country = ? 
                    WHERE id = ?
                """, (country, job_id))
                updated += 1
            else:
                excluded += 1
            
            processed += 1
            
            # Show progress
            if processed % 100 == 0:
                print(f"  Processed: {processed}/{total} ({processed/total*100:.1f}%)")
        
        # Commit batch
        conn.commit()
    
    print(f"\n✓ Processing complete!")
    print(f"  - Rows processed: {processed}")
    print(f"  - Rows updated with country: {updated}")
    print(f"  - Rows excluded (EMEA/EU/etc.): {excluded}")
    
    # Show sample results
    print("\nSample results (first 10):")
    cursor.execute("""
        SELECT location, country 
        FROM jobs 
        WHERE country IS NOT NULL 
        LIMIT 10
    """)
    
    print(f"\n{'Location':<50} {'Country':<20}")
    print("-" * 70)
    for row in cursor.fetchall():
        location = row[0][:48] + ".." if len(row[0]) > 48 else row[0]
        print(f"{location:<50} {row[1]:<20}")
    
    conn.close()


def verify_results():
    """
    Verify the extraction results with statistics.
    """
    print("\n" + "="*60)
    print("VERIFICATION & STATISTICS")
    print("="*60)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Total rows
    cursor.execute("SELECT COUNT(*) FROM jobs")
    total_jobs = cursor.fetchone()[0]
    
    # Rows with location data
    cursor.execute("SELECT COUNT(*) FROM jobs WHERE location IS NOT NULL AND location != ''")
    with_location = cursor.fetchone()[0]
    
    # Rows with country
    cursor.execute("SELECT COUNT(*) FROM jobs WHERE country IS NOT NULL AND country != ''")
    with_country = cursor.fetchone()[0]
    
    # Rows excluded
    excluded = with_location - with_country
    
    print(f"\nStatistics:")
    print(f"  Total jobs in database: {total_jobs:,}")
    print(f"  Jobs with location data: {with_location:,} ({with_location/total_jobs*100:.1f}%)")
    print(f"  Jobs with country extracted: {with_country:,} ({with_country/total_jobs*100:.1f}%)")
    print(f"  Jobs excluded (EMEA/EU): {excluded:,} ({excluded/with_location*100:.1f}% of locations)")
    
    # Show top countries
    print("\n" + "="*60)
    print("TOP 20 COUNTRIES BY JOB COUNT")
    print("="*60)
    
    cursor.execute("""
        SELECT country, COUNT(*) as job_count 
        FROM jobs 
        WHERE country IS NOT NULL 
        GROUP BY country 
        ORDER BY job_count DESC 
        LIMIT 20
    """)
    
    print(f"\n{'Rank':<6} {'Country':<30} {'Jobs':<10} {'%':<10}")
    print("-" * 60)
    
    for idx, row in enumerate(cursor.fetchall(), 1):
        percentage = (row[1] / with_country) * 100
        print(f"{idx:<6} {row[0]:<30} {row[1]:<10,} {percentage:<10.1f}%")
    
    # Check for any remaining unprocessed rows
    cursor.execute("""
        SELECT COUNT(*) 
        FROM jobs 
        WHERE location IS NOT NULL 
        AND location != ''
        AND (country IS NULL OR country = '')
    """)
    unprocessed = cursor.fetchone()[0]
    
    print("\n" + "="*60)
    if unprocessed > 0:
        print(f"⚠ Warning: {unprocessed} rows still need processing")
        
        # Show examples of unprocessed locations
        cursor.execute("""
            SELECT location 
            FROM jobs 
            WHERE location IS NOT NULL 
            AND location != ''
            AND (country IS NULL OR country = '')
            LIMIT 5
        """)
        print("\nExamples of unprocessed locations:")
        for row in cursor.fetchall():
            print(f"  - {row[0]}")
    else:
        print(f"✓ All rows with location data have been processed!")
    
    conn.close()


def show_metropolitan_areas():
    """
    Show which metropolitan areas were mapped to countries.
    """
    print("\n" + "="*60)
    print("METROPOLITAN AREA MAPPINGS")
    print("="*60)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print(f"\n{'Metropolitan Area':<50} {'Mapped to Country':<20} {'Count':<10}")
    print("-" * 80)
    
    for metro, country in METRO_TO_COUNTRY.items():
        cursor.execute("""
            SELECT COUNT(*) 
            FROM jobs 
            WHERE location LIKE ? 
            AND country = ?
        """, (f"%{metro}%", country))
        
        count = cursor.fetchone()[0]
        if count > 0:
            print(f"{metro:<50} {country:<20} {count:<10,}")
    
    conn.close()


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("  JOBS DATABASE - LOCATION TO COUNTRY EXTRACTION")
    print("="*60)
    print(f"\nDatabase: {DB_PATH}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Metropolitan areas configured: {len(METRO_TO_COUNTRY)}")
    print(f"Exclude patterns: {', '.join(EXCLUDE_PATTERNS)}\n")
    
    try:
        # Process location column
        process_location_column()
        
        # Verify results
        verify_results()
        
        # Show metropolitan area mappings
        show_metropolitan_areas()
        
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