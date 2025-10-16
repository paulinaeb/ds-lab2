import sqlite3
import json
from datetime import datetime
import os

def create_database_and_import():
    """
    Creates a SQLite database with a jobs table and imports data from data_cleaned.json
    """
    
    # Database file name
    db_name = 'jobs_database.db'
    
    try:
        # Connect to SQLite database (creates it if it doesn't exist)
        print("Connecting to SQLite database...")
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        # Create the jobs table
        print("Creating jobs table...")
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            description TEXT,
            primary_description TEXT,
            detail_url TEXT UNIQUE,
            location TEXT,
            skill TEXT,
            insight TEXT,
            job_state TEXT,
            poster_id TEXT,
            company_name TEXT,
            company_logo TEXT,
            created_at TEXT,
            scraped_at TEXT,
            imported_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        '''
        cursor.execute(create_table_query)
        
        # Create indexes for better query performance
        print("Creating indexes...")
        indexes = [
            'CREATE INDEX IF NOT EXISTS idx_company_name ON jobs(company_name)',
            'CREATE INDEX IF NOT EXISTS idx_location ON jobs(location)',
            'CREATE INDEX IF NOT EXISTS idx_job_state ON jobs(job_state)',
            'CREATE INDEX IF NOT EXISTS idx_created_at ON jobs(created_at)',
            'CREATE INDEX IF NOT EXISTS idx_detail_url ON jobs(detail_url)'
        ]
        
        for index_query in indexes:
            cursor.execute(index_query)
        
        # Check if cleaned data file exists
        if not os.path.exists('data_cleaned.json'):
            print("Error: data_cleaned.json not found. Please run clean_json.py first.")
            return
        
        # Read the cleaned JSON data
        print("Reading cleaned JSON data...")
        with open('data_cleaned.json', 'r', encoding='utf-8') as file:
            jobs_data = json.load(file)
        
        print(f"Found {len(jobs_data)} jobs to import")
        
        # Prepare insert query
        insert_query = '''
        INSERT OR IGNORE INTO jobs (
            title, description, primary_description, detail_url, location,
            skill, insight, job_state, poster_id, company_name,
            company_logo, created_at, scraped_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        
        # Import data in batches
        batch_size = 1000
        imported_count = 0
        skipped_count = 0
        
        for i in range(0, len(jobs_data), batch_size):
            batch = jobs_data[i:i + batch_size]
            batch_data = []
            
            for job in batch:
                # Prepare data tuple, handling missing keys gracefully
                job_tuple = (
                    job.get('Title'),
                    job.get('Description'),
                    job.get('Primary Description'),
                    job.get('Detail URL'),
                    job.get('Location'),
                    job.get('Skill'),
                    job.get('Insight'),
                    job.get('Job State'),
                    job.get('Poster Id'),
                    job.get('Company Name'),
                    job.get('Company Logo'),
                    job.get('Created At'),
                    job.get('Scraped At')
                )
                batch_data.append(job_tuple)
            
            # Execute batch insert
            try:
                cursor.executemany(insert_query, batch_data)
                imported_count += len(batch_data)
                print(f"Imported batch {i//batch_size + 1}: {len(batch_data)} records")
            except sqlite3.IntegrityError as e:
                # Handle duplicate URLs
                for job_tuple in batch_data:
                    try:
                        cursor.execute(insert_query, job_tuple)
                        imported_count += 1
                    except sqlite3.IntegrityError:
                        skipped_count += 1
        
        # Commit all changes
        conn.commit()
        
        # Get final count
        cursor.execute("SELECT COUNT(*) FROM jobs")
        total_records = cursor.fetchone()[0]
        
        print(f"\nImport completed!")
        print(f"Total records in database: {total_records}")
        print(f"New records imported: {imported_count}")
        print(f"Duplicate records skipped: {skipped_count}")
        
        # Show some sample data
        print("\nSample of imported data:")
        cursor.execute("SELECT title, company_name, location FROM jobs LIMIT 5")
        samples = cursor.fetchall()
        for sample in samples:
            print(f"- {sample[0]} at {sample[1]} ({sample[2]})")
        
        # Show database statistics
        print(f"\nDatabase statistics:")
        cursor.execute("SELECT COUNT(DISTINCT company_name) FROM jobs WHERE company_name IS NOT NULL")
        unique_companies = cursor.fetchone()[0]
        print(f"Unique companies: {unique_companies}")
        
        cursor.execute("SELECT COUNT(DISTINCT location) FROM jobs WHERE location IS NOT NULL")
        unique_locations = cursor.fetchone()[0]
        print(f"Unique locations: {unique_locations}")
        
        cursor.execute("SELECT job_state, COUNT(*) FROM jobs GROUP BY job_state")
        job_states = cursor.fetchall()
        print("Job states distribution:")
        for state, count in job_states:
            print(f"  {state}: {count}")
        
    except json.JSONDecodeError as e:
        print(f"Error reading JSON file: {e}")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
            print(f"\nDatabase connection closed. Database saved as '{db_name}'")

def show_table_schema():
    """Show the database schema"""
    try:
        conn = sqlite3.connect('jobs_database.db')
        cursor = conn.cursor()
        
        print("\nDatabase Schema:")
        cursor.execute("PRAGMA table_info(jobs)")
        columns = cursor.fetchall()
        
        for column in columns:
            print(f"  {column[1]} ({column[2]}) - {'NOT NULL' if column[3] else 'NULLABLE'}")
        
        conn.close()
    except sqlite3.Error as e:
        print(f"Error reading schema: {e}")

if __name__ == "__main__":
    create_database_and_import()
    show_table_schema()