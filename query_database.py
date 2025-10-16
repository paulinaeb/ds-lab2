import sqlite3
import pandas as pd

def query_database():
    """
    Utility functions to query the jobs database
    """
    
    try:
        conn = sqlite3.connect('jobs_database.db')
        
        print("Jobs Database Query Tool")
        print("=" * 50)
        
        # Basic statistics
        print("\n1. Database Overview:")
        df_count = pd.read_sql_query("SELECT COUNT(*) as total_jobs FROM jobs", conn)
        print(f"Total jobs: {df_count['total_jobs'].iloc[0]}")
        
        # Top companies
        print("\n2. Top 10 Companies by Job Count:")
        df_companies = pd.read_sql_query("""
            SELECT company_name, COUNT(*) as job_count 
            FROM jobs 
            WHERE company_name IS NOT NULL 
            GROUP BY company_name 
            ORDER BY job_count DESC 
            LIMIT 10
        """, conn)
        print(df_companies.to_string(index=False))
        
        # Top locations
        print("\n3. Top 10 Locations:")
        df_locations = pd.read_sql_query("""
            SELECT location, COUNT(*) as job_count 
            FROM jobs 
            WHERE location IS NOT NULL 
            GROUP BY location 
            ORDER BY job_count DESC 
            LIMIT 10
        """, conn)
        print(df_locations.to_string(index=False))
        
        # Job states
        print("\n4. Job States Distribution:")
        df_states = pd.read_sql_query("""
            SELECT job_state, COUNT(*) as count 
            FROM jobs 
            GROUP BY job_state 
            ORDER BY count DESC
        """, conn)
        print(df_states.to_string(index=False))
        
        # Recent jobs
        print("\n5. Sample of Recent Jobs:")
        df_recent = pd.read_sql_query("""
            SELECT title, company_name, location, created_at 
            FROM jobs 
            WHERE created_at IS NOT NULL 
            ORDER BY created_at DESC 
            LIMIT 5
        """, conn)
        print(df_recent.to_string(index=False))
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")

def export_to_csv():
    """Export database to CSV files"""
    try:
        conn = sqlite3.connect('jobs_database.db')
        
        # Export full data
        print("Exporting full database to CSV...")
        df_all = pd.read_sql_query("SELECT * FROM jobs", conn)
        df_all.to_csv('jobs_export.csv', index=False)
        print(f"Exported {len(df_all)} records to 'jobs_export.csv'")
        
        # Export summary by company
        print("Exporting company summary...")
        df_company_summary = pd.read_sql_query("""
            SELECT 
                company_name,
                COUNT(*) as total_jobs,
                COUNT(DISTINCT location) as locations_count,
                MIN(created_at) as first_job_date,
                MAX(created_at) as latest_job_date
            FROM jobs 
            WHERE company_name IS NOT NULL 
            GROUP BY company_name 
            ORDER BY total_jobs DESC
        """, conn)
        df_company_summary.to_csv('company_summary.csv', index=False)
        print(f"Exported company summary to 'company_summary.csv'")
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")

def search_jobs(keyword, limit=10):
    """Search for jobs containing a keyword"""
    try:
        conn = sqlite3.connect('jobs_database.db')
        
        query = """
            SELECT title, company_name, location, description
            FROM jobs 
            WHERE title LIKE ? OR description LIKE ? OR company_name LIKE ?
            LIMIT ?
        """
        
        search_term = f"%{keyword}%"
        df_results = pd.read_sql_query(query, conn, params=[search_term, search_term, search_term, limit])
        
        print(f"\nSearch results for '{keyword}':")
        print("=" * 50)
        
        for idx, row in df_results.iterrows():
            print(f"\n{idx+1}. {row['title']}")
            print(f"   Company: {row['company_name']}")
            print(f"   Location: {row['location']}")
            print(f"   Description: {row['description'][:200]}...")
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Check if database exists
    import os
    if not os.path.exists('jobs_database.db'):
        print("Database not found. Please run create_database.py first.")
    else:
        query_database()
        
        # Ask user if they want to export
        response = input("\nWould you like to export data to CSV? (y/n): ")
        if response.lower() == 'y':
            export_to_csv()
        
        # Ask for search
        search_term = input("\nEnter a keyword to search jobs (or press Enter to skip): ")
        if search_term.strip():
            search_jobs(search_term)