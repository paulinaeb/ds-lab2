from flask import Flask, render_template, request, jsonify
import sqlite3
import json
from datetime import datetime

app = Flask(__name__)

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect('jobs_database.db')
    conn.row_factory = sqlite3.Row  # This enables column access by name
    return conn

@app.route('/')
def index():
    """Main dashboard page"""
    try:
        conn = get_db_connection()
        
        # Get basic statistics
        stats = {}
        
        # Total jobs
        cursor = conn.execute('SELECT COUNT(*) as total FROM jobs')
        stats['total_jobs'] = cursor.fetchone()['total']
        
        # Total companies
        cursor = conn.execute('SELECT COUNT(DISTINCT company_name) as total FROM jobs WHERE company_name IS NOT NULL')
        stats['total_companies'] = cursor.fetchone()['total']
        
        # Total locations
        cursor = conn.execute('SELECT COUNT(DISTINCT location) as total FROM jobs WHERE location IS NOT NULL')
        stats['total_locations'] = cursor.fetchone()['total']
        
        # Top 5 companies
        cursor = conn.execute('''
            SELECT company_name, COUNT(*) as count 
            FROM jobs 
            WHERE company_name IS NOT NULL AND company_name != ''
            GROUP BY company_name 
            ORDER BY count DESC 
            LIMIT 5
        ''')
        stats['top_companies'] = cursor.fetchall()
        
        # Top 5 locations
        cursor = conn.execute('''
            SELECT location, COUNT(*) as count 
            FROM jobs 
            WHERE location IS NOT NULL AND location != ''
            GROUP BY location 
            ORDER BY count DESC 
            LIMIT 5
        ''')
        stats['top_locations'] = cursor.fetchall()
        
        conn.close()
        return render_template('index.html', stats=stats)
        
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/search')
def search():
    """Search page"""
    return render_template('search.html')

@app.route('/api/search')
def api_search():
    """API endpoint for searching jobs"""
    query = request.args.get('q', '').strip()
    limit = request.args.get('limit', 20, type=int)
    
    if not query:
        return jsonify({'error': 'No search query provided'}), 400
    
    try:
        conn = get_db_connection()
        
        cursor = conn.execute('''
            SELECT title, company_name, location, description, detail_url
            FROM jobs 
            WHERE (title LIKE ? OR description LIKE ? OR company_name LIKE ?)
            AND title IS NOT NULL
            ORDER BY 
                CASE WHEN title LIKE ? THEN 1
                     WHEN company_name LIKE ? THEN 2
                     ELSE 3 END
            LIMIT ?
        ''', (f'%{query}%', f'%{query}%', f'%{query}%', f'%{query}%', f'%{query}%', limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'title': row['title'],
                'company': row['company_name'] or 'Unknown Company',
                'location': row['location'] or 'Unknown Location',
                'description': (row['description'][:200] + '...') if row['description'] and len(row['description']) > 200 else (row['description'] or 'No description available'),
                'url': row['detail_url']
            })
        
        conn.close()
        return jsonify({'results': results, 'count': len(results)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/companies')
def companies():
    """Companies page"""
    try:
        conn = get_db_connection()
        
        cursor = conn.execute('''
            SELECT 
                company_name,
                COUNT(*) as job_count,
                COUNT(DISTINCT location) as locations,
                MAX(created_at) as latest_job
            FROM jobs 
            WHERE company_name IS NOT NULL AND company_name != ''
            GROUP BY company_name 
            ORDER BY job_count DESC
            LIMIT 50
        ''')
        
        companies = cursor.fetchall()
        conn.close()
        
        return render_template('companies.html', companies=companies)
        
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/locations')
def locations():
    """Locations page"""
    try:
        conn = get_db_connection()
        
        cursor = conn.execute('''
            SELECT 
                location,
                COUNT(*) as job_count,
                COUNT(DISTINCT company_name) as companies
            FROM jobs 
            WHERE location IS NOT NULL AND location != ''
            GROUP BY location 
            ORDER BY job_count DESC
            LIMIT 50
        ''')
        
        locations = cursor.fetchall()
        conn.close()
        
        return render_template('locations.html', locations=locations)
        
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/api/company/<company_name>')
def api_company_jobs(company_name):
    """Get jobs for a specific company"""
    try:
        conn = get_db_connection()
        
        cursor = conn.execute('''
            SELECT title, location, description, detail_url, created_at
            FROM jobs 
            WHERE company_name = ?
            ORDER BY created_at DESC
            LIMIT 20
        ''', (company_name,))
        
        jobs = []
        for row in cursor.fetchall():
            jobs.append({
                'title': row['title'],
                'location': row['location'] or 'Unknown Location',
                'description': (row['description'][:150] + '...') if row['description'] and len(row['description']) > 150 else (row['description'] or 'No description'),
                'url': row['detail_url'],
                'created_at': row['created_at']
            })
        
        conn.close()
        return jsonify({'jobs': jobs})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Jobs Database Web Interface...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)