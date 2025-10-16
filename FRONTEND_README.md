## 🌐 Flask Web Application 

**Files:**
- `web_app.py` - Main Flask application
- `templates/` - HTML templates with custom CSS

**Features:**
- Professional-looking web interface
- Real-time job search
- Company and location browsing
- Responsive design
- REST API endpoints

**How to run:**
```bash
py web_app.py
```
Then open: http://localhost:5000

## 🔗 Database Schema

The SQLite database contains these main fields:
- `title` - Job title
- `company_name` - Company name  
- `location` - Job location
- `description` - Job description
- `detail_url` - Link to original job posting
- `skill` - Required skills
- `created_at` - Job creation date
- `scraped_at` - Data collection date