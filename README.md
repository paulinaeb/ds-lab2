# 💼 Jobs Database Project

A comprehensive data pipeline and web application for analyzing job market data.

## 📁 Project Structure

```
ds-lab-2/
├── 📊 Web Application
│   ├── web_app.py              # Flask web server
│   └── templates/              # HTML templates
│       ├── index.html          # Dashboard homepage
│       ├── search.html         # Job search interface
│       ├── companies.html      # Companies overview
│       └── locations.html      # Locations overview
│
├── 🗄️ Database & Data Processing
│   ├── jobs_database.db        # SQLite database 
│   ├── data.json              # Original raw job data
│   ├── data_cleaned.json      # Processed job data
│   └── utils/
│       ├── clean_json.py      # Data cleaning utilities
│       └── create_database.py # Database setup and import
│
├── 🔍 Query Tools
│   └── main.py               # Main application entry point
│
├── ⚙️ Configuration
│   ├── requirements.txt      # Python dependencies
│   └── venv/                # Virtual environment
```

## 🚀 Quick Start

### **Web Application**

Start the Flask web server:

```bash
# Activate virtual environment (if not already active)
venv\Scripts\activate

# Run the web application
python web_app.py
```

🌐 **Access at:** http://localhost:5000


## ✨ Features

### 🌐 **Web Interface**
- **Dashboard:** Overview statistics and top companies/locations
- **Search:** Real-time job search with keyword filtering
- **Companies:** Browse jobs by company with statistics
- **Locations:** Explore job opportunities by geographic location
- **Responsive Design:** Mobile-friendly interface
- **Direct Links:** Access to original LinkedIn job postings

### 📊 **Data Analytics**
- **93,078+ Job Records** from 19,547 companies across 5,866 locations
- **SQLite Database** with optimized indexes for fast queries
- **RESTful API** endpoints for programmatic access
- **Export Capabilities** for CSV data export

### 🔍 **Search & Filtering**
- Full-text search across job titles, descriptions, and companies
- Location-based filtering
- Company-specific job listings
- Relevance-based result ranking

## 🗄️ Database Schema

The SQLite database (`jobs_database.db`) contains:

| Field | Type | Description |
|-------|------|-------------|
| `id` | INTEGER | Primary key (auto-increment) |
| `title` | TEXT | Job title |
| `company_name` | TEXT | Company name |
| `location` | TEXT | Job location |
| `description` | TEXT | Full job description |
| `primary_description` | TEXT | Short job summary |
| `detail_url` | TEXT | LinkedIn job URL (unique) |
| `skill` | TEXT | Required skills |
| `job_state` | TEXT | Job status (LISTED/etc.) |
| `poster_id` | TEXT | Job poster ID |
| `company_logo` | TEXT | Company logo URL |
| `created_at` | TEXT | Job creation timestamp |
| `scraped_at` | TEXT | Data collection timestamp |
| `imported_at` | DATETIME | Database import timestamp |


## 🛠️ Technical Stack

- **Backend:** Python Flask
- **Database:** SQLite with optimized indexes
- **Frontend:** HTML5, CSS3, JavaScript (Vanilla)
- **Data Processing:** Pandas, JSON
- **Deployment:** Local development server

## 📈 Database Statistics

- **Total Jobs:** 93,078
- **Unique Companies:** 19,547
- **Unique Locations:** 5,866
- **Geographic Coverage:** European job market focus

## 🔧 API Endpoints

The Flask application provides REST API access:

- `GET /` - Dashboard homepage
- `GET /search` - Job search interface
- `GET /api/search?q={query}&limit={n}` - JSON search API
- `GET /companies` - Companies overview
- `GET /locations` - Locations overview
- `GET /api/company/{name}` - Company-specific jobs

## 📋 Requirements

See `requirements.txt` for complete dependency list. Key packages:
- Flask 3.1.2
- Pandas 2.3.3

## 📞 Usage Examples

**Search for cybersecurity jobs:**
```
http://localhost:5000/api/search?q=cybersecurity&limit=10
```

**Get all TieTalent jobs:**
```
http://localhost:5000/api/company/TieTalent
```

---
