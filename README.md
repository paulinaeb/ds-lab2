# 💼 Jobs Database Project

A comprehensive data pipeline and web application for analyzing job market data with real-time streaming capabilities.

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
├── 🌊 Stream Mining (Future Extension)
│   ├── producer.py           # Kafka data producer
│   ├── consumer.py           # Kafka data consumer  
│   └── values.yaml          # Kafka configuration
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

## 🌊 Stream Mining Module

The `sm/` directory contains infrastructure for **future extensions**:

- **Real-time Data Ingestion:** Kafka-based streaming pipeline
- **Live Job Processing:** Stream processing for incoming job data
- **Scalable Architecture:** Designed for high-volume data streams
- **Event-driven Updates:** Real-time database updates from job feeds

**Current Status:** Foundation implemented, will be extended for:
- Live job feed integration
- Real-time analytics dashboard updates
- Stream processing with Apache Kafka
- Event-driven data pipeline orchestration

## 🛠️ Technical Stack

- **Backend:** Python Flask
- **Database:** SQLite with optimized indexes
- **Frontend:** HTML5, CSS3, JavaScript (Vanilla)
- **Data Processing:** Pandas, JSON
- **Streaming (Future):** Apache Kafka
- **Deployment:** Local development server

## 📈 Database Statistics

- **Total Jobs:** 93,078
- **Unique Companies:** 19,547
- **Unique Locations:** 5,866
- **Top Location:** Dublin, Ireland (2,285 jobs)
- **Top Company:** TieTalent (1,246 jobs)
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
- Kafka-python 2.2.15 (for streaming module)

## 🔄 Future Enhancements

1. **Stream Mining Extensions:**
   - Real-time job feed integration
   - Live dashboard updates
   - Trend analysis and alerts

2. **Analytics Improvements:**
   - Interactive charts and visualizations
   - Job market trend analysis
   - Salary insights integration

3. **Search Enhancements:**
   - Advanced filtering options
   - Saved searches and alerts
   - Machine learning-based recommendations

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

*This project demonstrates modern data pipeline architecture with web interface capabilities and streaming infrastructure foundation.*