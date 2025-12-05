
# 💼 Cybersecurity Jobs Analysis

A comprehensive data science project implementing baseline NLP methods for analyzing and comparing cybersecurity job descriptions from the European market.

## 🎯 Project Objectives

This project implements and compares three baseline NLP methods for analyzing cybersecurity job descriptions:

1. **TF-IDF + Cosine Similarity** - Sparse bag-of-words representation
2. **LDA Topic Modeling** - Latent semantic topic discovery
3. **Word2Vec Embeddings** - Dense semantic word representations

**Research Goals:**
- Extract meaningful patterns from 155K+ job descriptions
- Compare baseline methods using standardized evaluation metrics
- Establish performance benchmarks for future transformer-based models
- Enable job similarity search and recommendation systems

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Virtual environment (recommended)
- ~2GB disk space for database and outputs

### Installation

```powershell
# Clone repository
git clone https://github.com/paulinaeb/ds-lab2
cd ds-lab2

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Data Pipeline

### 1. Database Setup

```powershell
# Create SQLite database from JSON data
python utils/create_database.py
```

**Database Schema:**
- **Table:** `jobs` (155,350 records)
- **Key Columns:** id, title, company_name, location, description, skill, created_at
- **Processed Columns:** cleaned_description, cleaned_skills, country

### 2. Exploratory Data Analysis

```powershell
# Run comprehensive EDA
python utils/eda.py

# EDA includes:
# - Company distribution analysis
# - Location/country mapping (41 metro areas)
# - Job title frequency analysis
# - Skill extraction and ranking
# - Temporal patterns
# - Description length statistics
# - Job state distribution
```

### 3. Data Preprocessing

```powershell
# Clean skill field (remove LinkedIn artifacts)
python preprocessing/clean_skills.py

# Clean job descriptions (remove boilerplate)
python preprocessing/clean_description.py

# Standardize locations to country level
python preprocessing/extract_country.py
```

**Preprocessing Details:**
- **Skills:** Removes "Skills:", "+X more", "X of Y skills match..."
- **Descriptions:** Removes 19 common boilerplate phrases
- **Locations:** Maps 41 metropolitan areas to 15 countries
- **Non-destructive:** Creates new columns (cleaned_skills, cleaned_description, country)

## 🤖 Baseline Methods

### Method 1: TF-IDF + Cosine Similarity

**Purpose:** Sparse bag-of-words representation for document similarity

```powershell
# Run TF-IDF baseline (full dataset)
python baseline/tfidf_baseline.py

# Run on sample for faster execution
python baseline/tfidf_baseline.py --sample 10000
```

### Method 2: LDA Topic Modeling

**Purpose:** Discover latent semantic topics in job descriptions

```powershell
# Run LDA baseline (default: 10 topics)
python baseline/lda_baseline.py --sample 10000

# Find optimal number of topics
python baseline/lda_baseline.py --find-optimal-k --sample 10000

# Custom topic count
python baseline/lda_baseline.py --n-topics 15 --sample 10000
```


### Method 3: Word2Vec Embeddings

**Purpose:** Dense semantic word representations for similarity and clustering

```powershell
# Run Word2Vec baseline
python baseline/word2vec_baseline.py --sample 10000

# Custom embedding size
python baseline/word2vec_baseline.py --vector-size 200 --sample 10000
```

### Reproducibility

All methods use **fixed random seeds** (seed=42) for reproducible results:
- TF-IDF: Deterministic (no randomness)
- LDA: `random_state=42` in LatentDirichletAllocation
- Word2Vec: `seed=42` in Word2Vec model

## 🌐 Web Application

```powershell
# Start Flask server
python web_app.py
```

**Access at:** http://localhost:5000

**Features:**
- Dashboard with database statistics
- Full-text search across 155K job descriptions
- Company and location filtering
- Direct links to original LinkedIn postings


## 🗄️ Database Schema

The SQLite database (jobs_database.db) contains:

| Field | Type | Description |
|-------|------|-------------|
| `id` | INTEGER | Primary key (auto-increment) |
| `title` | TEXT | Job title |
| `company_name` | TEXT | Company name |
| `location` | TEXT | Job location |
| `country` | TEXT | Standardized country name |
| `description` | TEXT | Full job description |
| `cleaned_description` | TEXT | Preprocessed description |
| `primary_description` | TEXT | Short job summary |
| `detail_url` | TEXT | LinkedIn job URL (unique) |
| `skill` | TEXT | Required skills (raw) |
| `cleaned_skills` | TEXT | Preprocessed skills |
| `job_state` | TEXT | Job status (LISTED/etc.) |
| `poster_id` | TEXT | Job poster ID |
| `company_logo` | TEXT | Company logo URL |
| `created_at` | TEXT | Job creation timestamp |
| `scraped_at` | TEXT | Data collection timestamp |
| `imported_at` | DATETIME | Database import timestamp |

## 📝 Citation & Methodology

For detailed methodology & references, see project documentation.

## 🛠️ Technical Stack

- **Language:** Python 3.11
- **NLP:** scikit-learn 1.3.0, gensim 4.3.0
- **Data Processing:** pandas 2.0.3, numpy 1.24.3
- **Visualization:** matplotlib 3.7.1, seaborn 0.12.2, wordcloud 1.9.0
- **Database:** SQLite 3.42+
- **Web Framework:** Flask 3.1.2

## 🔜 Future Work

- Implement BERT/transformer-based models
- Add cross-validation for evaluation
- Implement recommendation system
- Add model comparison visualizations
- Export results to LaTeX tables for publication
