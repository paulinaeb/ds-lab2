"""
Jobs Database Project - Main Entry Point

This is the main entry point for the Jobs Database Project.
Choose from different modes of operation:

1. Web Application - Flask-based web interface
2. Database Queries - Interactive command-line queries
3. Data Processing - Clean and import job data

"""

import os
import sys
import subprocess

def print_banner():
    """Print application banner"""
    print("=" * 60)
    print("         💼 JOBS DATABASE PROJECT")
    print("         Data Pipeline & Web Application")
    print("=" * 60)
    print()

def check_database():
    """Check if database exists"""
    return os.path.exists('jobs_database.db')

def check_data():
    """Check if data files exist"""
    return os.path.exists('data.json')

def main_menu():
    """Display main menu and handle user choice"""
    while True:
        print_banner()
        
        # Check system status
        has_db = check_database()
        has_data = check_data()
        
        print("📊 SYSTEM STATUS:")
        print(f"   Database: {'✅ Ready' if has_db else '❌ Missing'}")
        print(f"   Data Files: {'✅ Available' if has_data else '❌ Missing'}")
        print()
        
        print("🚀 AVAILABLE OPTIONS:")
        print()
        
        if has_db:
            print("1. 🌐 Start Web Application (Flask)")
            print("2. 🔍 Interactive Database Queries")
            print("3. 📊 Advanced Analytics (Pandas)")
        
        if has_data and not has_db:
            print("4. 🗄️ Setup Database (Clean & Import Data)")
        elif has_data:
            print("4. 🔄 Rebuild Database (Clean & Import)")
        
        print("6. 📋 View Project Structure")
        print("7. ❓ Help & Documentation")
        print("0. 🚪 Exit")
        print()
        
        choice = input("Select an option (0-7): ").strip()
        
        if choice == '1' and has_db:
            start_web_app()
        elif choice == '2' and has_db:
            run_queries()
        elif choice == '3' and has_db:
            run_analytics()
        elif choice == '4' and has_data:
            setup_database()
        elif choice == '6':
            show_project_structure()
        elif choice == '7':
            show_help()
        elif choice == '0':
            print("\n👋 Thank you for using Jobs Database Project!")
            sys.exit(0)
        else:
            print("\n❌ Invalid option or missing requirements. Please try again.")
            input("Press Enter to continue...")

def start_web_app():
    """Start the Flask web application"""
    print("\n🌐 Starting Flask Web Application...")
    print("📍 Server will be available at: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    print()
    
    try:
        subprocess.run([sys.executable, 'web_app.py'])
    except KeyboardInterrupt:
        print("\n🛑 Web server stopped.")
    except FileNotFoundError:
        print("❌ web_app.py not found!")
    
    input("\nPress Enter to return to main menu...")

def run_queries():
    """Run interactive database queries"""
    print("\n🔍 Starting Interactive Query Tool...")
    print()
    
    try:
        subprocess.run([sys.executable, 'query_database.py'])
    except FileNotFoundError:
        print("❌ query_database.py not found!")
    
    input("\nPress Enter to return to main menu...")

def run_analytics():
    """Run advanced analytics"""
    print("\n📊 Advanced Analytics Options:")
    print("1. Company Analysis")
    print("2. Location Trends")
    print("3. Job Market Overview")
    print("4. Export Reports")
    
    # This could be extended with specific analytics functions
    input("\n⚠️  Advanced analytics will be implemented in future updates.\nPress Enter to continue...")

def setup_database():
    """Setup or rebuild the database"""
    print("\n🗄️ Database Setup Process:")
    print("1. Cleaning JSON data...")
    print("2. Creating database structure...")
    print("3. Importing job records...")
    print()
    
    try:
        # Step 1: Clean data
        print("📋 Step 1: Cleaning data...")
        subprocess.run([sys.executable, 'utils/clean_json.py'], check=True)
        
        # Step 2: Create database
        print("🏗️  Step 2: Creating database...")
        subprocess.run([sys.executable, 'utils/create_database.py'], check=True)
        
        print("\n✅ Database setup completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during database setup: {e}")
    except FileNotFoundError:
        print("❌ Required setup files not found in utils/ directory!")
    
    input("\nPress Enter to return to main menu...")
    

def show_project_structure():
    """Display project structure"""
    print("\n📁 PROJECT STRUCTURE")
    print("=" * 50)
    
    structure = """
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
│   ├── jobs_database.db        # SQLite database (93K+ jobs)
│   ├── data.json              # Original raw job data
│   ├── data_cleaned.json      # Processed job data
│   └── utils/
│       ├── clean_json.py      # Data cleaning utilities
│       └── create_database.py # Database setup and import
│
├── 🔍 Query Tools
│   ├── query_database.py      # Advanced database queries
│   └── main.py               # Main application entry point
│
└── ⚙️ Configuration
    ├── requirements.txt      # Python dependencies
    └── venv/                # Virtual environment
"""
    
    print(structure)
    input("Press Enter to return to main menu...")

def show_help():
    """Show help and documentation"""
    print("\n❓ HELP & DOCUMENTATION")
    print("=" * 40)
    print()
    print("📚 Quick Guide:")
    print("1. First-time setup: Choose option 4 to create database")
    print("2. Daily use: Choose option 1 to start web interface")
    print("3. Data analysis: Use option 2 for command-line queries")
    print()
    print("🌐 Web Interface Features:")
    print("   • Dashboard with job market overview")
    print("   • Real-time job search functionality")
    print("   • Company and location browsing")
    print("   • Direct links to original job postings")
    print()
    print("🗄️ Database Info:")
    print("   • 93,078+ job records")
    print("   • 19,547 unique companies") 
    print("   • 5,866 unique locations")
    print("   • European job market focus")
    print()
    print("📋 API Endpoints:")
    print("   • GET /api/search?q={query} - Search jobs")
    print("   • GET /api/company/{name} - Company jobs")
    print()
    print("📂 Files:")
    print("   • README.md - Complete project documentation")
    print("   • requirements.txt - Python dependencies")
    
    input("\nPress Enter to return to main menu...")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)