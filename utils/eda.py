"""
Exploratory Data Analysis for Jobs Dataset

This script performs comprehensive EDA on the cleaned jobs data.
It generates visualizations and statistical summaries.

Usage:
    python utils/eda.py --file data_cleaned.json
    python utils/eda.py --file data_cleaned.json --save-only
"""

import os
import sys
import argparse
import json
from collections import Counter
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Common stopwords for text analysis
STOPWORDS = {
    'and', 'or', 'the', 'for', 'to', 'of', 'in', 'with', 'a', 'on', 'at', 'is',
    'are', 'be', 'as', 'an', 'will', 'has', 'have', 'it', 'by', 'from', 'this',
    'that', 'our', 'your', 'their', 'we', 'you', 'they', 'experience', 'work',
    'working', 'team', 'including', 'us', 'job', 'role', 'strong', 'ability'
}


class JobsEDA:
    """Exploratory Data Analysis for Jobs Dataset"""
    
    def __init__(self, data_path, output_dir="outputs/eda_plots", save_only=True):
        self.data_path = data_path
        self.output_dir = output_dir
        self.save_only = save_only
        self.df = None
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"  EXPLORATORY DATA ANALYSIS - JOBS DATASET")
        print(f"{'='*60}\n")
        print(f"Output directory: {os.path.abspath(self.output_dir)}")
        print(f"Mode: Saving plots to files\n")
    
    def load_data(self):
        """Load and prepare the dataset"""
        print("Loading data...")
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                self.df = pd.DataFrame(data)
                print(f"✓ Loaded {len(self.df)} job records")
            else:
                print(f"✗ Unexpected data format: {type(data)}")
                return False
            
            print(f"✓ Columns found: {list(self.df.columns)}\n")
            return True
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def basic_info(self):
        """Display basic dataset information"""
        print("\n" + "="*60)
        print("1. BASIC DATASET INFORMATION")
        print("="*60)
        
        print(f"\nDataset shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        
        print("\nColumn names and types:")
        for col in self.df.columns:
            print(f"  - {col}: {self.df[col].dtype}")
        
        print(f"\nMemory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\nFirst 3 records (sample):")
        print(self.df.head(3).to_string())
        
        print("\nMissing values:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Missing': missing,
            'Percentage': missing_pct
        })
        missing_info = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
        if len(missing_info) > 0:
            print(missing_info.to_string())
        else:
            print("No missing values found!")
    
    def analyze_companies(self):
        """Analyze company distribution"""
        print("\n" + "="*60)
        print("2. COMPANY ANALYSIS")
        print("="*60)
        
        try:
            companies = self.df['Company Name'].value_counts().head(20)
            print(f"\nTop 20 companies (out of {self.df['Company Name'].nunique():,} unique):")
            print(companies.to_string())
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 8))
            companies.plot(kind='barh', ax=ax, color='steelblue')
            ax.set_xlabel('Number of Job Postings')
            ax.set_ylabel('Company')
            ax.set_title(f'Top 20 Companies by Number of Job Postings\n(Total: {len(self.df):,} jobs)')
            ax.invert_yaxis()
            plt.tight_layout()
            self._save_plot('top_companies.png')
            print("✓ Saved: top_companies.png")
        except Exception as e:
            print(f"✗ Error in company analysis: {e}")
    
    def analyze_locations(self):
        """Analyze location distribution"""
        print("\n" + "="*60)
        print("3. LOCATION ANALYSIS")
        print("="*60)
        
        try:
            locations = self.df['Location'].value_counts().head(20)
            print(f"\nTop 20 locations (out of {self.df['Location'].nunique():,} unique):")
            print(locations.to_string())
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 8))
            locations.plot(kind='barh', ax=ax, color='coral')
            ax.set_xlabel('Number of Job Postings')
            ax.set_ylabel('Location')
            ax.set_title(f'Top 20 Locations by Number of Job Postings\n(Total: {len(self.df):,} jobs)')
            ax.invert_yaxis()
            plt.tight_layout()
            self._save_plot('top_locations.png')
            print("✓ Saved: top_locations.png")
        except Exception as e:
            print(f"✗ Error in location analysis: {e}")
    
    def analyze_job_titles(self):
        """Analyze job titles and extract keywords"""
        print("\n" + "="*60)
        print("4. JOB TITLE ANALYSIS")
        print("="*60)
        
        try:
            # Top titles
            titles = self.df['Title'].value_counts().head(20)
            print(f"\nTop 20 job titles (out of {self.df['Title'].nunique():,} unique):")
            print(titles.to_string())
            
            # Extract keywords from titles
            print("\nExtracting keywords from job titles...")
            all_words = []
            for title in self.df['Title'].dropna():
                words = str(title).lower().split()
                for word in words:
                    # Clean word - remove special characters
                    word = ''.join(c for c in word if c.isalpha())
                    if len(word) > 2 and word not in STOPWORDS:
                        all_words.append(word)
            
            word_freq = Counter(all_words).most_common(30)
            print("\nTop 30 keywords in job titles:")
            for word, count in word_freq:
                print(f"  {word}: {count:,}")
            
            # Plot keywords
            if len(word_freq) > 0:
                words, counts = zip(*word_freq[:20])
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.barh(range(len(words)), counts, color='teal')
                ax.set_yticks(range(len(words)))
                ax.set_yticklabels(words)
                ax.set_xlabel('Frequency')
                ax.set_title('Top 20 Keywords in Job Titles')
                ax.invert_yaxis()
                plt.tight_layout()
                self._save_plot('title_keywords.png')
                print("✓ Saved: title_keywords.png")
        except Exception as e:
            print(f"✗ Error in job title analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def analyze_skills(self):
        """Analyze skills mentioned in job postings"""
        print("\n" + "="*60)
        print("5. SKILLS ANALYSIS")
        print("="*60)
        
        try:
            print("Extracting skills from job postings...")
            all_skills = []
            skills_count = 0
            
            for skill_text in self.df['Skill'].dropna():
                # Skills are in format "Skills: Analytical Skills, Business, +8 more"
                if 'Skills:' in str(skill_text):
                    skills_count += 1
                    skills_part = str(skill_text).split('Skills:')[1].strip()
                    # Remove "+X more" pattern
                    skills_part = skills_part.split('+')[0].strip()
                    # Split by comma
                    skills = [s.strip() for s in skills_part.split(',') if s.strip()]
                    all_skills.extend(skills)
            
            print(f"Found skills in {skills_count:,} job postings ({skills_count/len(self.df)*100:.1f}%)")
            
            if len(all_skills) > 0:
                skill_freq = Counter(all_skills).most_common(25)
                print(f"\nTop 25 skills (out of {len(set(all_skills)):,} unique):")
                for skill, count in skill_freq:
                    print(f"  {skill}: {count:,}")
                
                # Plot
                skills, counts = zip(*skill_freq[:20])
                fig, ax = plt.subplots(figsize=(12, 10))
                ax.barh(range(len(skills)), counts, color='purple')
                ax.set_yticks(range(len(skills)))
                ax.set_yticklabels(skills)
                ax.set_xlabel('Frequency')
                ax.set_title('Top 20 Skills Mentioned in Job Postings')
                ax.invert_yaxis()
                plt.tight_layout()
                self._save_plot('top_skills.png')
                print("✓ Saved: top_skills.png")
            else:
                print("No skills data found")
        except Exception as e:
            print(f"✗ Error in skills analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def analyze_dates(self):
        """Analyze posting dates"""
        print("\n" + "="*60)
        print("6. TEMPORAL ANALYSIS")
        print("="*60)
        
        try:
            dates = pd.to_datetime(self.df['Created At'], errors='coerce')
            valid_dates = dates.dropna()
            
            if len(valid_dates) == 0:
                print("✗ No valid dates found")
                return
            
            print(f"\nDate range: {valid_dates.min()} to {valid_dates.max()}")
            print(f"Valid dates: {len(valid_dates):,} ({len(valid_dates)/len(self.df)*100:.1f}%)")
            
            # Group by date
            date_counts = valid_dates.dt.date.value_counts().sort_index()
            print(f"\nNumber of unique dates: {len(date_counts)}")
            
            # Plot timeline
            fig, ax = plt.subplots(figsize=(14, 6))
            date_counts.plot(ax=ax, color='purple', marker='o', linewidth=2, markersize=4)
            ax.set_xlabel('Date')
            ax.set_ylabel('Number of Postings')
            ax.set_title('Job Postings Over Time')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            self._save_plot('postings_timeline.png')
            print("✓ Saved: postings_timeline.png")
            
            # Show day of week distribution
            weekday_counts = valid_dates.dt.day_name().value_counts()
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekday_counts = weekday_counts.reindex(weekday_order, fill_value=0)
            
            print("\nPostings by day of week:")
            print(weekday_counts.to_string())
            
            # Plot weekday distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            weekday_counts.plot(kind='bar', ax=ax, color='navy')
            ax.set_xlabel('Day of Week')
            ax.set_ylabel('Number of Postings')
            ax.set_title('Job Postings by Day of Week')
            plt.xticks(rotation=45)
            plt.tight_layout()
            self._save_plot('postings_by_weekday.png')
            print("✓ Saved: postings_by_weekday.png")
            
        except Exception as e:
            print(f"✗ Error in temporal analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def analyze_descriptions(self):
        """Analyze job descriptions"""
        print("\n" + "="*60)
        print("7. DESCRIPTION ANALYSIS")
        print("="*60)
        
        try:
            descriptions = self.df['Description'].dropna().astype(str)
            
            # Length statistics
            lengths = descriptions.str.len()
            print(f"\nDescription length statistics (characters):")
            print(lengths.describe().to_string())
            
            # Plot distribution
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.hist(lengths, bins=50, color='green', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Description Length (characters)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Job Description Lengths')
            ax.axvline(lengths.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {lengths.median():.0f}')
            ax.axvline(lengths.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {lengths.mean():.0f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            self._save_plot('description_lengths.png')
            print("✓ Saved: description_lengths.png")
            
            # Word count
            word_counts = descriptions.str.split().str.len()
            print(f"\nWord count statistics:")
            print(word_counts.describe().to_string())
            
        except Exception as e:
            print(f"✗ Error in description analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def analyze_job_state(self):
        """Analyze job state distribution"""
        print("\n" + "="*60)
        print("8. JOB STATE ANALYSIS")
        print("="*60)
        
        try:
            states = self.df['Job State'].value_counts()
            print(f"\nJob states distribution:")
            print(states.to_string())
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            states.plot(kind='bar', ax=ax, color='orange')
            ax.set_xlabel('Job State')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Job States')
            plt.xticks(rotation=45)
            plt.tight_layout()
            self._save_plot('job_states.png')
            print("✓ Saved: job_states.png")
            
        except Exception as e:
            print(f"✗ Error in job state analysis: {e}")
    
    def generate_summary_report(self):
        """Generate a text summary report"""
        print("\n" + "="*60)
        print("9. GENERATING SUMMARY REPORT")
        print("="*60)
        
        try:
            report_path = os.path.join(self.output_dir, 'eda_summary.txt')
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("EXPLORATORY DATA ANALYSIS SUMMARY\n")
                f.write("="*60 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Dataset: {self.data_path}\n\n")
                
                f.write(f"Total Records: {len(self.df):,}\n")
                f.write(f"Total Columns: {len(self.df.columns)}\n\n")
                
                f.write("Columns:\n")
                for col in self.df.columns:
                    f.write(f"  - {col} ({self.df[col].dtype})\n")
                
                f.write(f"\nMissing Values:\n")
                missing = self.df.isnull().sum()
                if missing.sum() > 0:
                    for col in missing[missing > 0].index:
                        f.write(f"  - {col}: {missing[col]:,} ({missing[col]/len(self.df)*100:.1f}%)\n")
                else:
                    f.write("  No missing values\n")
                
                # Add unique value counts
                f.write(f"\nUnique Values per Column:\n")
                for col in self.df.columns:
                    f.write(f"  - {col}: {self.df[col].nunique():,}\n")
            
            print(f"✓ Summary report saved to: {report_path}")
        except Exception as e:
            print(f"✗ Error generating summary report: {e}")
    
    def run_full_analysis(self):
        """Run complete EDA pipeline"""
        if not self.load_data():
            return False
        
        try:
            self.basic_info()
            self.analyze_companies()
            self.analyze_locations()
            self.analyze_job_titles()
            self.analyze_skills()
            self.analyze_dates()
            self.analyze_descriptions()
            self.analyze_job_state()
            self.generate_summary_report()
            
            print("\n" + "="*60)
            print("✓ EDA COMPLETED SUCCESSFULLY")
            print("="*60)
            print(f"\nAll outputs saved to: {os.path.abspath(self.output_dir)}")
            print(f"Total plots generated: Check the output directory")
            return True
        except Exception as e:
            print(f"\n✗ Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_plot(self, filename):
        """Save plot to file"""
        filepath = os.path.join(self.output_dir, filename)
        try:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"✗ Error saving plot {filename}: {e}")
            plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Exploratory Data Analysis for Jobs Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--file', '-f',
        default='data_cleaned.json',
        help='Path to JSON data file (default: data_cleaned.json)'
    )
    parser.add_argument(
        '--outdir', '-o',
        default='outputs/eda_plots',
        help='Output directory for plots (default: outputs/eda_plots)'
    )
    
    args = parser.parse_args()
    
    # Run EDA
    eda = JobsEDA(args.file, args.outdir, save_only=True)
    success = eda.run_full_analysis()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()