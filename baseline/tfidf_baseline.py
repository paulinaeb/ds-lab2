"""
TF-IDF + Cosine Similarity Baseline

This script implements TF-IDF vectorization with cosine similarity
for job description analysis. Includes comprehensive evaluation metrics.

Usage:
    python baseline/tfidf_baseline.py --sample 10000
    python baseline/tfidf_baseline.py --full  # Use all 155K docs (slower)
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import time
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
DB_PATH = 'jobs_database.db'
OUTPUT_DIR = 'outputs/baseline/tfidf'
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class TFIDFBaseline:
    def __init__(self, sample_size=None):
        """
        Initialize TF-IDF baseline
        
        Args:
            sample_size: Number of documents to sample (None = use all)
        """
        self.sample_size = sample_size
        self.vectorizer = None
        self.tfidf_matrix = None
        self.df = None
        self.similarity_matrix = None
        self.metrics = {}
        
    def load_data(self):
        """Load data from SQLite database"""
        print("\n" + "="*60)
        print("LOADING DATA FROM DATABASE")
        print("="*60)
        
        conn = sqlite3.connect(DB_PATH)
        
        # Load required columns
        query = """
            SELECT id, title, company_name, location, country,
                   cleaned_description, cleaned_skills, created_at
            FROM jobs
            WHERE cleaned_description IS NOT NULL 
            AND cleaned_description != ''
        """
        
        self.df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"✓ Loaded {len(self.df):,} jobs with cleaned descriptions")
        
        # Sample if requested
        if self.sample_size and self.sample_size < len(self.df):
            self.df = self.df.sample(n=self.sample_size, random_state=42)
            print(f"✓ Sampled {self.sample_size:,} documents for analysis")
        
        # Basic statistics
        self.df['desc_length'] = self.df['cleaned_description'].str.split().str.len()
        print(f"\nDocument length statistics:")
        print(f"  Mean: {self.df['desc_length'].mean():.1f} words")
        print(f"  Median: {self.df['desc_length'].median():.1f} words")
        print(f"  Min: {self.df['desc_length'].min()} words")
        print(f"  Max: {self.df['desc_length'].max()} words")
        
    def build_tfidf(self):
        """Build TF-IDF vectors"""
        print("\n" + "="*60)
        print("BUILDING TF-IDF VECTORS")
        print("="*60)
        
        start_time = time.time()
        
        # Configure vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=5,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            strip_accents='unicode',
            dtype=np.float32
        )
        
        print("Vectorizer configuration:")
        print(f"  max_features: 5,000")
        print(f"  min_df: 5 (word must appear in ≥5 docs)")
        print(f"  max_df: 0.8 (ignore words in >80% of docs)")
        print(f"  ngram_range: (1, 2) (unigrams + bigrams)")
        print(f"  stop_words: english")
        
        # Transform
        corpus = self.df['cleaned_description'].fillna('').tolist()
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        
        training_time = time.time() - start_time
        
        # Store metrics
        self.metrics['training_time'] = training_time
        self.metrics['vocab_size'] = len(self.vectorizer.vocabulary_)
        self.metrics['n_docs'] = self.tfidf_matrix.shape[0]
        self.metrics['n_features'] = self.tfidf_matrix.shape[1]
        self.metrics['sparsity'] = 1.0 - (self.tfidf_matrix.nnz / 
                                          (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1]))
        
        print(f"\n✓ TF-IDF matrix created:")
        print(f"  Shape: {self.tfidf_matrix.shape} (docs × features)")
        print(f"  Vocabulary size: {self.metrics['vocab_size']:,}")
        print(f"  Sparsity: {self.metrics['sparsity']:.4f}")
        print(f"  Training time: {training_time:.2f}s")
        
        # Memory usage
        memory_mb = self.tfidf_matrix.data.nbytes / (1024**2)
        print(f"  Memory usage: {memory_mb:.2f} MB")
        self.metrics['memory_mb'] = memory_mb
        
    def compute_similarity(self):
        """Compute cosine similarity matrix"""
        print("\n" + "="*60)
        print("COMPUTING COSINE SIMILARITY")
        print("="*60)
        
        start_time = time.time()
        
        # For large datasets, sample for similarity computation
        if len(self.df) > 10000:
            n_sample = 5000
            print(f"⚠ Large dataset detected. Sampling {n_sample} docs for similarity matrix...")
            sample_indices = np.random.choice(len(self.df), n_sample, replace=False)
            sample_matrix = self.tfidf_matrix[sample_indices]
            self.similarity_matrix = cosine_similarity(sample_matrix)
            print(f"✓ Similarity matrix shape: {self.similarity_matrix.shape}")
        else:
            self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
            print(f"✓ Similarity matrix shape: {self.similarity_matrix.shape}")
        
        inference_time = time.time() - start_time
        self.metrics['inference_time'] = inference_time
        
        # Compute statistics (excluding diagonal)
        np.fill_diagonal(self.similarity_matrix, np.nan)
        
        self.metrics['sim_mean'] = np.nanmean(self.similarity_matrix)
        self.metrics['sim_std'] = np.nanstd(self.similarity_matrix)
        self.metrics['sim_min'] = np.nanmin(self.similarity_matrix)
        self.metrics['sim_max'] = np.nanmax(self.similarity_matrix)
        self.metrics['sim_median'] = np.nanmedian(self.similarity_matrix)
        self.metrics['sim_25'] = np.nanpercentile(self.similarity_matrix, 25)
        self.metrics['sim_75'] = np.nanpercentile(self.similarity_matrix, 75)
        self.metrics['sim_90'] = np.nanpercentile(self.similarity_matrix, 90)
        self.metrics['sim_95'] = np.nanpercentile(self.similarity_matrix, 95)
        self.metrics['sim_99'] = np.nanpercentile(self.similarity_matrix, 99)
        
        print(f"\nSimilarity statistics:")
        print(f"  Mean: {self.metrics['sim_mean']:.4f}")
        print(f"  Median: {self.metrics['sim_median']:.4f}")
        print(f"  Std Dev: {self.metrics['sim_std']:.4f}")
        print(f"  Min: {self.metrics['sim_min']:.4f}")
        print(f"  Max: {self.metrics['sim_max']:.4f}")
        print(f"  95th percentile: {self.metrics['sim_95']:.4f}")
        print(f"  Inference time: {inference_time:.2f}s")
        
    def analyze_top_terms(self):
        """Analyze most important terms"""
        print("\n" + "="*60)
        print("TOP TERMS ANALYSIS")
        print("="*60)
        
        # Get feature names
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        # Calculate average TF-IDF score per term
        avg_tfidf = np.asarray(self.tfidf_matrix.mean(axis=0)).flatten()
        
        # Sort by average TF-IDF
        top_indices = avg_tfidf.argsort()[::-1][:50]
        
        print("\nTop 50 terms by average TF-IDF score:")
        print(f"\n{'Rank':<6} {'Term':<30} {'Avg TF-IDF':<12} {'Type':<10}")
        print("-" * 60)
        
        top_terms = []
        for i, idx in enumerate(top_indices, 1):
            term = feature_names[idx]
            score = avg_tfidf[idx]
            term_type = "bigram" if " " in term else "unigram"
            print(f"{i:<6} {term:<30} {score:<12.6f} {term_type:<10}")
            top_terms.append((term, score, term_type))
        
        # Save to file
        with open(f"{OUTPUT_DIR}/top_terms.txt", 'w', encoding='utf-8') as f:
            f.write("TOP 50 TERMS BY AVERAGE TF-IDF SCORE\n")
            f.write("="*60 + "\n\n")
            f.write(f"{'Rank':<6} {'Term':<30} {'Avg TF-IDF':<12} {'Type':<10}\n")
            f.write("-" * 60 + "\n")
            for i, (term, score, term_type) in enumerate(top_terms, 1):
                f.write(f"{i:<6} {term:<30} {score:<12.6f} {term_type:<10}\n")
        
        print(f"\n✓ Saved to {OUTPUT_DIR}/top_terms.txt")
        
        return top_terms
        
    def find_similar_pairs(self):
        """Find most similar document pairs"""
        print("\n" + "="*60)
        print("DOCUMENT SIMILARITY PAIRS")
        print("="*60)
        
        # Get upper triangle indices (avoid duplicates and diagonal)
        triu_indices = np.triu_indices_from(self.similarity_matrix, k=1)
        similarities = self.similarity_matrix[triu_indices]
        
        # Find top 10 most similar
        top_10_idx = similarities.argsort()[-10:][::-1]
        
        print("\n📊 TOP 10 MOST SIMILAR JOB PAIRS:")
        print("="*60)
        
        similar_pairs = []
        for rank, idx in enumerate(top_10_idx, 1):
            i, j = triu_indices[0][idx], triu_indices[1][idx]
            sim_score = similarities[idx]
            
            job1 = self.df.iloc[i]
            job2 = self.df.iloc[j]
            
            print(f"\n{rank}. Similarity: {sim_score:.4f}")
            print(f"   Job A: {job1['title']}")
            print(f"          Company: {job1['company_name']}")
            print(f"          Country: {job1['country']}")
            print(f"   Job B: {job2['title']}")
            print(f"          Company: {job2['company_name']}")
            print(f"          Country: {job2['country']}")
            
            similar_pairs.append({
                'rank': rank,
                'similarity': sim_score,
                'job1_title': job1['title'],
                'job1_company': job1['company_name'],
                'job1_country': job1['country'],
                'job2_title': job2['title'],
                'job2_company': job2['company_name'],
                'job2_country': job2['country']
            })
        
        # Save to file
        with open(f"{OUTPUT_DIR}/similar_pairs.txt", 'w', encoding='utf-8') as f:
            f.write("MOST SIMILAR JOB PAIRS\n")
            f.write("="*60 + "\n\n")
            for pair in similar_pairs:
                f.write(f"Rank {pair['rank']}: Similarity = {pair['similarity']:.4f}\n")
                f.write(f"  Job A: {pair['job1_title']} @ {pair['job1_company']} ({pair['job1_country']})\n")
                f.write(f"  Job B: {pair['job2_title']} @ {pair['job2_company']} ({pair['job2_country']})\n\n")
        
        print(f"\n✓ Saved to {OUTPUT_DIR}/similar_pairs.txt")
        
        return similar_pairs
        
    def visualize_results(self):
        """Create visualizations"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # 1. Similarity distribution
        plt.figure(figsize=(12, 6))
        
        # Flatten similarity matrix (exclude diagonal)
        sim_values = self.similarity_matrix[~np.eye(self.similarity_matrix.shape[0], dtype=bool)]
        sim_values = sim_values[~np.isnan(sim_values)]
        
        plt.hist(sim_values, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(self.metrics['sim_mean'], color='red', linestyle='--', 
                   label=f"Mean: {self.metrics['sim_mean']:.4f}", linewidth=2)
        plt.axvline(self.metrics['sim_median'], color='green', linestyle='--', 
                   label=f"Median: {self.metrics['sim_median']:.4f}", linewidth=2)
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.title('Distribution of Pairwise Cosine Similarities\nTF-IDF Vectors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/similarity_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved similarity_distribution.png")
        
        # 2. Similarity heatmap (sample)
        if self.similarity_matrix.shape[0] <= 100:
            n_show = self.similarity_matrix.shape[0]
        else:
            n_show = 50
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.similarity_matrix[:n_show, :n_show], 
                   cmap='RdYlGn', center=0.5, square=True,
                   cbar_kws={'label': 'Cosine Similarity'},
                   xticklabels=False, yticklabels=False)
        plt.title(f'Cosine Similarity Heatmap (First {n_show} Documents)')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/similarity_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved similarity_heatmap.png")
        
        # 3. Document length distribution
        plt.figure(figsize=(12, 6))
        plt.hist(self.df['desc_length'], bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(self.df['desc_length'].mean(), color='red', linestyle='--', 
                   label=f"Mean: {self.df['desc_length'].mean():.1f}", linewidth=2)
        plt.axvline(self.df['desc_length'].median(), color='green', linestyle='--', 
                   label=f"Median: {self.df['desc_length'].median():.1f}", linewidth=2)
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.title('Distribution of Document Lengths (Cleaned Descriptions)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/document_lengths.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved document_lengths.png")
        
        # 4. Top terms bar chart
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        avg_tfidf = np.asarray(self.tfidf_matrix.mean(axis=0)).flatten()
        top_20_indices = avg_tfidf.argsort()[::-1][:20]
        
        plt.figure(figsize=(12, 8))
        top_terms = feature_names[top_20_indices]
        top_scores = avg_tfidf[top_20_indices]
        
        colors = ['steelblue' if ' ' not in term else 'coral' for term in top_terms]
        
        plt.barh(range(len(top_terms)), top_scores, color=colors, alpha=0.8)
        plt.yticks(range(len(top_terms)), top_terms)
        plt.xlabel('Average TF-IDF Score')
        plt.title('Top 20 Terms by Average TF-IDF Score\n(Blue = Unigram, Orange = Bigram)')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/top_terms.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved top_terms.png")
        
    def save_metrics(self):
        """Save all metrics to file"""
        print("\n" + "="*60)
        print("SAVING METRICS")
        print("="*60)
        
        with open(f"{OUTPUT_DIR}/tfidf_metrics.txt", 'w') as f:
            f.write("TF-IDF + COSINE SIMILARITY METRICS\n")
            f.write("="*60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Database: {DB_PATH}\n\n")
            
            f.write("--- DATASET ---\n")
            f.write(f"Total documents: {self.metrics['n_docs']:,}\n")
            f.write(f"Sample size: {'Full dataset' if not self.sample_size else f'{self.sample_size:,}'}\n\n")
            
            f.write("--- REPRESENTATION QUALITY ---\n")
            f.write(f"Vocabulary size: {self.metrics['vocab_size']:,}\n")
            f.write(f"Feature dimensionality: {self.metrics['n_features']:,}\n")
            f.write(f"Sparsity: {self.metrics['sparsity']:.4f}\n")
            f.write(f"Avg document length: {self.df['desc_length'].mean():.1f} words\n")
            f.write(f"Median document length: {self.df['desc_length'].median():.1f} words\n\n")
            
            f.write("--- SIMILARITY STATISTICS ---\n")
            f.write(f"Mean: {self.metrics['sim_mean']:.4f}\n")
            f.write(f"Median: {self.metrics['sim_median']:.4f}\n")
            f.write(f"Std Dev: {self.metrics['sim_std']:.4f}\n")
            f.write(f"Min: {self.metrics['sim_min']:.4f}\n")
            f.write(f"Max: {self.metrics['sim_max']:.4f}\n")
            f.write(f"25th percentile: {self.metrics['sim_25']:.4f}\n")
            f.write(f"75th percentile: {self.metrics['sim_75']:.4f}\n")
            f.write(f"90th percentile: {self.metrics['sim_90']:.4f}\n")
            f.write(f"95th percentile: {self.metrics['sim_95']:.4f}\n")
            f.write(f"99th percentile: {self.metrics['sim_99']:.4f}\n\n")
            
            f.write("--- COMPUTATIONAL COST ---\n")
            f.write(f"Training time: {self.metrics['training_time']:.2f}s\n")
            f.write(f"Inference time: {self.metrics['inference_time']:.2f}s\n")
            f.write(f"Memory usage: {self.metrics['memory_mb']:.2f} MB\n")
        
        print(f"✓ Saved metrics to {OUTPUT_DIR}/tfidf_metrics.txt")
        
    def save_model(self):
        """Save trained model"""
        model_data = {
            'vectorizer': self.vectorizer,
            'feature_names': self.vectorizer.get_feature_names_out(),
            'metrics': self.metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f"{OUTPUT_DIR}/tfidf_model.pkl", 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Saved model to {OUTPUT_DIR}/tfidf_model.pkl")
        
    def run_full_pipeline(self):
        """Execute complete analysis pipeline"""
        print("\n" + "="*60)
        print("  TF-IDF + COSINE SIMILARITY BASELINE")
        print("="*60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {OUTPUT_DIR}")
        
        # Execute pipeline
        self.load_data()
        self.build_tfidf()
        self.compute_similarity()
        self.analyze_top_terms()
        self.find_similar_pairs()
        self.visualize_results()
        self.save_metrics()
        self.save_model()
        
        print("\n" + "="*60)
        print("✓ ANALYSIS COMPLETE!")
        print("="*60)
        print(f"\nOutputs saved to: {OUTPUT_DIR}/")
        print("  - tfidf_metrics.txt")
        print("  - top_terms.txt")
        print("  - similar_pairs.txt")
        print("  - similarity_distribution.png")
        print("  - similarity_heatmap.png")
        print("  - document_lengths.png")
        print("  - top_terms.png")
        print("  - tfidf_model.pkl")


def main():
    parser = argparse.ArgumentParser(description='TF-IDF + Cosine Similarity Baseline')
    parser.add_argument('--sample', type=int, default=10000,
                       help='Number of documents to sample (default: 10000)')
    parser.add_argument('--full', action='store_true',
                       help='Use full dataset (155K docs, slower)')
    
    args = parser.parse_args()
    
    sample_size = None if args.full else args.sample
    
    baseline = TFIDFBaseline(sample_size=sample_size)
    baseline.run_full_pipeline()


if __name__ == '__main__':
    main()