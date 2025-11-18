"""
LDA Topic Modeling Baseline

This script implements Latent Dirichlet Allocation (LDA) for discovering
hidden topics in job descriptions. Includes comprehensive evaluation metrics.

Usage:
    python baseline/lda_baseline.py --sample 10000 --topics 10
    python baseline/lda_baseline.py --full --topics 15
    python baseline/lda_baseline.py --find-optimal-k  # Test multiple K values
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import argparse
import time
import pickle
import os
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configuration
DB_PATH = 'jobs_database.db'
OUTPUT_DIR = 'outputs/baseline/lda'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/wordclouds", exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class LDABaseline:
    def __init__(self, n_topics=10, sample_size=None):
        """
        Initialize LDA baseline
        
        Args:
            n_topics: Number of topics to discover
            sample_size: Number of documents to sample (None = use all)
        """
        self.n_topics = n_topics
        self.sample_size = sample_size
        self.vectorizer = None
        self.count_matrix = None
        self.lda_model = None
        self.df = None
        self.doc_topic_matrix = None
        self.topic_word_matrix = None
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
        
    def build_lda(self):
        """Build LDA topic model"""
        print("\n" + "="*60)
        print("BUILDING LDA TOPIC MODEL")
        print("="*60)
        
        start_time = time.time()
        
        self.vectorizer = CountVectorizer(
            max_features=5000,
            min_df=5,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        print("Vectorizer configuration:")
        print(f"  max_features: 5,000")
        print(f"  min_df: 5 (word must appear in ≥5 docs)")
        print(f"  max_df: 0.8 (ignore words in >80% of docs)")
        print(f"  ngram_range: (1, 2) (unigrams + bigrams)")
        print(f"  Using COUNT vectors (not TF-IDF)")
        
        # Transform to count matrix
        corpus = self.df['cleaned_description'].fillna('').tolist()
        self.count_matrix = self.vectorizer.fit_transform(corpus)
        
        print(f"\n✓ Count matrix created:")
        print(f"  Shape: {self.count_matrix.shape} (docs × features)")
        print(f"  Vocabulary size: {len(self.vectorizer.vocabulary_):,}")
        
        # Train LDA
        print(f"\nTraining LDA with {self.n_topics} topics...")
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=50,
            learning_method='online',
            learning_offset=50.,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        self.doc_topic_matrix = self.lda_model.fit_transform(self.count_matrix)
        self.topic_word_matrix = self.lda_model.components_
        
        training_time = time.time() - start_time
        
        # Store metrics
        self.metrics['training_time'] = training_time
        self.metrics['n_docs'] = self.count_matrix.shape[0]
        self.metrics['n_features'] = self.count_matrix.shape[1]
        self.metrics['n_topics'] = self.n_topics
        
        print(f"✓ LDA model trained in {training_time:.2f}s")
        print(f"  Document-topic matrix: {self.doc_topic_matrix.shape}")
        print(f"  Topic-word matrix: {self.topic_word_matrix.shape}")
        
    def evaluate_model(self):
        """Evaluate LDA model quality"""
        print("\n" + "="*60)
        print("EVALUATING MODEL QUALITY")
        print("="*60)
        
        # Perplexity (lower is better)
        perplexity = self.lda_model.perplexity(self.count_matrix)
        self.metrics['perplexity'] = perplexity
        print(f"Perplexity: {perplexity:.2f} (lower = better)")
        
        # Log-likelihood (higher is better)
        log_likelihood = self.lda_model.score(self.count_matrix)
        self.metrics['log_likelihood'] = log_likelihood
        print(f"Log-likelihood: {log_likelihood:.2f} (higher = better)")
        
        # Topic concentration
        topic_prevalence = self.doc_topic_matrix.mean(axis=0)
        self.metrics['topic_prevalence'] = topic_prevalence
        print(f"\nTopic prevalence (% of corpus):")
        for i, prev in enumerate(topic_prevalence):
            print(f"  Topic {i}: {prev*100:.2f}%")
        
        # Topic diversity (entropy)
        entropy = -np.sum(topic_prevalence * np.log(topic_prevalence + 1e-10))
        self.metrics['topic_entropy'] = entropy
        print(f"\nTopic entropy: {entropy:.4f} (higher = more diverse)")
        
        # Dominant topic per document
        dominant_topics = self.doc_topic_matrix.argmax(axis=1)
        topic_counts = Counter(dominant_topics)
        print(f"\nDocuments per dominant topic:")
        for topic_id in range(self.n_topics):
            count = topic_counts.get(topic_id, 0)
            pct = (count / len(self.df)) * 100
            print(f"  Topic {topic_id}: {count:,} docs ({pct:.1f}%)")
        
    def extract_topics(self):
        """Extract and display top words per topic"""
        print("\n" + "="*60)
        print("EXTRACTING TOPICS")
        print("="*60)
        
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        topics_data = []
        
        print("\nTop 15 words per topic:")
        print("="*60)
        
        for topic_idx, topic in enumerate(self.topic_word_matrix):
            top_indices = topic.argsort()[-15:][::-1]
            top_words = feature_names[top_indices]
            top_weights = topic[top_indices]
            
            print(f"\nTopic {topic_idx}:")
            print(f"  Words: {', '.join(top_words)}")
            print(f"  Weights: {', '.join([f'{w:.4f}' for w in top_weights[:5]])}...")
            
            topics_data.append({
                'topic_id': topic_idx,
                'top_words': list(top_words),
                'top_weights': list(top_weights)
            })
        
        # Save to file
        with open(f"{OUTPUT_DIR}/lda_topics.txt", 'w', encoding='utf-8') as f:
            f.write("LDA TOPICS - TOP 15 WORDS PER TOPIC\n")
            f.write("="*60 + "\n\n")
            f.write(f"Number of topics: {self.n_topics}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for topic_data in topics_data:
                f.write(f"TOPIC {topic_data['topic_id']}\n")
                f.write("-"*60 + "\n")
                f.write("Top words:\n")
                for word, weight in zip(topic_data['top_words'], topic_data['top_weights']):
                    f.write(f"  {word:<30} {weight:.6f}\n")
                f.write("\n")
                
                # Suggest manual label placeholder
                f.write("Manual label: [TO BE ASSIGNED]\n")
                f.write("="*60 + "\n\n")
        
        print(f"\n✓ Saved topics to {OUTPUT_DIR}/lda_topics.txt")
        
        return topics_data
        
    def analyze_topics_by_country(self):
        """Analyze topic distribution by country"""
        print("\n" + "="*60)
        print("ANALYZING TOPICS BY COUNTRY")
        print("="*60)
        
        # Filter out null countries
        df_with_country = self.df[self.df['country'].notna()].copy()
        
        if len(df_with_country) == 0:
            print("⚠ No country data available")
            return
        
        # Get document-topic matrix for these documents
        doc_topics = self.doc_topic_matrix[df_with_country.index]
        
        # Group by country
        countries = df_with_country['country'].unique()
        top_countries = df_with_country['country'].value_counts().head(15).index
        
        country_topic_dist = []
        for country in top_countries:
            country_docs = df_with_country[df_with_country['country'] == country].index
            country_doc_topics = self.doc_topic_matrix[country_docs]
            avg_topics = country_doc_topics.mean(axis=0)
            country_topic_dist.append(avg_topics)
        
        country_topic_matrix = np.array(country_topic_dist)
        
        print(f"\nTop 5 topics per country (top 15 countries):")
        for i, country in enumerate(top_countries):
            top_topic_indices = country_topic_matrix[i].argsort()[-3:][::-1]
            print(f"\n{country}:")
            for topic_idx in top_topic_indices:
                pct = country_topic_matrix[i][topic_idx] * 100
                print(f"  Topic {topic_idx}: {pct:.1f}%")
        
        # Save to file
        with open(f"{OUTPUT_DIR}/topics_by_country.txt", 'w', encoding='utf-8') as f:
            f.write("TOPIC DISTRIBUTION BY COUNTRY\n")
            f.write("="*60 + "\n\n")
            for i, country in enumerate(top_countries):
                f.write(f"{country}:\n")
                for topic_idx in range(self.n_topics):
                    pct = country_topic_matrix[i][topic_idx] * 100
                    f.write(f"  Topic {topic_idx}: {pct:.2f}%\n")
                f.write("\n")
        
        print(f"\n✓ Saved to {OUTPUT_DIR}/topics_by_country.txt")
        
        return country_topic_matrix, top_countries
        
    def analyze_topics_by_title(self):
        """Analyze which job titles belong to which topics"""
        print("\n" + "="*60)
        print("ANALYZING TOPICS BY JOB TITLE")
        print("="*60)
        
        # Get dominant topic per document
        dominant_topics = self.doc_topic_matrix.argmax(axis=1)
        self.df['dominant_topic'] = dominant_topics
        
        print("\nTop 10 job titles per topic:")
        
        topic_titles = {}
        for topic_id in range(self.n_topics):
            topic_docs = self.df[self.df['dominant_topic'] == topic_id]
            top_titles = topic_docs['title'].value_counts().head(10)
            
            print(f"\nTopic {topic_id} ({len(topic_docs)} documents):")
            for title, count in top_titles.items():
                print(f"  {title}: {count}")
            
            topic_titles[topic_id] = top_titles
        
        # Save to file
        with open(f"{OUTPUT_DIR}/topics_by_title.txt", 'w', encoding='utf-8') as f:
            f.write("TOP JOB TITLES PER TOPIC\n")
            f.write("="*60 + "\n\n")
            for topic_id, titles in topic_titles.items():
                topic_docs = self.df[self.df['dominant_topic'] == topic_id]
                f.write(f"TOPIC {topic_id} ({len(topic_docs)} documents)\n")
                f.write("-"*60 + "\n")
                for title, count in titles.items():
                    f.write(f"  {title}: {count}\n")
                f.write("\n")
        
        print(f"\n✓ Saved to {OUTPUT_DIR}/topics_by_title.txt")
        
        return topic_titles
        
    def compare_topics_with_skills(self):
        """Compare topics with explicitly listed skills"""
        print("\n" + "="*60)
        print("COMPARING TOPICS WITH SKILLS")
        print("="*60)
        
        # Get dominant topic per document
        dominant_topics = self.doc_topic_matrix.argmax(axis=1)
        self.df['dominant_topic'] = dominant_topics
        
        print("\nTop 10 skills per topic:")
        
        topic_skills = {}
        for topic_id in range(self.n_topics):
            topic_docs = self.df[self.df['dominant_topic'] == topic_id]
            
            # Extract and count skills
            all_skills = []
            for skills_str in topic_docs['cleaned_skills'].dropna():
                if isinstance(skills_str, str):
                    skills = [s.strip() for s in skills_str.split(',') if s.strip()]
                    all_skills.extend(skills)
            
            skill_counts = Counter(all_skills)
            top_skills = skill_counts.most_common(10)
            
            print(f"\nTopic {topic_id}:")
            for skill, count in top_skills:
                print(f"  {skill}: {count}")
            
            topic_skills[topic_id] = top_skills
        
        # Save to file
        with open(f"{OUTPUT_DIR}/topics_vs_skills.txt", 'w', encoding='utf-8') as f:
            f.write("TOP SKILLS PER TOPIC\n")
            f.write("="*60 + "\n\n")
            for topic_id, skills in topic_skills.items():
                topic_docs = self.df[self.df['dominant_topic'] == topic_id]
                f.write(f"TOPIC {topic_id} ({len(topic_docs)} documents)\n")
                f.write("-"*60 + "\n")
                for skill, count in skills:
                    f.write(f"  {skill}: {count}\n")
                f.write("\n")
        
        print(f"\n✓ Saved to {OUTPUT_DIR}/topics_vs_skills.txt")
        
        return topic_skills
        
    def visualize_results(self):
        """Create visualizations"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # 1. Topic prevalence bar chart
        plt.figure(figsize=(12, 6))
        topic_prevalence = self.doc_topic_matrix.mean(axis=0)
        plt.bar(range(self.n_topics), topic_prevalence * 100, color='steelblue', alpha=0.8)
        plt.xlabel('Topic ID')
        plt.ylabel('Prevalence (%)')
        plt.title(f'Topic Prevalence Across {len(self.df):,} Documents')
        plt.xticks(range(self.n_topics))
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/topic_prevalence.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved topic_prevalence.png")
        
        # 2. Document-topic distribution heatmap (sample)
        n_show = min(100, len(self.df))
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.doc_topic_matrix[:n_show], 
                   cmap='YlOrRd', cbar_kws={'label': 'Topic Probability'},
                   xticklabels=[f'T{i}' for i in range(self.n_topics)],
                   yticklabels=False)
        plt.xlabel('Topics')
        plt.ylabel(f'Documents (first {n_show})')
        plt.title('Document-Topic Distribution Matrix')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/doc_topic_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved doc_topic_heatmap.png")
        
        # 3. Dominant topic distribution
        dominant_topics = self.doc_topic_matrix.argmax(axis=1)
        topic_counts = Counter(dominant_topics)
        
        plt.figure(figsize=(12, 6))
        topics = sorted(topic_counts.keys())
        counts = [topic_counts[t] for t in topics]
        plt.bar(topics, counts, color='coral', alpha=0.8)
        plt.xlabel('Dominant Topic')
        plt.ylabel('Number of Documents')
        plt.title('Distribution of Dominant Topics')
        plt.xticks(topics)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/dominant_topics.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved dominant_topics.png")
        
        # 4. Word clouds per topic
        from wordcloud import WordCloud
        
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        for topic_idx, topic in enumerate(self.topic_word_matrix):
            # Create word frequency dict
            word_freq = {}
            top_indices = topic.argsort()[-30:][::-1]
            for idx in top_indices:
                word_freq[feature_names[idx]] = topic[idx]
            
            # Generate word cloud
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='white',
                                colormap='viridis').generate_from_frequencies(word_freq)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Topic {topic_idx} - Word Cloud')
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/wordclouds/topic_{topic_idx}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✓ Saved {self.n_topics} word cloud images")
        
        # 5. Topics by country heatmap (if available)
        if 'country' in self.df.columns and self.df['country'].notna().any():
            country_topic_matrix, top_countries = self.analyze_topics_by_country()
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(country_topic_matrix.T, 
                       cmap='YlGnBu', annot=True, fmt='.2f',
                       xticklabels=top_countries,
                       yticklabels=[f'Topic {i}' for i in range(self.n_topics)],
                       cbar_kws={'label': 'Average Topic Probability'})
            plt.xlabel('Country')
            plt.ylabel('Topic')
            plt.title('Topic Distribution by Country (Top 15 Countries)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/topics_by_country.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved topics_by_country.png")
        
    def save_metrics(self):
        """Save all metrics to file"""
        print("\n" + "="*60)
        print("SAVING METRICS")
        print("="*60)
        
        with open(f"{OUTPUT_DIR}/lda_metrics.txt", 'w') as f:
            f.write("LDA TOPIC MODELING METRICS\n")
            f.write("="*60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Database: {DB_PATH}\n\n")
            
            f.write("--- DATASET ---\n")
            f.write(f"Total documents: {self.metrics['n_docs']:,}\n")
            f.write(f"Sample size: {'Full dataset' if not self.sample_size else f'{self.sample_size:,}'}\n\n")
            
            f.write("--- MODEL CONFIGURATION ---\n")
            f.write(f"Number of topics: {self.metrics['n_topics']}\n")
            f.write(f"Vocabulary size: {self.metrics['n_features']:,}\n")
            f.write(f"Max iterations: 50\n")
            f.write(f"Learning method: online\n\n")
            
            f.write("--- MODEL QUALITY ---\n")
            f.write(f"Perplexity: {self.metrics['perplexity']:.4f} (lower = better)\n")
            f.write(f"Log-likelihood: {self.metrics['log_likelihood']:.4f} (higher = better)\n")
            f.write(f"Topic entropy: {self.metrics['topic_entropy']:.4f} (higher = more diverse)\n\n")
            
            f.write("--- TOPIC PREVALENCE ---\n")
            for i, prev in enumerate(self.metrics['topic_prevalence']):
                f.write(f"Topic {i}: {prev*100:.2f}%\n")
            f.write("\n")
            
            f.write("--- COMPUTATIONAL COST ---\n")
            f.write(f"Training time: {self.metrics['training_time']:.2f}s\n")
        
        print(f"✓ Saved metrics to {OUTPUT_DIR}/lda_metrics.txt")
        
    def save_model(self):
        """Save trained model"""
        model_data = {
            'lda_model': self.lda_model,
            'vectorizer': self.vectorizer,
            'feature_names': self.vectorizer.get_feature_names_out(),
            'doc_topic_matrix': self.doc_topic_matrix,
            'topic_word_matrix': self.topic_word_matrix,
            'metrics': self.metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f"{OUTPUT_DIR}/lda_model.pkl", 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Saved model to {OUTPUT_DIR}/lda_model.pkl")
        
        # Also save document-topic distributions as CSV
        doc_topic_df = pd.DataFrame(
            self.doc_topic_matrix,
            columns=[f'topic_{i}' for i in range(self.n_topics)]
        )
        doc_topic_df['dominant_topic'] = self.doc_topic_matrix.argmax(axis=1)
        doc_topic_df['id'] = self.df['id'].values
        doc_topic_df['title'] = self.df['title'].values
        
        doc_topic_df.to_csv(f"{OUTPUT_DIR}/document_topics.csv", index=False)
        print(f"✓ Saved document topics to {OUTPUT_DIR}/document_topics.csv")
        
    def run_full_pipeline(self):
        """Execute complete analysis pipeline"""
        print("\n" + "="*60)
        print("  LDA TOPIC MODELING BASELINE")
        print("="*60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Number of topics: {self.n_topics}")
        print(f"Output directory: {OUTPUT_DIR}")
        
        # Execute pipeline
        self.load_data()
        self.build_lda()
        self.evaluate_model()
        self.extract_topics()
        self.analyze_topics_by_title()
        self.compare_topics_with_skills()
        self.visualize_results()
        self.save_metrics()
        self.save_model()
        
        print("\n" + "="*60)
        print("✓ ANALYSIS COMPLETE!")
        print("="*60)
        print(f"\nOutputs saved to: {OUTPUT_DIR}/")
        print("  - lda_metrics.txt")
        print("  - lda_topics.txt")
        print("  - topics_by_title.txt")
        print("  - topics_vs_skills.txt")
        print("  - topics_by_country.txt")
        print("  - topic_prevalence.png")
        print("  - doc_topic_heatmap.png")
        print("  - dominant_topics.png")
        print("  - topics_by_country.png")
        print(f"  - wordclouds/ ({self.n_topics} images)")
        print("  - document_topics.csv")
        print("  - lda_model.pkl")


def find_optimal_k(sample_size=10000):
    """Find optimal number of topics"""
    print("\n" + "="*60)
    print("FINDING OPTIMAL NUMBER OF TOPICS")
    print("="*60)
    
    k_values = [5, 10, 15, 20, 25]
    perplexities = []
    log_likelihoods = []
    
    for k in k_values:
        print(f"\nTesting K={k}...")
        baseline = LDABaseline(n_topics=k, sample_size=sample_size)
        baseline.load_data()
        baseline.build_lda()
        baseline.evaluate_model()
        
        perplexities.append(baseline.metrics['perplexity'])
        log_likelihoods.append(baseline.metrics['log_likelihood'])
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(k_values, perplexities, marker='o', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Topics (K)')
    ax1.set_ylabel('Perplexity')
    ax1.set_title('Perplexity vs. Number of Topics\n(Lower is better)')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(k_values, log_likelihoods, marker='o', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Number of Topics (K)')
    ax2.set_ylabel('Log-Likelihood')
    ax2.set_title('Log-Likelihood vs. Number of Topics\n(Higher is better)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/optimal_k_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved to {OUTPUT_DIR}/optimal_k_analysis.png")
    
    # Save results
    with open(f"{OUTPUT_DIR}/optimal_k_results.txt", 'w') as f:
        f.write("OPTIMAL K ANALYSIS\n")
        f.write("="*60 + "\n\n")
        f.write(f"{'K':<8} {'Perplexity':<15} {'Log-Likelihood':<15}\n")
        f.write("-"*40 + "\n")
        for k, perp, ll in zip(k_values, perplexities, log_likelihoods):
            f.write(f"{k:<8} {perp:<15.4f} {ll:<15.4f}\n")
    
    print(f"✓ Saved to {OUTPUT_DIR}/optimal_k_results.txt")


def main():
    parser = argparse.ArgumentParser(description='LDA Topic Modeling Baseline')
    parser.add_argument('--sample', type=int, default=10000,
                       help='Number of documents to sample (default: 10000)')
    parser.add_argument('--full', action='store_true',
                       help='Use full dataset (155K docs, slower)')
    parser.add_argument('--topics', type=int, default=10,
                       help='Number of topics (default: 10)')
    parser.add_argument('--find-optimal-k', action='store_true',
                       help='Test multiple K values to find optimal')
    
    args = parser.parse_args()
    
    if args.find_optimal_k:
        sample_size = None if args.full else args.sample
        find_optimal_k(sample_size=sample_size)
    else:
        sample_size = None if args.full else args.sample
        baseline = LDABaseline(n_topics=args.topics, sample_size=sample_size)
        baseline.run_full_pipeline()


if __name__ == '__main__':
    main()