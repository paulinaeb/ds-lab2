"""
Word2Vec Baseline

This script implements Word2Vec embeddings with document-level averaging
for job description analysis. Includes comprehensive evaluation metrics.

Usage:
    python baseline/word2vec_baseline.py --sample 10000
    python baseline/word2vec_baseline.py --full
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
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
OUTPUT_DIR = 'outputs/baseline/word2vec'
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class Word2VecBaseline:
    def __init__(self, vector_size=100, window=5, sample_size=None):
        """
        Initialize Word2Vec baseline
        
        Args:
            vector_size: Dimension of word embeddings (default: 100)
            window: Context window size (default: 5)
            sample_size: Number of documents to sample (None = use all)
        """
        self.vector_size = vector_size
        self.window = window
        self.sample_size = sample_size
        self.model = None
        self.df = None
        self.doc_embeddings = None
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
        
    def train_word2vec(self):
        """Train Word2Vec model"""
        print("\n" + "="*60)
        print("TRAINING WORD2VEC MODEL")
        print("="*60)
        
        start_time = time.time()
        
        # Tokenize documents
        print("Tokenizing documents...")
        corpus = self.df['cleaned_description'].fillna('').tolist()
        tokenized_docs = [simple_preprocess(doc, deacc=True) for doc in corpus]
        
        print(f"✓ Tokenized {len(tokenized_docs):,} documents")
        
        # Train Word2Vec
        print(f"\nTraining Word2Vec model...")
        print(f"  Vector size: {self.vector_size}")
        print(f"  Window: {self.window}")
        print(f"  Algorithm: Skip-gram")
        print(f"  Min count: 5")
        print(f"  Epochs: 10")
        
        self.model = Word2Vec(
            sentences=tokenized_docs,
            vector_size=self.vector_size,
            window=self.window,
            min_count=5,
            workers=4,
            sg=1,  # Skip-gram
            epochs=10,
            seed=42
        )
        
        training_time = time.time() - start_time
        
        # Store metrics
        self.metrics['training_time'] = training_time
        self.metrics['vocab_size'] = len(self.model.wv)
        self.metrics['vector_size'] = self.vector_size
        self.metrics['n_docs'] = len(tokenized_docs)
        
        print(f"\n✓ Model trained in {training_time:.2f}s")
        print(f"  Vocabulary size: {self.metrics['vocab_size']:,} words")
        print(f"  Vector dimension: {self.vector_size}")
        
        # Calculate vocabulary coverage
        all_words = set()
        for doc in tokenized_docs:
            all_words.update(doc)
        
        covered_words = sum(1 for word in all_words if word in self.model.wv)
        coverage = covered_words / len(all_words)
        self.metrics['vocab_coverage'] = coverage
        
        print(f"  Vocabulary coverage: {coverage*100:.2f}%")
        
    def create_document_embeddings(self):
        """Create document embeddings by averaging word vectors"""
        print("\n" + "="*60)
        print("CREATING DOCUMENT EMBEDDINGS")
        print("="*60)
        
        start_time = time.time()
        
        corpus = self.df['cleaned_description'].fillna('').tolist()
        doc_embeddings = []
        
        print("Averaging word vectors for each document...")
        for i, doc in enumerate(corpus):
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1:,}/{len(corpus):,} documents")
            
            tokens = simple_preprocess(doc, deacc=True)
            word_vectors = [self.model.wv[word] for word in tokens if word in self.model.wv]
            
            if len(word_vectors) == 0:
                # If no words in vocabulary, use zero vector
                doc_embeddings.append(np.zeros(self.vector_size))
            else:
                # Average word vectors
                doc_embeddings.append(np.mean(word_vectors, axis=0))
        
        self.doc_embeddings = np.array(doc_embeddings)
        
        inference_time = time.time() - start_time
        self.metrics['inference_time'] = inference_time
        
        print(f"\n✓ Document embeddings created:")
        print(f"  Shape: {self.doc_embeddings.shape}")
        print(f"  Time: {inference_time:.2f}s")
        
        # Calculate statistics
        zero_docs = np.sum(np.all(self.doc_embeddings == 0, axis=1))
        print(f"  Documents with zero embedding: {zero_docs} ({zero_docs/len(self.df)*100:.2f}%)")
        
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
            sample_embeddings = self.doc_embeddings[sample_indices]
            self.similarity_matrix = cosine_similarity(sample_embeddings)
            print(f"✓ Similarity matrix shape: {self.similarity_matrix.shape}")
        else:
            self.similarity_matrix = cosine_similarity(self.doc_embeddings)
            print(f"✓ Similarity matrix shape: {self.similarity_matrix.shape}")
        
        computation_time = time.time() - start_time
        
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
        print(f"  Computation time: {computation_time:.2f}s")
        
    def analyze_word_similarities(self):
        """Analyze word-level similarities"""
        print("\n" + "="*60)
        print("ANALYZING WORD SIMILARITIES")
        print("="*60)
        
        # Key terms to analyze
        key_terms = ['security', 'cloud', 'python', 'analyst', 'engineer', 
                     'network', 'data', 'management', 'risk', 'compliance']
        
        word_similarities = {}
        
        print("\nTop 10 similar words for key terms:")
        print("="*60)
        
        for term in key_terms:
            if term in self.model.wv:
                similar_words = self.model.wv.most_similar(term, topn=10)
                word_similarities[term] = similar_words
                
                print(f"\n'{term}':")
                for word, score in similar_words:
                    print(f"  {word:<20} {score:.4f}")
            else:
                print(f"\n'{term}': NOT IN VOCABULARY")
        
        # Save to file
        with open(f"{OUTPUT_DIR}/word_similarities.txt", 'w', encoding='utf-8') as f:
            f.write("WORD SIMILARITIES - TOP 10 SIMILAR WORDS\n")
            f.write("="*60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for term, similar_words in word_similarities.items():
                f.write(f"\n'{term}':\n")
                f.write("-"*40 + "\n")
                for word, score in similar_words:
                    f.write(f"  {word:<20} {score:.4f}\n")
        
        print(f"\n✓ Saved to {OUTPUT_DIR}/word_similarities.txt")
        
        return word_similarities
        
    def analyze_skill_embeddings(self):
        """Analyze skill embeddings and clustering"""
        print("\n" + "="*60)
        print("ANALYZING SKILL EMBEDDINGS")
        print("="*60)
        
        # Extract unique skills from dataset
        all_skills = set()
        for skills_str in self.df['cleaned_skills'].dropna():
            if isinstance(skills_str, str):
                skills = [s.strip().lower() for s in skills_str.split(',') if s.strip()]
                all_skills.update(skills)
        
        # Filter skills that exist in Word2Vec vocabulary
        skill_embeddings = {}
        for skill in all_skills:
            # Try exact match and common variations
            skill_variants = [skill, skill.replace(' ', ''), skill.replace('-', '')]
            for variant in skill_variants:
                if variant in self.model.wv:
                    skill_embeddings[skill] = self.model.wv[variant]
                    break
        
        print(f"\nFound {len(skill_embeddings)} skills in vocabulary (out of {len(all_skills)})")
        print(f"Coverage: {len(skill_embeddings)/len(all_skills)*100:.1f}%")
        
        if len(skill_embeddings) == 0:
            print("⚠ No skills found in vocabulary")
            return None
        
        # Get most common skills
        skill_counts = Counter()
        for skills_str in self.df['cleaned_skills'].dropna():
            if isinstance(skills_str, str):
                skills = [s.strip().lower() for s in skills_str.split(',') if s.strip()]
                skill_counts.update(skills)
        
        # Analyze top 50 skills
        top_skills = [skill for skill, count in skill_counts.most_common(50) 
                     if skill in skill_embeddings][:30]
        
        if len(top_skills) < 5:
            print("⚠ Not enough skills for clustering analysis")
            return None
        
        print(f"\nAnalyzing top {len(top_skills)} skills")
        
        # Create skill similarity matrix
        skill_vectors = np.array([skill_embeddings[skill] for skill in top_skills])
        skill_similarity = cosine_similarity(skill_vectors)
        
        # Cluster skills
        n_clusters = min(5, len(top_skills) // 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        skill_clusters = kmeans.fit_predict(skill_vectors)
        
        print(f"\nSkill clusters (K={n_clusters}):")
        for cluster_id in range(n_clusters):
            cluster_skills = [top_skills[i] for i in range(len(top_skills)) 
                            if skill_clusters[i] == cluster_id]
            print(f"\nCluster {cluster_id}: {len(cluster_skills)} skills")
            print(f"  {', '.join(cluster_skills[:10])}")
        
        # Save results
        with open(f"{OUTPUT_DIR}/skill_clusters.txt", 'w', encoding='utf-8') as f:
            f.write("SKILL CLUSTERING RESULTS\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total skills analyzed: {len(top_skills)}\n")
            f.write(f"Number of clusters: {n_clusters}\n\n")
            
            for cluster_id in range(n_clusters):
                cluster_skills = [top_skills[i] for i in range(len(top_skills)) 
                                if skill_clusters[i] == cluster_id]
                f.write(f"\nCluster {cluster_id} ({len(cluster_skills)} skills):\n")
                f.write("-"*40 + "\n")
                for skill in cluster_skills:
                    f.write(f"  - {skill}\n")
        
        print(f"\n✓ Saved to {OUTPUT_DIR}/skill_clusters.txt")
        
        return top_skills, skill_similarity, skill_clusters
        
    def find_similar_pairs(self):
        """Find most similar document pairs"""
        print("\n" + "="*60)
        print("FINDING SIMILAR DOCUMENT PAIRS")
        print("="*60)
        
        # Get upper triangle indices
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
            f.write("MOST SIMILAR JOB PAIRS (Word2Vec)\n")
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
        
        sim_values = self.similarity_matrix[~np.eye(self.similarity_matrix.shape[0], dtype=bool)]
        sim_values = sim_values[~np.isnan(sim_values)]
        
        plt.hist(sim_values, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        plt.axvline(self.metrics['sim_mean'], color='red', linestyle='--', 
                   label=f"Mean: {self.metrics['sim_mean']:.4f}", linewidth=2)
        plt.axvline(self.metrics['sim_median'], color='green', linestyle='--', 
                   label=f"Median: {self.metrics['sim_median']:.4f}", linewidth=2)
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.title('Distribution of Pairwise Cosine Similarities\nWord2Vec Document Embeddings')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/similarity_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved similarity_distribution.png")
        
        # 2. Similarity heatmap
        n_show = min(50, self.similarity_matrix.shape[0])
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.similarity_matrix[:n_show, :n_show], 
                   cmap='RdYlGn', center=0.5, square=True,
                   cbar_kws={'label': 'Cosine Similarity'},
                   xticklabels=False, yticklabels=False)
        plt.title(f'Cosine Similarity Heatmap (First {n_show} Documents)\nWord2Vec Embeddings')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/similarity_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved similarity_heatmap.png")
        
        # 3. t-SNE visualization of word embeddings
        print("\nGenerating t-SNE visualization (this may take a minute)...")
        
        # Get most common words
        vocab_words = list(self.model.wv.index_to_key)[:500]  # Top 500 words
        word_vectors = np.array([self.model.wv[word] for word in vocab_words])
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        word_tsne = tsne.fit_transform(word_vectors)
        
        # Plot
        plt.figure(figsize=(14, 10))
        plt.scatter(word_tsne[:, 0], word_tsne[:, 1], alpha=0.5, s=10)
        
        # Annotate some interesting words
        interesting_words = ['security', 'cloud', 'python', 'analyst', 'engineer',
                           'network', 'management', 'data', 'aws', 'azure']
        for word in interesting_words:
            if word in vocab_words:
                idx = vocab_words.index(word)
                plt.annotate(word, (word_tsne[idx, 0], word_tsne[idx, 1]),
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.title('t-SNE Visualization of Word Embeddings (Top 500 Words)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/word_embeddings_tsne.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved word_embeddings_tsne.png")
        
        # 4. Skill similarity heatmap (if available)
        skill_data = self.analyze_skill_embeddings()
        if skill_data:
            top_skills, skill_similarity, skill_clusters = skill_data
            
            plt.figure(figsize=(14, 12))
            sns.heatmap(skill_similarity, 
                       xticklabels=top_skills, yticklabels=top_skills,
                       cmap='RdYlGn', center=0.5,
                       cbar_kws={'label': 'Cosine Similarity'})
            plt.title(f'Skill Similarity Matrix (Top {len(top_skills)} Skills)')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/skill_similarity_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved skill_similarity_heatmap.png")
        
    def save_metrics(self):
        """Save all metrics to file"""
        print("\n" + "="*60)
        print("SAVING METRICS")
        print("="*60)
        
        with open(f"{OUTPUT_DIR}/word2vec_metrics.txt", 'w') as f:
            f.write("WORD2VEC + COSINE SIMILARITY METRICS\n")
            f.write("="*60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Database: {DB_PATH}\n\n")
            
            f.write("--- DATASET ---\n")
            f.write(f"Total documents: {self.metrics['n_docs']:,}\n")
            f.write(f"Sample size: {'Full dataset' if not self.sample_size else f'{self.sample_size:,}'}\n\n")
            
            f.write("--- MODEL CONFIGURATION ---\n")
            f.write(f"Algorithm: Skip-gram\n")
            f.write(f"Vector size: {self.metrics['vector_size']}\n")
            f.write(f"Window size: {self.window}\n")
            f.write(f"Min count: 5\n")
            f.write(f"Epochs: 10\n\n")
            
            f.write("--- REPRESENTATION QUALITY ---\n")
            f.write(f"Vocabulary size: {self.metrics['vocab_size']:,} words\n")
            f.write(f"Vocabulary coverage: {self.metrics['vocab_coverage']*100:.2f}%\n")
            f.write(f"Embedding dimension: {self.metrics['vector_size']}\n")
            f.write(f"Representation: Dense (vs. sparse TF-IDF)\n\n")
            
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
        
        print(f"✓ Saved metrics to {OUTPUT_DIR}/word2vec_metrics.txt")
        
    def save_model(self):
        """Save trained model and embeddings"""
        # Save Word2Vec model
        self.model.save(f"{OUTPUT_DIR}/word2vec_model.bin")
        print(f"✓ Saved Word2Vec model to {OUTPUT_DIR}/word2vec_model.bin")
        
        # Save document embeddings
        np.save(f"{OUTPUT_DIR}/document_embeddings.npy", self.doc_embeddings)
        print(f"✓ Saved document embeddings to {OUTPUT_DIR}/document_embeddings.npy")
        
        # Save metadata
        model_data = {
            'vector_size': self.vector_size,
            'window': self.window,
            'vocab_size': self.metrics['vocab_size'],
            'metrics': self.metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f"{OUTPUT_DIR}/model_metadata.pkl", 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Saved metadata to {OUTPUT_DIR}/model_metadata.pkl")
        
    def run_full_pipeline(self):
        """Execute complete analysis pipeline"""
        print("\n" + "="*60)
        print("  WORD2VEC BASELINE")
        print("="*60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Vector size: {self.vector_size}")
        print(f"Window size: {self.window}")
        print(f"Output directory: {OUTPUT_DIR}")
        
        # Execute pipeline
        self.load_data()
        self.train_word2vec()
        self.create_document_embeddings()
        self.compute_similarity()
        self.analyze_word_similarities()
        self.find_similar_pairs()
        self.visualize_results()
        self.save_metrics()
        self.save_model()
        
        print("\n" + "="*60)
        print("✓ ANALYSIS COMPLETE!")
        print("="*60)
        print(f"\nOutputs saved to: {OUTPUT_DIR}/")
        print("  - word2vec_metrics.txt")
        print("  - word_similarities.txt")
        print("  - skill_clusters.txt")
        print("  - similar_pairs.txt")
        print("  - similarity_distribution.png")
        print("  - similarity_heatmap.png")
        print("  - word_embeddings_tsne.png")
        print("  - skill_similarity_heatmap.png")
        print("  - word2vec_model.bin")
        print("  - document_embeddings.npy")
        print("  - model_metadata.pkl")


def main():
    parser = argparse.ArgumentParser(description='Word2Vec Baseline')
    parser.add_argument('--sample', type=int, default=10000,
                       help='Number of documents to sample (default: 10000)')
    parser.add_argument('--full', action='store_true',
                       help='Use full dataset (155K docs, slower)')
    parser.add_argument('--vector-size', type=int, default=100,
                       help='Embedding dimension (default: 100)')
    parser.add_argument('--window', type=int, default=5,
                       help='Context window size (default: 5)')
    
    args = parser.parse_args()
    
    sample_size = None if args.full else args.sample
    
    baseline = Word2VecBaseline(
        vector_size=args.vector_size,
        window=args.window,
        sample_size=sample_size
    )
    baseline.run_full_pipeline()


if __name__ == '__main__':
    main()