"""
Baseline Methods for Job Description Analysis

This script implements three baseline text representation methods:
1. TF-IDF + Cosine Similarity (Bag of Words)
2. LDA Topic Modeling (Latent Dirichlet Allocation)
3. Word Embedding Averaging (Word2Vec)

Usage:
    python baseline/baseline_methods.py --file data_cleaned.json --sample 1000
    python baseline/baseline_methods.py --file data_cleaned.json
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Text processing
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation

# Word embeddings
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


class BaselineMethods:
    """Baseline text representation methods for job descriptions"""
    
    def __init__(self, data_path, output_dir="outputs/baseline", sample_size=None):
        self.data_path = data_path
        self.output_dir = output_dir
        self.sample_size = sample_size
        self.df = None
        self.texts = None
        
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"  BASELINE METHODS - JOB DESCRIPTION ANALYSIS")
        print(f"{'='*60}\n")
        print(f"Output directory: {os.path.abspath(self.output_dir)}\n")
    
    def load_data(self):
        """Load and prepare job descriptions"""
        print("Loading data...")
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.df = pd.DataFrame(data)
            
            # Sample if requested
            if self.sample_size and self.sample_size < len(self.df):
                print(f"Sampling {self.sample_size} records from {len(self.df)} total...")
                self.df = self.df.sample(n=self.sample_size, random_state=42)
            
            # Extract descriptions
            self.texts = self.df['Description'].dropna().astype(str).tolist()
            
            print(f"✓ Loaded {len(self.texts)} job descriptions")
            print(f"✓ Average length: {np.mean([len(t) for t in self.texts]):.0f} characters\n")
            return True
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def clean_text(self, text):
        """Basic text cleaning"""
        # Lowercase
        text = text.lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    # ========================================================================
    # METHOD 1: TF-IDF + COSINE SIMILARITY
    # ========================================================================
    
    def tfidf_analysis(self):
        """Bag of Words: TF-IDF + Cosine Similarity"""
        print("\n" + "="*60)
        print("METHOD 1: TF-IDF + COSINE SIMILARITY")
        print("="*60)
        
        try:
            # Clean texts
            print("Cleaning texts...")
            cleaned_texts = [self.clean_text(t) for t in self.texts]
            
            # Create TF-IDF matrix
            print("Creating TF-IDF vectors...")
            vectorizer = TfidfVectorizer(
                max_features=1000,
                min_df=5,
                max_df=0.8,
                stop_words='english'
            )
            tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
            
            print(f"✓ TF-IDF matrix shape: {tfidf_matrix.shape}")
            print(f"  - Documents: {tfidf_matrix.shape[0]}")
            print(f"  - Features (terms): {tfidf_matrix.shape[1]}")
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculate average TF-IDF scores
            avg_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            top_indices = avg_tfidf.argsort()[-20:][::-1]
            
            print("\nTop 20 terms by average TF-IDF score:")
            for idx in top_indices:
                print(f"  {feature_names[idx]}: {avg_tfidf[idx]:.4f}")
            
            # Compute cosine similarity matrix (sample for efficiency)
            print("\nComputing cosine similarities...")
            sample_size = min(100, tfidf_matrix.shape[0])
            sample_matrix = tfidf_matrix[:sample_size]
            similarity_matrix = cosine_similarity(sample_matrix)
            
            print(f"✓ Similarity matrix shape: {similarity_matrix.shape}")
            print(f"  - Mean similarity: {similarity_matrix.mean():.4f}")
            print(f"  - Std similarity: {similarity_matrix.std():.4f}")
            
            # Find most similar job pairs
            print("\nTop 5 most similar job description pairs:")
            # Get upper triangle indices (avoid diagonal and duplicates)
            triu_indices = np.triu_indices_from(similarity_matrix, k=1)
            similarities = similarity_matrix[triu_indices]
            top_pairs = similarities.argsort()[-5:][::-1]
            
            for rank, pair_idx in enumerate(top_pairs, 1):
                i, j = triu_indices[0][pair_idx], triu_indices[1][pair_idx]
                sim = similarity_matrix[i, j]
                print(f"\n  {rank}. Similarity: {sim:.4f}")
                print(f"     Job {i}: {self.texts[i][:100]}...")
                print(f"     Job {j}: {self.texts[j][:100]}...")
            
            # Visualize similarity distribution
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Heatmap
            sns.heatmap(similarity_matrix[:50, :50], cmap='YlOrRd', ax=axes[0], cbar=True)
            axes[0].set_title('Cosine Similarity Heatmap (first 50 docs)')
            axes[0].set_xlabel('Document')
            axes[0].set_ylabel('Document')
            
            # Distribution
            axes[1].hist(similarities, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            axes[1].set_xlabel('Cosine Similarity')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Distribution of Pairwise Similarities')
            axes[1].axvline(similarities.mean(), color='red', linestyle='--', 
                           label=f'Mean: {similarities.mean():.3f}')
            axes[1].legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'tfidf_similarity.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\n✓ Saved: tfidf_similarity.png")
            
            # Save metrics
            with open(os.path.join(self.output_dir, 'tfidf_metrics.txt'), 'w') as f:
                f.write("TF-IDF + COSINE SIMILARITY METRICS\n")
                f.write("="*60 + "\n\n")
                f.write(f"Number of documents: {tfidf_matrix.shape[0]}\n")
                f.write(f"Vocabulary size: {tfidf_matrix.shape[1]}\n")
                f.write(f"Sparsity: {1 - (tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])):.4f}\n")
                f.write(f"\nSimilarity statistics (sample of {sample_size} docs):\n")
                f.write(f"  Mean: {similarity_matrix.mean():.4f}\n")
                f.write(f"  Std: {similarity_matrix.std():.4f}\n")
                f.write(f"  Min: {similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)].min():.4f}\n")
                f.write(f"  Max: {similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)].max():.4f}\n")
            
            print("✓ Method 1 completed successfully\n")
            return True
            
        except Exception as e:
            print(f"✗ Error in TF-IDF analysis: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ========================================================================
    # METHOD 2: LDA TOPIC MODELING
    # ========================================================================
    
    def lda_topic_modeling(self, n_topics=10):
        """Topic Models: LDA (Latent Dirichlet Allocation)"""
        print("\n" + "="*60)
        print("METHOD 2: LDA TOPIC MODELING")
        print("="*60)
        
        try:
            print("Cleaning texts...")
            cleaned_texts = [self.clean_text(t) for t in self.texts]
            
            # Create document-term matrix
            print(f"Creating document-term matrix for {n_topics} topics...")
            vectorizer = CountVectorizer(
                max_features=1000,
                min_df=5,
                max_df=0.8,
                stop_words='english'
            )
            doc_term_matrix = vectorizer.fit_transform(cleaned_texts)
            
            print(f"✓ Document-term matrix shape: {doc_term_matrix.shape}")
            
            # Fit LDA model
            print("Fitting LDA model...")
            lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=20,
                learning_method='online',
                n_jobs=-1,
                verbose=0
            )
            lda_output = lda_model.fit_transform(doc_term_matrix)
            
            print(f"✓ LDA training complete")
            print(f"  - Perplexity: {lda_model.perplexity(doc_term_matrix):.2f}")
            print(f"  - Log-likelihood: {lda_model.score(doc_term_matrix):.2f}")
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Display topics
            print(f"\nTop 10 words per topic:")
            topics_words = []
            for topic_idx, topic in enumerate(lda_model.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics_words.append(top_words)
                print(f"\nTopic {topic_idx + 1}:")
                print(f"  {', '.join(top_words)}")
            
            # Document-topic distribution
            print("\nDocument-topic distribution statistics:")
            print(f"  Shape: {lda_output.shape}")
            print(f"  Mean topic weight per doc: {lda_output.mean(axis=0)}")
            
            # Dominant topic per document
            dominant_topics = lda_output.argmax(axis=1)
            topic_counts = Counter(dominant_topics)
            print("\nDominant topic distribution:")
            for topic_id, count in sorted(topic_counts.items()):
                print(f"  Topic {topic_id + 1}: {count} documents ({count/len(dominant_topics)*100:.1f}%)")
            
            # Visualizations
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Topic distribution
            topic_dist = pd.Series(dominant_topics).value_counts().sort_index()
            axes[0, 0].bar(range(len(topic_dist)), topic_dist.values, color='teal')
            axes[0, 0].set_xlabel('Topic')
            axes[0, 0].set_ylabel('Number of Documents')
            axes[0, 0].set_title('Dominant Topic Distribution')
            axes[0, 0].set_xticks(range(n_topics))
            axes[0, 0].set_xticklabels([f'T{i+1}' for i in range(n_topics)])
            
            # Topic coherence (average topic weight)
            avg_topic_weights = lda_output.mean(axis=0)
            axes[0, 1].bar(range(len(avg_topic_weights)), avg_topic_weights, color='coral')
            axes[0, 1].set_xlabel('Topic')
            axes[0, 1].set_ylabel('Average Weight')
            axes[0, 1].set_title('Average Topic Weights Across Documents')
            axes[0, 1].set_xticks(range(n_topics))
            axes[0, 1].set_xticklabels([f'T{i+1}' for i in range(n_topics)])
            
            # Document-topic heatmap (sample)
            sample_size = min(50, lda_output.shape[0])
            sns.heatmap(lda_output[:sample_size].T, cmap='YlOrRd', ax=axes[1, 0], cbar=True)
            axes[1, 0].set_xlabel('Document')
            axes[1, 0].set_ylabel('Topic')
            axes[1, 0].set_title(f'Document-Topic Distribution (first {sample_size} docs)')
            
            # Topic word cloud style (top words per topic)
            topic_words_str = [f"T{i+1}: {', '.join(words[:5])}" for i, words in enumerate(topics_words)]
            axes[1, 1].axis('off')
            axes[1, 1].text(0.1, 0.9, "Top 5 Words per Topic:", fontsize=12, weight='bold', 
                           transform=axes[1, 1].transAxes)
            for i, text in enumerate(topic_words_str):
                axes[1, 1].text(0.1, 0.85 - i*0.08, text, fontsize=9, 
                               transform=axes[1, 1].transAxes, family='monospace')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'lda_topics.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\n✓ Saved: lda_topics.png")
            
            # Save metrics and topics
            with open(os.path.join(self.output_dir, 'lda_metrics.txt'), 'w') as f:
                f.write("LDA TOPIC MODELING METRICS\n")
                f.write("="*60 + "\n\n")
                f.write(f"Number of topics: {n_topics}\n")
                f.write(f"Number of documents: {doc_term_matrix.shape[0]}\n")
                f.write(f"Vocabulary size: {doc_term_matrix.shape[1]}\n")
                f.write(f"Perplexity: {lda_model.perplexity(doc_term_matrix):.2f}\n")
                f.write(f"Log-likelihood: {lda_model.score(doc_term_matrix):.2f}\n\n")
                
                f.write("Topics (top 10 words each):\n")
                for i, words in enumerate(topics_words):
                    f.write(f"\nTopic {i+1}:\n")
                    f.write(f"  {', '.join(words)}\n")
            
            print("✓ Method 2 completed successfully\n")
            return True
            
        except Exception as e:
            print(f"✗ Error in LDA analysis: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ========================================================================
    # METHOD 3: WORD EMBEDDING AVERAGING (Word2Vec)
    # ========================================================================
    
    def word_embedding_analysis(self, vector_size=100, window=5, min_count=2):
        """Word Embedding Averaging using Word2Vec"""
        print("\n" + "="*60)
        print("METHOD 3: WORD EMBEDDING AVERAGING (Word2Vec)")
        print("="*60)
        
        try:
            print("Preprocessing texts for Word2Vec...")
            # Tokenize texts
            tokenized_texts = [simple_preprocess(text, deacc=True) for text in self.texts]
            
            # Train Word2Vec model
            print(f"Training Word2Vec model (vector_size={vector_size}, window={window})...")
            w2v_model = Word2Vec(
                sentences=tokenized_texts,
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                workers=4,
                epochs=10,
                sg=1  # Skip-gram
            )
            
            print(f"✓ Word2Vec model trained")
            print(f"  - Vocabulary size: {len(w2v_model.wv)}")
            print(f"  - Vector dimensions: {vector_size}")
            
            # Show some word similarities
            print("\nExample word similarities:")
            test_words = ['software', 'manager', 'data', 'python', 'team']
            for word in test_words:
                if word in w2v_model.wv:
                    similar = w2v_model.wv.most_similar(word, topn=5)
                    print(f"\n  '{word}' most similar to:")
                    for sim_word, score in similar:
                        print(f"    - {sim_word}: {score:.4f}")
            
            # Create document embeddings by averaging word vectors
            print("\nCreating document embeddings...")
            doc_embeddings = []
            valid_docs = 0
            
            for tokens in tokenized_texts:
                # Get vectors for words in vocabulary
                word_vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
                
                if word_vectors:
                    # Average word vectors
                    doc_vector = np.mean(word_vectors, axis=0)
                    doc_embeddings.append(doc_vector)
                    valid_docs += 1
                else:
                    # If no words in vocabulary, use zero vector
                    doc_embeddings.append(np.zeros(vector_size))
            
            doc_embeddings = np.array(doc_embeddings)
            
            print(f"✓ Document embeddings created")
            print(f"  - Shape: {doc_embeddings.shape}")
            print(f"  - Valid documents: {valid_docs}/{len(tokenized_texts)}")
            
            # Compute cosine similarities between documents
            print("\nComputing document similarities...")
            sample_size = min(100, doc_embeddings.shape[0])
            sample_embeddings = doc_embeddings[:sample_size]
            similarity_matrix = cosine_similarity(sample_embeddings)
            
            print(f"✓ Similarity matrix shape: {similarity_matrix.shape}")
            triu_indices = np.triu_indices_from(similarity_matrix, k=1)
            similarities = similarity_matrix[triu_indices]
            print(f"  - Mean similarity: {similarities.mean():.4f}")
            print(f"  - Std similarity: {similarities.std():.4f}")
            
            # Find most similar documents
            print("\nTop 5 most similar document pairs:")
            top_pairs = similarities.argsort()[-5:][::-1]
            
            for rank, pair_idx in enumerate(top_pairs, 1):
                i, j = triu_indices[0][pair_idx], triu_indices[1][pair_idx]
                sim = similarity_matrix[i, j]
                print(f"\n  {rank}. Similarity: {sim:.4f}")
                print(f"     Doc {i}: {self.texts[i][:100]}...")
                print(f"     Doc {j}: {self.texts[j][:100]}...")
            
            # Visualizations
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Similarity heatmap
            sns.heatmap(similarity_matrix[:50, :50], cmap='YlGnBu', ax=axes[0], cbar=True)
            axes[0].set_title('Document Similarity Heatmap (Word2Vec)\n(first 50 docs)')
            axes[0].set_xlabel('Document')
            axes[0].set_ylabel('Document')
            
            # Similarity distribution
            axes[1].hist(similarities, bins=50, color='purple', edgecolor='black', alpha=0.7)
            axes[1].set_xlabel('Cosine Similarity')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Distribution of Document Similarities (Word2Vec)')
            axes[1].axvline(similarities.mean(), color='red', linestyle='--',
                           label=f'Mean: {similarities.mean():.3f}')
            axes[1].legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'word2vec_similarity.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\n✓ Saved: word2vec_similarity.png")
            
            # Save metrics
            with open(os.path.join(self.output_dir, 'word2vec_metrics.txt'), 'w') as f:
                f.write("WORD2VEC EMBEDDING METRICS\n")
                f.write("="*60 + "\n\n")
                f.write(f"Vector size: {vector_size}\n")
                f.write(f"Window size: {window}\n")
                f.write(f"Vocabulary size: {len(w2v_model.wv)}\n")
                f.write(f"Number of documents: {doc_embeddings.shape[0]}\n")
                f.write(f"Valid documents with embeddings: {valid_docs}\n")
                f.write(f"\nSimilarity statistics (sample of {sample_size} docs):\n")
                f.write(f"  Mean: {similarities.mean():.4f}\n")
                f.write(f"  Std: {similarities.std():.4f}\n")
                f.write(f"  Min: {similarities.min():.4f}\n")
                f.write(f"  Max: {similarities.max():.4f}\n")
            
            # Save model
            model_path = os.path.join(self.output_dir, 'word2vec.model')
            w2v_model.save(model_path)
            print(f"✓ Saved Word2Vec model: word2vec.model")
            
            print("✓ Method 3 completed successfully\n")
            return True
            
        except Exception as e:
            print(f"✗ Error in Word2Vec analysis: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_methods(self, n_topics=10):
        """Run all baseline methods"""
        if not self.load_data():
            return False
        
        print("Running all baseline methods...\n")
        
        success = True
        success &= self.tfidf_analysis()
        success &= self.lda_topic_modeling(n_topics=n_topics)
        success &= self.word_embedding_analysis()
        
        if success:
            print("\n" + "="*60)
            print("✓ ALL BASELINE METHODS COMPLETED SUCCESSFULLY")
            print("="*60)
            print(f"\nAll outputs saved to: {os.path.abspath(self.output_dir)}")
        
        return success


def main():
    parser = argparse.ArgumentParser(
        description='Baseline Methods for Job Description Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--file', '-f',
        default='data_cleaned.json',
        help='Path to JSON data file (default: data_cleaned.json)'
    )
    parser.add_argument(
        '--outdir', '-o',
        default='outputs/baseline',
        help='Output directory (default: outputs/baseline)'
    )
    parser.add_argument(
        '--sample', '-s',
        type=int,
        default=None,
        help='Sample size (default: use all data)'
    )
    parser.add_argument(
        '--topics', '-t',
        type=int,
        default=10,
        help='Number of LDA topics (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Run baseline methods
    baseline = BaselineMethods(args.file, args.outdir, args.sample)
    success = baseline.run_all_methods(n_topics=args.topics)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()