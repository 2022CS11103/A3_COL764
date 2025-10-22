import os
import json
import numpy as np
import time
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from pyserini.search.lucene import LuceneSearcher
import ir_datasets
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configuration
DATASET_NAME = "msmarco-passage/trec-dl-hard"
K_VALUES = [10, 20, 50, 100]
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model configurations
MODELS = {
    'BERT_MiniLM': {
        'name': 'cross-encoder/ms-marco-MiniLM-L-12-v2',
        'type': 'cross-encoder',
        'description': 'MiniLM-L12 cross-encoder fine-tuned on MS-MARCO'
    },
    'BERT_TinyBERT': {
        'name': 'cross-encoder/ms-marco-TinyBERT-L-6',
        'type': 'cross-encoder',
        'description': 'TinyBERT-L6 cross-encoder (lighter alternative)'
    },
    'DistilBERT': {
        'name': 'cross-encoder/ms-marco-distilbert-base-v4',
        'type': 'cross-encoder',
        'description': 'DistilBERT cross-encoder for MS-MARCO'
    },
    'BiEncoder_MiniLM': {
        'name': 'sentence-transformers/msmarco-MiniLM-L-6-v3',
        'type': 'bi-encoder',
        'description': 'MiniLM-L6 bi-encoder (dual encoder architecture)'
    },
    'BiEncoder_MPNet': {
        'name': 'sentence-transformers/msmarco-distilbert-base-tas-b',
        'type': 'bi-encoder',
        'description': 'DistilBERT bi-encoder with TAS-Balanced training'
    }
}

print(f"Using device: {DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")

# ============================================================================
# STEP 1: Load Dataset
# ============================================================================
print("\n[STEP 1] Loading TREC-DL-Hard dataset...")
start_time = time.time()

dataset = ir_datasets.load(DATASET_NAME)
queries = {}
qrels = {}

for query in dataset.queries_iter():
    queries[query.query_id] = query.text

for qrel in dataset.qrels_iter():
    if qrel.query_id not in qrels:
        qrels[qrel.query_id] = {}
    qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

print(f"✓ Loaded {len(queries)} queries in {time.time() - start_time:.2f}s")
print(f"✓ Loaded qrels for {len(qrels)} queries")

# Print relevance distribution
rel_dist = defaultdict(int)
for qid in qrels:
    for doc_id, rel in qrels[qid].items():
        rel_dist[rel] += 1

print(f"\nRelevance distribution:")
for rel_val in sorted(rel_dist.keys()):
    print(f"  Relevance {rel_val}: {rel_dist[rel_val]} documents")

# ============================================================================
# STEP 2: Initialize BM25 Retriever
# ============================================================================
print("\n[STEP 2] Initializing BM25 retriever...")
try:
    searcher = LuceneSearcher.from_prebuilt_index("msmarco-passage")
    print("✓ Successfully loaded pre-built MS-MARCO index")
except Exception as e:
    print(f"✗ Error loading index: {e}")
    exit(1)

# ============================================================================
# STEP 3: Model Loading Functions
# ============================================================================

class ModelWrapper:
    """Wrapper class for different model types"""
    
    def __init__(self, model_config):
        self.config = model_config
        self.model_type = model_config['type']
        
        if self.model_type == 'cross-encoder':
            print(f"Loading cross-encoder: {model_config['name']}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
            self.model = AutoModelForSequenceClassification.from_pretrained(model_config['name'])
            self.model = self.model.to(DEVICE)
            self.model.eval()
        elif self.model_type == 'bi-encoder':
            print(f"Loading bi-encoder: {model_config['name']}")
            self.model = SentenceTransformer(model_config['name'], device=DEVICE)
            self.model.eval()
    
    def score_pairs(self, query_text, doc_texts, batch_size=BATCH_SIZE):
        """Score query-document pairs"""
        if self.model_type == 'cross-encoder':
            return self._score_cross_encoder(query_text, doc_texts, batch_size)
        elif self.model_type == 'bi-encoder':
            return self._score_bi_encoder(query_text, doc_texts, batch_size)
    
    def _score_cross_encoder(self, query_text, doc_texts, batch_size):
        """Score using cross-encoder (BERT-style)"""
        scores = []
        
        for i in range(0, len(doc_texts), batch_size):
            batch_docs = doc_texts[i:i + batch_size]
            batch_queries = [query_text] * len(batch_docs)
            
            inputs = self.tokenizer(
                batch_queries,
                batch_docs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                if outputs.logits.shape[1] == 1:
                    batch_scores = outputs.logits[:, 0].cpu().numpy()
                else:
                    batch_scores = outputs.logits[:, 1].cpu().numpy()
            
            scores.extend(batch_scores)
        
        return scores
    
    def _score_bi_encoder(self, query_text, doc_texts, batch_size):
        """Score using bi-encoder (dual encoder)"""
        # Encode query once
        query_embedding = self.model.encode(query_text, convert_to_tensor=True, show_progress_bar=False)
        
        # Encode documents in batches
        all_scores = []
        for i in range(0, len(doc_texts), batch_size):
            batch_docs = doc_texts[i:i + batch_size]
            doc_embeddings = self.model.encode(batch_docs, convert_to_tensor=True, show_progress_bar=False)
            
            # Compute cosine similarity
            scores = util.cos_sim(query_embedding, doc_embeddings)[0].cpu().numpy()
            all_scores.extend(scores)
        
        return all_scores

# ============================================================================
# STEP 4: Retrieval and Reranking Functions
# ============================================================================

def retrieve_topk(query_id, query_text, k):
    """Retrieve top-k documents using BM25"""
    try:
        hits = searcher.search(query_text, k=k)
        retrieved_docs = []
        for hit in hits:
            retrieved_docs.append({
                'doc_id': hit.docid,
                'bm25_score': hit.score,
                'rank': len(retrieved_docs) + 1
            })
        return retrieved_docs
    except Exception as e:
        print(f"Error retrieving for query {query_id}: {e}")
        return []

def get_document_texts(doc_ids):
    """Fetch document texts from the index"""
    doc_texts = []
    for doc_id in doc_ids:
        try:
            doc_raw = searcher.doc(doc_id).raw()
            if '\n' in doc_raw:
                doc_text = doc_raw.split('\n', 1)[1]
            else:
                doc_text = doc_raw
            doc_texts.append(doc_text)
        except Exception as e:
            doc_texts.append("")
    return doc_texts

def rerank_documents(model_wrapper, query_text, retrieved_docs):
    """Rerank documents using the given model"""
    if not retrieved_docs:
        return retrieved_docs
    
    doc_ids = [doc['doc_id'] for doc in retrieved_docs]
    doc_texts = get_document_texts(doc_ids)
    
    scores = model_wrapper.score_pairs(query_text, doc_texts)
    
    for doc, score in zip(retrieved_docs, scores):
        doc['rerank_score'] = float(score)
    
    sorted_docs = sorted(retrieved_docs, key=lambda x: x['rerank_score'], reverse=True)
    
    for i, doc in enumerate(sorted_docs[:10]):
        doc['final_rank'] = i + 1
    
    return sorted_docs[:10]

# ============================================================================
# STEP 5: Evaluation Metrics
# ============================================================================

def compute_dcg(ranking, qrel_dict, k):
    """Compute DCG@k"""
    dcg = 0.0
    for i in range(min(k, len(ranking))):
        doc_id = ranking[i]['doc_id']
        relevance = qrel_dict.get(doc_id, 0)
        dcg += relevance / np.log2(i + 2)
    return dcg

def compute_idcg(qrel_dict, k):
    """Compute IDCG@k"""
    rel_scores = sorted(qrel_dict.values(), reverse=True)
    idcg = 0.0
    for i in range(min(k, len(rel_scores))):
        idcg += rel_scores[i] / np.log2(i + 2)
    return idcg

def compute_ndcg(ranking, qrel_dict, k):
    """Compute NDCG@k"""
    dcg = compute_dcg(ranking, qrel_dict, k)
    idcg = compute_idcg(qrel_dict, k)
    return dcg / idcg if idcg > 0 else 0.0

def compute_mrr(ranking, qrel_dict, k):
    """Compute MRR@k"""
    for i in range(min(k, len(ranking))):
        doc_id = ranking[i]['doc_id']
        if qrel_dict.get(doc_id, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0

def compute_precision(ranking, qrel_dict, k):
    """Compute Precision@k"""
    relevant = 0
    for i in range(min(k, len(ranking))):
        doc_id = ranking[i]['doc_id']
        if qrel_dict.get(doc_id, 0) > 0:
            relevant += 1
    return relevant / k if k > 0 else 0.0

def compute_recall(ranking, qrel_dict, k):
    """Compute Recall@k"""
    total_relevant = sum(1 for rel in qrel_dict.values() if rel > 0)
    if total_relevant == 0:
        return 0.0
    
    relevant_retrieved = 0
    for i in range(min(k, len(ranking))):
        doc_id = ranking[i]['doc_id']
        if qrel_dict.get(doc_id, 0) > 0:
            relevant_retrieved += 1
    
    return relevant_retrieved / total_relevant

# ============================================================================
# STEP 6: Baseline - BM25 Only
# ============================================================================

def evaluate_bm25_baseline(k):
    """Evaluate BM25 baseline (no reranking)"""
    results = {
        'NDCG@1': [], 'NDCG@5': [], 'NDCG@10': [],
        'MRR@10': [], 'P@10': [], 'R@10': []
    }
    
    for query_id in tqdm(sorted(queries.keys()), desc=f"BM25 Baseline (k={k})"):
        if query_id not in qrels:
            continue
        
        query_text = queries[query_id]
        retrieved_docs = retrieve_topk(query_id, query_text, k)
        
        qrel_dict = qrels[query_id]
        results['NDCG@1'].append(compute_ndcg(retrieved_docs[:10], qrel_dict, 1))
        results['NDCG@5'].append(compute_ndcg(retrieved_docs[:10], qrel_dict, 5))
        results['NDCG@10'].append(compute_ndcg(retrieved_docs[:10], qrel_dict, 10))
        results['MRR@10'].append(compute_mrr(retrieved_docs[:10], qrel_dict, 10))
        results['P@10'].append(compute_precision(retrieved_docs[:10], qrel_dict, 10))
        results['R@10'].append(compute_recall(retrieved_docs[:10], qrel_dict, 10))
    
    return results

# ============================================================================
# STEP 7: Main Evaluation Loop
# ============================================================================

print("\n[STEP 3] Starting comprehensive evaluation...")

all_results = {}

# Evaluate BM25 baseline
print("\n" + "="*70)
print("EVALUATING BM25 BASELINE (NO RERANKING)")
print("="*70)

all_results['BM25_Baseline'] = {}
for k in K_VALUES:
    all_results['BM25_Baseline'][k] = evaluate_bm25_baseline(k)

# Evaluate each reranking model
for model_name, model_config in MODELS.items():
    print("\n" + "="*70)
    print(f"EVALUATING: {model_name}")
    print(f"Description: {model_config['description']}")
    print("="*70)
    
    try:
        model_wrapper = ModelWrapper(model_config)
        all_results[model_name] = {}
        
        for k in K_VALUES:
            print(f"\nEvaluating with k={k}...")
            results = {
                'NDCG@1': [], 'NDCG@5': [], 'NDCG@10': [],
                'MRR@10': [], 'P@10': [], 'R@10': []
            }
            
            start_time = time.time()
            
            for query_id in tqdm(sorted(queries.keys()), desc=f"Processing queries (k={k})"):
                if query_id not in qrels:
                    continue
                
                query_text = queries[query_id]
                
                try:
                    retrieved_docs = retrieve_topk(query_id, query_text, k)
                    reranked_docs = rerank_documents(model_wrapper, query_text, retrieved_docs)
                    
                    qrel_dict = qrels[query_id]
                    results['NDCG@1'].append(compute_ndcg(reranked_docs, qrel_dict, 1))
                    results['NDCG@5'].append(compute_ndcg(reranked_docs, qrel_dict, 5))
                    results['NDCG@10'].append(compute_ndcg(reranked_docs, qrel_dict, 10))
                    results['MRR@10'].append(compute_mrr(reranked_docs, qrel_dict, 10))
                    results['P@10'].append(compute_precision(reranked_docs, qrel_dict, 10))
                    results['R@10'].append(compute_recall(reranked_docs, qrel_dict, 10))
                except Exception as e:
                    print(f"Error processing query {query_id}: {e}")
                    continue
            
            elapsed = time.time() - start_time
            print(f"Completed in {elapsed:.2f}s")
            
            all_results[model_name][k] = results
        
        # Clean up
        del model_wrapper
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        continue

# ============================================================================
# STEP 8: Results Summary and Analysis
# ============================================================================

print("\n" + "="*70)
print("COMPREHENSIVE RESULTS SUMMARY")
print("="*70)

summary_data = {}

for model_name in all_results.keys():
    summary_data[model_name] = {}
    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"{'='*70}")
    
    for k in K_VALUES:
        print(f"\nResults for k={k}:")
        print("-" * 70)
        summary_data[model_name][k] = {}
        
        for metric in ['NDCG@1', 'NDCG@5', 'NDCG@10', 'MRR@10', 'P@10', 'R@10']:
            values = all_results[model_name][k][metric]
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            print(f"{metric:12} : Mean={mean_val:.4f}  Std={std_val:.4f}")
            
            summary_data[model_name][k][metric] = {
                'mean': float(mean_val),
                'std': float(std_val)
            }

# ============================================================================
# STEP 9: Save Results
# ============================================================================

print("\n[STEP 9] Saving results...")

with open('task2_results.json', 'w') as f:
    json.dump(summary_data, f, indent=2)

print("✓ Results saved to task2_results.json")

# ============================================================================
# STEP 10: Generate Comparison Plots
# ============================================================================

print("\n[STEP 10] Generating comparison plots...")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Create comparison plots for each k value
for k in K_VALUES:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Model Comparison for k={k}', fontsize=16, fontweight='bold')
    
    metrics = ['NDCG@1', 'NDCG@5', 'NDCG@10', 'MRR@10', 'P@10', 'R@10']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        model_names = list(summary_data.keys())
        means = [summary_data[model][k][metric]['mean'] for model in model_names]
        stds = [summary_data[model][k][metric]['std'] for model in model_names]
        
        bars = ax.bar(range(len(model_names)), means, yerr=stds, capsize=5, alpha=0.7)
        ax.set_ylabel(metric, fontweight='bold')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylim([0, max(means) * 1.2])
        
        # Add value labels on bars
        for i, (bar, mean) in enumerate(zip(bars, means)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'task2_comparison_k{k}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot for k={k}")

# Create line plots showing metric trends across k values
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Metric Trends Across Different k Values', fontsize=16, fontweight='bold')

metrics = ['NDCG@1', 'NDCG@5', 'NDCG@10', 'MRR@10', 'P@10', 'R@10']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]
    
    for model_name in summary_data.keys():
        k_vals = []
        metric_vals = []
        for k in K_VALUES:
            k_vals.append(k)
            metric_vals.append(summary_data[model_name][k][metric]['mean'])
        
        ax.plot(k_vals, metric_vals, marker='o', label=model_name, linewidth=2)
    
    ax.set_xlabel('k value', fontweight='bold')
    ax.set_ylabel(metric, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task2_trends.png', dpi=300, bbox_inches='tight')
print("✓ Saved trends plot")

# ============================================================================
# STEP 11: Generate Performance Comparison Table
# ============================================================================

print("\n[STEP 11] Generating performance comparison tables...")

# Best k value comparison
k_best = 100  # Using largest k for fair comparison

comparison_table = []
for model_name in summary_data.keys():
    row = {'Model': model_name}
    for metric in ['NDCG@10', 'MRR@10']:
        row[metric] = f"{summary_data[model_name][k_best][metric]['mean']:.4f}"
    comparison_table.append(row)

# Save as CSV
import csv
with open('task2_comparison_table.csv', 'w', newline='') as f:
    if comparison_table:
        writer = csv.DictWriter(f, fieldnames=comparison_table[0].keys())
        writer.writeheader()
        writer.writerows(comparison_table)

print("✓ Saved comparison table to task2_comparison_table.csv")

# ============================================================================
# STEP 12: Statistical Analysis
# ============================================================================

print("\n[STEP 12] Computing improvements over baseline...")

improvements = {}

for model_name in summary_data.keys():
    if model_name == 'BM25_Baseline':
        continue
    
    improvements[model_name] = {}
    
    for k in K_VALUES:
        improvements[model_name][k] = {}
        
        for metric in ['NDCG@10', 'MRR@10']:
            baseline_val = summary_data['BM25_Baseline'][k][metric]['mean']
            model_val = summary_data[model_name][k][metric]['mean']
            
            improvement = ((model_val - baseline_val) / baseline_val) * 100
            improvements[model_name][k][metric] = improvement

print("\nPercentage improvements over BM25 baseline:")
for model_name in improvements.keys():
    print(f"\n{model_name}:")
    for k in K_VALUES:
        print(f"  k={k}:")
        for metric in ['NDCG@10', 'MRR@10']:
            print(f"    {metric}: {improvements[model_name][k][metric]:+.2f}%")

# Save improvements
with open('task2_improvements.json', 'w') as f:
    json.dump(improvements, f, indent=2)

print("\n" + "="*70)
print("Task 2 completed successfully!")
print("="*70)
print("\nGenerated files:")
print("  - task2_results.json: Complete results for all models")
print("  - task2_comparison_k*.png: Comparison plots for each k value")
print("  - task2_trends.png: Metric trends across k values")
print("  - task2_comparison_table.csv: Summary comparison table")
print("  - task2_improvements.json: Improvements over baseline")
print("="*70)