import os
import json
import numpy as np
import pickle
import time
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pyserini.search.lucene import LuceneSearcher
import ir_datasets


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"  
DATASET_NAME = "msmarco-passage/trec-dl-hard"
K_VALUES = [10, 20, 50, 100]  
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")

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

sample_qid = list(queries.keys())[0]
print(f"\nSample query {sample_qid}: {queries[sample_qid][:80]}...")

rel_dist = defaultdict(int)
for qid in qrels:
    for doc_id, rel in qrels[qid].items():
        rel_dist[rel] += 1

print(f"\nRelevance distribution:")
for rel_val in sorted(rel_dist.keys()):
    print(f"  Relevance {rel_val}: {rel_dist[rel_val]} documents")


print("\n[STEP 2] Initializing BM25 retriever with MS-MARCO index...")

try:
    searcher = LuceneSearcher.from_prebuilt_index("msmarco-passage")
    print("✓ Successfully loaded pre-built MS-MARCO index")
except Exception as e:
    print(f"✗ Error loading index: {e}")
    print("Make sure you have internet connection and pyserini is properly installed")
    exit(1)


print(f"\n[STEP 3] Loading BERT model: {MODEL_NAME}...")
start_time = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model = model.to(DEVICE)
model.eval()

print(f"✓ Model loaded successfully in {time.time() - start_time:.2f}s")

def retrieve_topk(query_id, query_text, k):
    """Retrieve top-k documents using BM25"""
    try:
        hits = searcher.search(query_text, k=k)
        retrieved_docs = []
        for hit in hits:
            doc_id = hit.docid
            score = hit.score
            retrieved_docs.append({
                'doc_id': doc_id,
                'bm25_score': score,
                'rank': len(retrieved_docs) + 1
            })
        return retrieved_docs
    except Exception as e:
        print(f"Error retrieving for query {query_id}: {e}")
        return []

def rerank_with_bert(query_text, retrieved_docs, batch_size=BATCH_SIZE):
    """Rerank documents using BERT cross-encoder"""
    if not retrieved_docs:
        return retrieved_docs
    
    pairs = []
    for doc in retrieved_docs:
        try:
            doc_raw = searcher.doc(doc['doc_id']).raw()
            if '\n' in doc_raw:
                doc_text = doc_raw.split('\n', 1)[1]
            else:
                doc_text = doc_raw
            pairs.append((query_text, doc_text))
        except Exception as e:
            
            pairs.append((query_text, ""))
    
    scores = []

    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i + batch_size]
  
        inputs = tokenizer(
            [p[0] for p in batch_pairs],
            [p[1] for p in batch_pairs],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            if outputs.logits.shape[1] == 1:
                batch_scores = outputs.logits[:, 0].cpu().numpy()
            else:
                batch_scores = outputs.logits[:, 1].cpu().numpy()
        
        scores.extend(batch_scores)

    for doc, score in zip(retrieved_docs, scores):
        doc['bert_score'] = float(score)
    
    sorted_docs = sorted(retrieved_docs, key=lambda x: x['bert_score'], reverse=True)

    for i, doc in enumerate(sorted_docs[:10]):
        doc['final_rank'] = i + 1
    
    return sorted_docs[:10]

def compute_dcg(ranking, qrel_dict, k):
    """Compute DCG@k"""
    dcg = 0.0
    for i in range(min(k, len(ranking))):
        doc_id = ranking[i]['doc_id']
        relevance = qrel_dict.get(doc_id, 0)
        dcg += relevance / np.log2(i + 2)
    return dcg

def compute_idcg(qrel_dict, k):
    """Compute IDCG@k (ideal DCG)"""
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
    """Compute MRR@k (Mean Reciprocal Rank)"""
    for i in range(min(k, len(ranking))):
        doc_id = ranking[i]['doc_id']
        if qrel_dict.get(doc_id, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0

print("\n[STEP 7] Running retrieve and rerank evaluation...")

results = {}

for k in K_VALUES:
    print(f"\n{'='*70}")
    print(f"Evaluating with k={k}")
    print(f"{'='*70}")
    
    results[k] = {
        'NDCG@1': [],
        'NDCG@5': [],
        'NDCG@10': [],
        'MRR@10': []
    }
    
    query_count = 0
    start_time = time.time()
    
    for query_id in sorted(queries.keys()):
        query_text = queries[query_id]

        if query_id not in qrels:
            continue
        
        query_count += 1
        
        try:

            retrieved_docs = retrieve_topk(query_id, query_text, k)

            reranked_docs = rerank_with_bert(query_text, retrieved_docs)

            qrel_dict = qrels[query_id]
            
            results[k]['NDCG@1'].append(compute_ndcg(reranked_docs, qrel_dict, 1))
            results[k]['NDCG@5'].append(compute_ndcg(reranked_docs, qrel_dict, 5))
            results[k]['NDCG@10'].append(compute_ndcg(reranked_docs, qrel_dict, 10))
            results[k]['MRR@10'].append(compute_mrr(reranked_docs, qrel_dict, 10))
            
            if query_count % 5 == 0:
                elapsed = time.time() - start_time
                print(f"  Processed {query_count} queries in {elapsed:.2f}s...")
        
        except Exception as e:
            print(f"  Error processing query {query_id}: {e}")
            continue
    
    elapsed = time.time() - start_time
    print(f"\n✓ Completed {query_count} queries for k={k} in {elapsed:.2f}s")

print("\n" + "="*70)
print("RESULTS SUMMARY - RETRIEVE AND RERANK WITH BERT")
print("="*70)

output_data = {}

for k in K_VALUES:
    print(f"\nResults for k={k}:")
    print("-" * 70)
    
    for metric in ['NDCG@1', 'NDCG@5', 'NDCG@10', 'MRR@10']:
        values = results[k][metric]
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        print(f"{metric:12} : Mean={mean_val:.4f}  Std={std_val:.4f}")
        
        if k not in output_data:
            output_data[k] = {}
        output_data[k][metric] = {
            'mean': mean_val,
            'std': std_val,
            'values': values
        }

print("\n[STEP 9] Saving results to file...")

output_file = 'task1_results.json'
with open(output_file, 'w') as f:
    data_to_save = {}
    for k in output_data:
        data_to_save[str(k)] = {}
        for metric in output_data[k]:
            data_to_save[str(k)][metric] = {
                'mean': float(output_data[k][metric]['mean']),
                'std': float(output_data[k][metric]['std']),
                'count': len(output_data[k][metric]['values'])
            }
    json.dump(data_to_save, f, indent=2)

print(f"✓ Results saved to {output_file}")

print("\n" + "="*70)
print("Task 1 completed successfully!")
print("="*70)