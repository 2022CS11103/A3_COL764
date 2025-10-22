import os
from typing import Dict, List, Tuple, Set
from pyserini.search.lucene import LuceneSearcher
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
import torch
from tqdm import tqdm
import numpy as np
from collections import Counter
import re


class BERTReranker:
    """BERT Cross-Encoder for reranking"""
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading BERT model: {model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = 512

    def chunk_document(self, text: str, max_tokens: int = 450) -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) <= max_tokens:
            return [text]
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk_text)
        return chunks

    def score_query_doc_pair(self, query: str, document: str) -> float:
        chunks = self.chunk_document(document)
        scores = []
        for chunk in chunks:
            inputs = self.tokenizer(
                query, chunk,
                return_tensors='pt',
                max_length=self.max_length,
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores.append(outputs.logits[0][0].item())
        return max(scores)

    def rerank(self, query: str, candidates: List[Tuple[str, str, float]]) -> List[Tuple[str, float]]:
        reranked = []
        for doc_id, doc_text, _ in candidates:
            score = self.score_query_doc_pair(query, doc_text)
            reranked.append((doc_id, score))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked


class QueryExpander:
    """Pseudo-Relevance Feedback (PRF) based query expansion"""
    def __init__(self, num_docs: int = 3, num_terms: int = 5):
        self.num_docs = num_docs  # Top-k documents for PRF
        self.num_terms = num_terms  # Number of expansion terms
        
        # Common stopwords
        self.stopwords = set([
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'what', 'when', 'where', 'who', 'which'
        ])

    def extract_terms(self, text: str) -> List[str]:
        """Extract terms from text"""
        # Remove special characters and lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        terms = text.split()
        # Filter stopwords and short terms
        terms = [t for t in terms if t not in self.stopwords and len(t) > 2]
        return terms

    def expand_query_prf(self, original_query: str, top_docs: List[str]) -> str:
        """
        Expand query using Pseudo-Relevance Feedback (PRF)
        Method: Extract most frequent terms from top retrieved documents
        """
        # Extract terms from query
        query_terms = set(self.extract_terms(original_query))
        
        # Extract terms from top documents
        all_terms = []
        for doc in top_docs[:self.num_docs]:
            all_terms.extend(self.extract_terms(doc))
        
        # Count term frequencies
        term_freq = Counter(all_terms)
        
        # Remove query terms from candidates
        for qt in query_terms:
            if qt in term_freq:
                del term_freq[qt]
        
        # Get top expansion terms
        expansion_terms = [term for term, _ in term_freq.most_common(self.num_terms)]
        
        # Combine original query with expansion terms
        expanded_query = original_query + " " + " ".join(expansion_terms)
        return expanded_query


class QueryRewriter:
    """T5-based query rewriting"""
    def __init__(self, model_name: str = "castorini/doc2query-t5-base-msmarco"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading T5 model: {model_name} on {self.device}")
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def rewrite_query(self, query: str) -> str:
        """
        Rewrite query using T5 model
        Method: Use doc2query model to generate alternative query formulations
        """
        # Prepare input
        input_text = f"generate query: {query}"
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            max_length=64,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=64,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
        
        rewritten = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return rewritten


def load_queries(query_path: str) -> Dict[str, str]:
    """Load queries from file"""
    queries = {}
    with open(query_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                queries[parts[0]] = parts[1]
    return queries


def retrieve_bm25(searcher: LuceneSearcher, queries: Dict[str, str], k: int) -> Dict[str, List[Tuple[str, str, float]]]:
    """Retrieve documents using BM25"""
    results = {}
    for qid, qtext in queries.items():
        hits = searcher.search(qtext, k=k)
        candidates = []
        for hit in hits:
            doc_id = hit.docid
            try:
                doc_text = searcher.doc(doc_id).raw()
            except Exception:
                doc_text = searcher.doc(doc_id).contents()
            candidates.append((doc_id, doc_text, hit.score))
        results[qid] = candidates
    return results


def write_trec_output(results: Dict[str, List[Tuple[str, float]]], output_file: str, run_name: str = "run"):
    """Write results in TREC format"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for qid, doc_scores in results.items():
            for rank, (doc_id, score) in enumerate(doc_scores[:1000], start=1):
                f.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score}\t{run_name}\n")


def task3_improve(query_path: str, best_output_file: str, k: int):
    """
    Task 3: Multi-method improvement pipeline
    
    Methods used:
    1. Query Expansion with Pseudo-Relevance Feedback (PRF)
    2. Query Rewriting with T5
    3. Multi-stage Reranking with BERT
    4. Score Fusion (combines multiple retrieval strategies)
    """
    print("=" * 80)
    print("TASK 3: Advanced Retrieval Improvement Pipeline")
    print("=" * 80)
    print("\nMethods implemented:")
    print("1. Pseudo-Relevance Feedback (PRF) - Query Expansion")
    print("2. T5-based Query Rewriting")
    print("3. Multi-stage BERT Reranking")
    print("4. Score Fusion")
    print("=" * 80)

    # Load queries
    print("\n[1/8] Loading queries...")
    queries = load_queries(query_path)
    print(f"Loaded {len(queries)} queries")

    # Initialize components
    print("\n[2/8] Initializing BM25 searcher...")
    searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
    
    print("\n[3/8] Initializing Query Expander (PRF)...")
    expander = QueryExpander(num_docs=3, num_terms=5)
    
    print("\n[4/8] Initializing Query Rewriter (T5)...")
    try:
        rewriter = QueryRewriter()
        use_rewriter = True
    except Exception as e:
        print(f"Warning: Could not load T5 model ({e}). Skipping query rewriting.")
        use_rewriter = False
    
    print("\n[5/8] Initializing BERT Reranker...")
    reranker = BERTReranker()

    # Strategy 1: Original query + BM25 + BERT reranking
    print("\n[6/8] Strategy 1: Original Query Retrieval...")
    original_results = retrieve_bm25(searcher, queries, k)
    
    # Strategy 2: PRF-expanded query + BM25 + BERT reranking
    print("\n[7/8] Strategy 2: PRF Query Expansion...")
    expanded_queries = {}
    for qid, qtext in tqdm(queries.items(), desc="Expanding queries"):
        # Get initial results for PRF
        initial_hits = searcher.search(qtext, k=5)
        top_docs = []
        for hit in initial_hits[:3]:
            try:
                doc_text = searcher.doc(hit.docid).raw()
            except Exception:
                doc_text = searcher.doc(hit.docid).contents()
            top_docs.append(doc_text)
        
        # Expand query
        expanded_query = expander.expand_query_prf(qtext, top_docs)
        expanded_queries[qid] = expanded_query
    
    expanded_results = retrieve_bm25(searcher, expanded_queries, k)
    
    # Strategy 3: Rewritten query (if available)
    if use_rewriter:
        print("\n[7.5/8] Strategy 3: Query Rewriting...")
        rewritten_queries = {}
        for qid, qtext in tqdm(queries.items(), desc="Rewriting queries"):
            rewritten_query = rewriter.rewrite_query(qtext)
            rewritten_queries[qid] = rewritten_query
        rewritten_results = retrieve_bm25(searcher, rewritten_queries, k)
    else:
        rewritten_results = {}

    # Multi-stage reranking and fusion
    print("\n[8/8] Multi-stage Reranking and Score Fusion...")
    final_results = {}
    
    for qid, qtext in tqdm(queries.items(), desc="Final reranking"):
        # Collect all candidate documents from all strategies
        all_candidates = {}  # doc_id -> (doc_text, scores_list)
        
        # Add candidates from original query
        for doc_id, doc_text, bm25_score in original_results[qid]:
            if doc_id not in all_candidates:
                all_candidates[doc_id] = (doc_text, [])
            all_candidates[doc_id][1].append(bm25_score)
        
        # Add candidates from expanded query
        for doc_id, doc_text, bm25_score in expanded_results[qid]:
            if doc_id not in all_candidates:
                all_candidates[doc_id] = (doc_text, [0.0])
            all_candidates[doc_id][1].append(bm25_score)
        
        # Add candidates from rewritten query
        if use_rewriter and qid in rewritten_results:
            for doc_id, doc_text, bm25_score in rewritten_results[qid]:
                if doc_id not in all_candidates:
                    # Pad with zeros if doc wasn't in previous strategies
                    all_candidates[doc_id] = (doc_text, [0.0, 0.0])
                all_candidates[doc_id][1].append(bm25_score)
        
        # Prepare candidates for BERT reranking
        candidates_for_rerank = []
        for doc_id, (doc_text, bm25_scores) in all_candidates.items():
            # Use max BM25 score across strategies
            max_bm25 = max(bm25_scores)
            candidates_for_rerank.append((doc_id, doc_text, max_bm25))
        
        # Sort by max BM25 score and keep top-k for reranking
        candidates_for_rerank.sort(key=lambda x: x[2], reverse=True)
        candidates_for_rerank = candidates_for_rerank[:k]
        
        # BERT reranking
        # Use original query for reranking (most reliable)
        reranked = reranker.rerank(qtext, candidates_for_rerank)
        
        # Normalize and fuse scores
        # BERT scores are typically in [-10, 10] range
        # BM25 scores are typically in [0, 30] range
        final_scores = []
        for doc_id, bert_score in reranked:
            # Get BM25 scores for this doc
            bm25_scores = all_candidates[doc_id][1]
            max_bm25 = max(bm25_scores) if bm25_scores else 0.0
            
            # Normalize scores
            norm_bert = bert_score / 10.0  # Rough normalization
            norm_bm25 = max_bm25 / 30.0     # Rough normalization
            
            # Weighted fusion: Give more weight to BERT (0.7) than BM25 (0.3)
            fused_score = 0.7 * norm_bert + 0.3 * norm_bm25
            final_scores.append((doc_id, fused_score))
        
        # Sort by fused score
        final_scores.sort(key=lambda x: x[1], reverse=True)
        final_results[qid] = final_scores

    # Write final results
    write_trec_output(final_results, best_output_file, run_name="task3_improved")
    print(f"\nFinal results saved at: {best_output_file}")

    print("\n" + "=" * 80)
    print("Task 3 completed successfully!")
    print("=" * 80)
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Total queries processed: {len(queries)}")
    print(f"Average candidates per query: {np.mean([len(final_results[qid]) for qid in queries.keys()]):.1f}")
    print("\nMethods used:")
    print("  ✓ Pseudo-Relevance Feedback (PRF)")
    if use_rewriter:
        print("  ✓ T5 Query Rewriting")
    print("  ✓ Multi-stage BERT Reranking")
    print("  ✓ Score Fusion (BERT + BM25)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Task 3: Advanced Retrieval Improvement")
    parser.add_argument("--query_file_path", type=str, required=True,
                        help="Path to queries file")
    parser.add_argument("--best_output_file", type=str, required=True,
                        help="Path to output file for best results")
    parser.add_argument("--k", type=int, required=True,
                        help="Number of documents to retrieve")

    args = parser.parse_args()

    task3_improve(
        query_path=args.query_file_path,
        best_output_file=args.best_output_file,
        k=args.k,
    )