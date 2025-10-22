import os
from typing import Dict, List, Tuple
from pyserini.search.lucene import LuceneSearcher
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from tqdm import tqdm


class BERTReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.model_name = model_name
        self.max_length = 512  # BERT's max input length

    def chunk_document(self, text: str, max_tokens: int = 450) -> List[str]:
        """Split long documents into smaller chunks for BERT scoring."""
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
        """Compute relevance score between query and document."""
        chunks = self.chunk_document(document)
        if len(chunks) == 1:
            inputs = self.tokenizer(
                query,
                document,
                return_tensors='pt',
                max_length=self.max_length,
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                score = outputs.logits[0][0].item()
            return score
        else:
            # MAX aggregation for multiple chunks
            chunk_scores = []
            for chunk in chunks:
                inputs = self.tokenizer(
                    query,
                    chunk,
                    return_tensors='pt',
                    max_length=self.max_length,
                    truncation=True,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    score = outputs.logits[0][0].item()
                    chunk_scores.append(score)
            return max(chunk_scores)

    def rerank(self, query: str, candidates: List[Tuple[str, str, float]]) -> List[Tuple[str, float]]:
        """Rerank a list of BM25 candidate documents."""
        reranked = []
        for doc_id, doc_text, _ in tqdm(candidates, desc="Reranking", leave=False):
            score = self.score_query_doc_pair(query, doc_text)
            reranked.append((doc_id, score))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked


def load_queries(query_path: str) -> Dict[str, str]:
    """Load queries from a TSV file."""
    queries = {}
    with open(query_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                queries[parts[0]] = parts[1]
    return queries


def retrieve_bm25(searcher: LuceneSearcher, queries: Dict[str, str], k: int) -> Dict[str, List[Tuple[str, str, float]]]:
    """Retrieve top-k BM25 candidates for each query."""
    results = {}
    print(f"Retrieving top-{k} candidates using BM25...")
    for qid, qtext in tqdm(queries.items(), desc="BM25 Retrieval"):
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
    """Write results in TREC format."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for qid, doc_scores in results.items():
            for rank, (doc_id, score) in enumerate(doc_scores[:10], start=1):
                f.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score}\t{run_name}\n")


def task1_rerank(
    query_path: str,
    bm25_output_file: str,
    reranked_output_file: str,
    k: int
):
    """
    Task 1: Retrieve and Rerank using BERT cross-encoder.
    """
    print("=" * 80)
    print("TASK 1: Retrieve and Rerank Framework")
    print("=" * 80)

    # Step 1: Load queries
    print("\n[1/4] Loading queries...")
    queries = load_queries(query_path)
    print(f"Loaded {len(queries)} queries")

    # Step 2: Initialize BM25 searcher
    print("\n[2/4] Initializing BM25 searcher...")
    searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
    print("BM25 searcher ready")

    # Step 3: Retrieve top-k documents
    print(f"\n[3/4] Retrieving top-{k} candidates...")
    bm25_results = retrieve_bm25(searcher, queries, k)

    # Write BM25 results
    bm25_output_dict = {qid: [(doc_id, score) for doc_id, _, score in candidates]
                        for qid, candidates in bm25_results.items()}
    write_trec_output(bm25_output_dict, bm25_output_file, run_name="bm25")
    print(f"BM25 results saved at: {bm25_output_file}")

    # Step 4: Rerank using BERT Cross-Encoder
    print("\n[4/4] Reranking with BERT Cross-Encoder...")
    reranker = BERTReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    reranked_results = {}
    for qid, qtext in tqdm(queries.items(), desc="Processing queries"):
        reranked = reranker.rerank(qtext, bm25_results[qid])
        reranked_results[qid] = reranked

    # Write reranked results
    write_trec_output(reranked_results, reranked_output_file, run_name="bert_rerank")
    print(f"Reranked results saved at: {reranked_output_file}")

    print("\n" + "=" * 80)
    print("Task 1 completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Task 1: Retrieve and Rerank using BERT")
    parser.add_argument("--query_file_path", type=str, required=True)
    parser.add_argument("--task1_bm25_output_file", type=str, required=True)
    parser.add_argument("--reranked_output_file", type=str, required=True)
    parser.add_argument("--k", type=int, required=True)

    args = parser.parse_args()

    task1_rerank(
        query_path=args.query_file_path,
        bm25_output_file=args.task1_bm25_output_file,
        reranked_output_file=args.reranked_output_file,
        k=args.k,
    )
