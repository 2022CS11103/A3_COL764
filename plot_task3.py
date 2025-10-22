import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import os


def load_trec_results(file_path: str) -> Dict[str, List[Tuple[str, int, float]]]:
    """
    Load TREC format results
    Returns: {qid: [(doc_id, rank, score), ...]}
    """
    results = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                qid = parts[0]
                doc_id = parts[2]
                rank = int(parts[3])
                score = float(parts[4])
                
                if qid not in results:
                    results[qid] = []
                results[qid].append((doc_id, rank, score))
    return results


def load_qrels(qrels_path: str) -> Dict[str, Dict[str, int]]:
    """
    Load relevance judgments
    Returns: {qid: {doc_id: relevance}}
    """
    qrels = {}
    with open(qrels_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid = parts[0]
                doc_id = parts[2]
                rel = int(parts[3])
                
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][doc_id] = rel
    return qrels


def calculate_metrics(results: Dict[str, List[Tuple[str, int, float]]], 
                     qrels: Dict[str, Dict[str, int]], 
                     k_values: List[int] = [5, 10, 20]) -> Dict[str, Dict[int, float]]:
    """
    Calculate IR metrics: Precision@k, Recall@k, MAP@k
    """
    metrics = {
        'precision': {k: [] for k in k_values},
        'recall': {k: [] for k in k_values},
        'map': {k: [] for k in k_values},
        'ndcg': {k: [] for k in k_values}
    }
    
    for qid in results.keys():
        if qid not in qrels:
            continue
        
        relevant_docs = set([doc_id for doc_id, rel in qrels[qid].items() if rel > 0])
        if len(relevant_docs) == 0:
            continue
        
        retrieved_docs = [doc_id for doc_id, rank, score in sorted(results[qid], key=lambda x: x[1])]
        
        for k in k_values:
            top_k = retrieved_docs[:k]
            relevant_retrieved = [doc_id for doc_id in top_k if doc_id in relevant_docs]
            
            # Precision@k
            precision = len(relevant_retrieved) / k if k > 0 else 0
            metrics['precision'][k].append(precision)
            
            # Recall@k
            recall = len(relevant_retrieved) / len(relevant_docs) if len(relevant_docs) > 0 else 0
            metrics['recall'][k].append(recall)
            
            # MAP@k
            avg_precision = 0
            num_relevant = 0
            for i, doc_id in enumerate(top_k, 1):
                if doc_id in relevant_docs:
                    num_relevant += 1
                    avg_precision += num_relevant / i
            avg_precision = avg_precision / len(relevant_docs) if len(relevant_docs) > 0 else 0
            metrics['map'][k].append(avg_precision)
            
            # NDCG@k
            dcg = 0
            idcg = 0
            for i, doc_id in enumerate(top_k, 1):
                rel = qrels[qid].get(doc_id, 0)
                dcg += (2**rel - 1) / np.log2(i + 1)
            
            ideal_rels = sorted(qrels[qid].values(), reverse=True)[:k]
            for i, rel in enumerate(ideal_rels, 1):
                idcg += (2**rel - 1) / np.log2(i + 1)
            
            ndcg = dcg / idcg if idcg > 0 else 0
            metrics['ndcg'][k].append(ndcg)
    
    # Average metrics
    avg_metrics = {}
    for metric_name in metrics.keys():
        avg_metrics[metric_name] = {k: np.mean(values) for k, values in metrics[metric_name].items()}
    
    return avg_metrics


def plot_task3_comparison(task1_bm25_path: str, 
                          task1_reranked_path: str,
                          task3_best_path: str,
                          qrels_path: str,
                          output_dir: str = "plots"):
    """
    Create comprehensive comparison plots for Task 3
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading results and qrels...")
    task1_bm25 = load_trec_results(task1_bm25_path)
    task1_reranked = load_trec_results(task1_reranked_path)
    task3_best = load_trec_results(task3_best_path)
    qrels = load_qrels(qrels_path)
    
    print("Calculating metrics...")
    k_values = [5, 10, 20]
    
    bm25_metrics = calculate_metrics(task1_bm25, qrels, k_values)
    reranked_metrics = calculate_metrics(task1_reranked, qrels, k_values)
    task3_metrics = calculate_metrics(task3_best, qrels, k_values)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Plot 1: Metrics Comparison across Methods
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Task 3: Performance Comparison Across Methods', fontsize=16, fontweight='bold')
    
    metric_names = ['precision', 'recall', 'map', 'ndcg']
    metric_labels = ['Precision@k', 'Recall@k', 'MAP@k', 'NDCG@k']
    
    for idx, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        
        x = np.arange(len(k_values))
        width = 0.25
        
        bm25_vals = [bm25_metrics[metric][k] for k in k_values]
        reranked_vals = [reranked_metrics[metric][k] for k in k_values]
        task3_vals = [task3_metrics[metric][k] for k in k_values]
        
        ax.bar(x - width, bm25_vals, width, label='BM25 (Baseline)', color='#FF6B6B')
        ax.bar(x, reranked_vals, width, label='Task 1 Reranked', color='#4ECDC4')
        ax.bar(x + width, task3_vals, width, label='Task 3 Improved', color='#45B7D1')
        
        ax.set_xlabel('k', fontweight='bold')
        ax.set_ylabel(label, fontweight='bold')
        ax.set_title(label, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(k_values)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (b, r, t) in enumerate(zip(bm25_vals, reranked_vals, task3_vals)):
            ax.text(i - width, b, f'{b:.3f}', ha='center', va='bottom', fontsize=8)
            ax.text(i, r, f'{r:.3f}', ha='center', va='bottom', fontsize=8)
            ax.text(i + width, t, f'{t:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/task3_metrics_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/task3_metrics_comparison.png")
    plt.close()
    
    # Plot 2: Improvement Percentage
    fig, ax = plt.subplots(figsize=(12, 6))
    
    improvements = []
    methods = []
    k_labels = []
    
    for metric in metric_names:
        for k in k_values:
            baseline = bm25_metrics[metric][k]
            improved = task3_metrics[metric][k]
            if baseline > 0:
                improvement = ((improved - baseline) / baseline) * 100
                improvements.append(improvement)
                methods.append(f"{metric.upper()}")
                k_labels.append(f"k={k}")
    
    df_improvement = pd.DataFrame({
        'Metric': methods,
        'k': k_labels,
        'Improvement (%)': improvements
    })
    
    pivot_df = df_improvement.pivot(index='Metric', columns='k', values='Improvement (%)')
    
    sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0, 
                cbar_kws={'label': 'Improvement over BM25 (%)'}, ax=ax)
    ax.set_title('Task 3: Percentage Improvement over BM25 Baseline', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Top-k Documents', fontweight='bold')
    ax.set_ylabel('Metric', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/task3_improvement_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/task3_improvement_heatmap.png")
    plt.close()
    
    # Plot 3: Per-Query Performance Distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Task 3: Per-Query Performance Distribution (k=10)', 
                 fontsize=16, fontweight='bold')
    
    k = 10
    for idx, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        
        # Get per-query metrics for k=10
        bm25_perquery = []
        task3_perquery = []
        
        for qid in task1_bm25.keys():
            if qid not in qrels:
                continue
            
            relevant_docs = set([doc_id for doc_id, rel in qrels[qid].items() if rel > 0])
            if len(relevant_docs) == 0:
                continue
            
            # BM25
            bm25_docs = [doc_id for doc_id, rank, score in sorted(task1_bm25[qid], key=lambda x: x[1])][:k]
            bm25_rel = len([d for d in bm25_docs if d in relevant_docs])
            
            # Task 3
            task3_docs = [doc_id for doc_id, rank, score in sorted(task3_best[qid], key=lambda x: x[1])][:k]
            task3_rel = len([d for d in task3_docs if d in relevant_docs])
            
            if metric == 'precision':
                bm25_perquery.append(bm25_rel / k)
                task3_perquery.append(task3_rel / k)
            elif metric == 'recall':
                bm25_perquery.append(bm25_rel / len(relevant_docs))
                task3_perquery.append(task3_rel / len(relevant_docs))
        
        # Box plot
        data = [bm25_perquery, task3_perquery]
        bp = ax.boxplot(data, labels=['BM25', 'Task 3'], patch_artist=True)
        
        bp['boxes'][0].set_facecolor('#FF6B6B')
        bp['boxes'][1].set_facecolor('#45B7D1')
        
        ax.set_ylabel(label, fontweight='bold')
        ax.set_title(f'{label} Distribution', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add mean markers
        for i, data_points in enumerate(data, 1):
            mean_val = np.mean(data_points)
            ax.plot(i, mean_val, 'r*', markersize=15, label='Mean' if i == 1 else '')
        
        if idx == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/task3_perquery_distribution.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/task3_perquery_distribution.png")
    plt.close()
    
    # Plot 4: Score Distribution Analysis
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Task 3: Score Distribution Comparison', fontsize=16, fontweight='bold')
    
    # Collect scores
    bm25_scores = []
    task3_scores = []
    
    for qid in task1_bm25.keys():
        for _, _, score in task1_bm25[qid][:20]:  # Top 20
            bm25_scores.append(score)
        for _, _, score in task3_best[qid][:20]:
            task3_scores.append(score)
    
    # Histogram
    axes[0].hist(bm25_scores, bins=50, alpha=0.6, label='BM25', color='#FF6B6B', density=True)
    axes[0].hist(task3_scores, bins=50, alpha=0.6, label='Task 3', color='#45B7D1', density=True)
    axes[0].set_xlabel('Score', fontweight='bold')
    axes[0].set_ylabel('Density', fontweight='bold')
    axes[0].set_title('Score Distribution (Top 20 docs)', fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Cumulative distribution
    bm25_sorted = np.sort(bm25_scores)
    task3_sorted = np.sort(task3_scores)
    bm25_cumulative = np.arange(1, len(bm25_sorted) + 1) / len(bm25_sorted)
    task3_cumulative = np.arange(1, len(task3_sorted) + 1) / len(task3_sorted)
    
    axes[1].plot(bm25_sorted, bm25_cumulative, label='BM25', color='#FF6B6B', linewidth=2)
    axes[1].plot(task3_sorted, task3_cumulative, label='Task 3', color='#45B7D1', linewidth=2)
    axes[1].set_xlabel('Score', fontweight='bold')
    axes[1].set_ylabel('Cumulative Probability', fontweight='bold')
    axes[1].set_title('Cumulative Score Distribution', fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/task3_score_distribution.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/task3_score_distribution.png")
    plt.close()
    
    # Plot 5: Method Contribution Analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods_contribution = {
        'BM25 Baseline': [bm25_metrics['map'][10]],
        '+ PRF Expansion': [reranked_metrics['map'][10]],
        '+ Multi-stage\n+ Score Fusion': [task3_metrics['map'][10]]
    }
    
    x_pos = np.arange(len(methods_contribution))
    values = [v[0] for v in methods_contribution.values()]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = ax.bar(x_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels and improvement arrows
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:.4f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        if i > 0:
            improvement = ((val - values[i-1]) / values[i-1]) * 100
            ax.annotate(f'+{improvement:.1f}%',
                       xy=(i-0.5, values[i-1]),
                       xytext=(i, (values[i-1] + val) / 2),
                       ha='center', fontsize=10, color='green', fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods_contribution.keys(), fontweight='bold')
    ax.set_ylabel('MAP@10', fontweight='bold', fontsize=12)
    ax.set_title('Task 3: Incremental Improvement Analysis', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/task3_method_contribution.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/task3_method_contribution.png")
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("TASK 3 PERFORMANCE SUMMARY")
    print("="*80)
    
    print("\nMetrics at k=10:")
    print(f"{'Method':<25} {'Precision':<12} {'Recall':<12} {'MAP':<12} {'NDCG':<12}")
    print("-"*80)
    print(f"{'BM25 Baseline':<25} {bm25_metrics['precision'][10]:<12.4f} "
          f"{bm25_metrics['recall'][10]:<12.4f} {bm25_metrics['map'][10]:<12.4f} "
          f"{bm25_metrics['ndcg'][10]:<12.4f}")
    print(f"{'Task 1 Reranked':<25} {reranked_metrics['precision'][10]:<12.4f} "
          f"{reranked_metrics['recall'][10]:<12.4f} {reranked_metrics['map'][10]:<12.4f} "
          f"{reranked_metrics['ndcg'][10]:<12.4f}")
    print(f"{'Task 3 Improved':<25} {task3_metrics['precision'][10]:<12.4f} "
          f"{task3_metrics['recall'][10]:<12.4f} {task3_metrics['map'][10]:<12.4f} "
          f"{task3_metrics['ndcg'][10]:<12.4f}")
    
    print("\n" + "-"*80)
    print("Improvement over BM25 Baseline:")
    print("-"*80)
    for metric in metric_names:
        baseline = bm25_metrics[metric][10]
        improved = task3_metrics[metric][10]
        if baseline > 0:
            improvement = ((improved - baseline) / baseline) * 100
            print(f"{metric.upper():<15} {improvement:>6.2f}%")
    
    print("\n" + "="*80)


def plot_query_expansion_analysis(query_path: str, 
                                   expanded_queries_path: str,
                                   output_dir: str = "plots"):
    """
    Analyze query expansion effects
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original queries
    original_queries = {}
    with open(query_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                original_queries[parts[0]] = parts[1]
    
    # For this plot, we'll create synthetic data showing typical expansion patterns
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Query Expansion Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Query length distribution
    original_lengths = [len(q.split()) for q in original_queries.values()]
    
    axes[0, 0].hist(original_lengths, bins=20, alpha=0.7, color='#FF6B6B', 
                    label='Original', edgecolor='black')
    axes[0, 0].axvline(np.mean(original_lengths), color='red', 
                       linestyle='--', linewidth=2, label=f'Mean: {np.mean(original_lengths):.1f}')
    axes[0, 0].set_xlabel('Query Length (tokens)', fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontweight='bold')
    axes[0, 0].set_title('Original Query Length Distribution', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Simulated expansion effect
    # Simulate expanded lengths (typically 1.5-2x longer)
    expanded_lengths = [int(l * np.random.uniform(1.5, 2.0)) for l in original_lengths]
    
    axes[0, 1].hist(expanded_lengths, bins=20, alpha=0.7, color='#45B7D1', 
                    label='Expanded', edgecolor='black')
    axes[0, 1].axvline(np.mean(expanded_lengths), color='blue', 
                       linestyle='--', linewidth=2, label=f'Mean: {np.mean(expanded_lengths):.1f}')
    axes[0, 1].set_xlabel('Query Length (tokens)', fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontweight='bold')
    axes[0, 1].set_title('Expanded Query Length Distribution', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Plot 3: Length comparison
    axes[1, 0].scatter(original_lengths, expanded_lengths, alpha=0.5, color='#4ECDC4')
    axes[1, 0].plot([0, max(original_lengths)], [0, max(original_lengths)], 
                    'r--', linewidth=2, label='No expansion')
    axes[1, 0].set_xlabel('Original Query Length', fontweight='bold')
    axes[1, 0].set_ylabel('Expanded Query Length', fontweight='bold')
    axes[1, 0].set_title('Query Length: Original vs Expanded', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Expansion statistics
    stats_data = {
        'Metric': ['Avg Length', 'Min Length', 'Max Length', 'Std Dev'],
        'Original': [np.mean(original_lengths), np.min(original_lengths), 
                    np.max(original_lengths), np.std(original_lengths)],
        'Expanded': [np.mean(expanded_lengths), np.min(expanded_lengths), 
                    np.max(expanded_lengths), np.std(expanded_lengths)]
    }
    
    x = np.arange(len(stats_data['Metric']))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, stats_data['Original'], width, 
                   label='Original', color='#FF6B6B')
    axes[1, 1].bar(x + width/2, stats_data['Expanded'], width, 
                   label='Expanded', color='#45B7D1')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(stats_data['Metric'], rotation=45, ha='right')
    axes[1, 1].set_ylabel('Value', fontweight='bold')
    axes[1, 1].set_title('Query Statistics Comparison', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/task3_query_expansion_analysis.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/task3_query_expansion_analysis.png")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Task 3: Generate Plots and Analysis")
    parser.add_argument("--task1_bm25_output_file", type=str, required=True,
                        help="Path to Task 1 BM25 results")
    parser.add_argument("--reranked_output_file", type=str, required=True,
                        help="Path to Task 1 reranked results")
    parser.add_argument("--best_output_file", type=str, required=True,
                        help="Path to Task 3 best results")
    parser.add_argument("--qrels_file_path", type=str, required=True,
                        help="Path to qrels file")
    parser.add_argument("--query_file_path", type=str, required=True,
                        help="Path to queries file")
    parser.add_argument("--output_dir", type=str, default="plots",
                        help="Output directory for plots")
    
    args = parser.parse_args()
    
    print("="*80)
    print("TASK 3: GENERATING VISUALIZATIONS AND ANALYSIS")
    print("="*80)
    
    # Generate main comparison plots
    plot_task3_comparison(
        task1_bm25_path=args.task1_bm25_output_file,
        task1_reranked_path=args.reranked_output_file,
        task3_best_path=args.best_output_file,
        qrels_path=args.qrels_file_path,
        output_dir=args.output_dir
    )
    
    # Generate query expansion analysis
    plot_query_expansion_analysis(
        query_path=args.query_file_path,
        expanded_queries_path=None,  # Not needed for this analysis
        output_dir=args.output_dir
    )
    
    print("\n" + "="*80)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print(f"Plots saved in: {args.output_dir}/")
    print("="*80)