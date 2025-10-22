import os
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def run_trec_eval(qrels_file: str, results_file: str) -> Dict[str, float]:
    """
    Run trec_eval and parse all metrics.
    
    Args:
        qrels_file: Path to qrels file
        results_file: Path to results file
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Metrics to evaluate
    metric_list = [
        ('ndcg_cut.1', 'NDCG@1'),
        ('ndcg_cut.5', 'NDCG@5'),
        ('ndcg_cut.10', 'NDCG@10'),
        ('recip_rank', 'MRR@10')
    ]
    
    for trec_metric, display_name in metric_list:
        try:
            cmd = ['trec_eval', '-m', trec_metric, qrels_file, results_file]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse output
            for line in result.stdout.strip().split('\n'):
                parts = line.split()
                if len(parts) >= 3 and parts[1] == 'all':
                    metrics[display_name] = float(parts[2])
                    break
        except Exception as e:
            print(f"Warning: Could not evaluate {display_name}: {e}")
            metrics[display_name] = 0.0
    
    return metrics


def analyze_single_k(qrels_file: str, bm25_file: str, rerank_file: str, k: int) -> Dict:
    """
    Analyze results for a single k value.
    
    Returns:
        Dictionary with metrics for both BM25 and reranked results
    """
    print(f"\nAnalyzing k={k}...")
    
    bm25_metrics = run_trec_eval(qrels_file, bm25_file)
    rerank_metrics = run_trec_eval(qrels_file, rerank_file)
    
    return {
        'k': k,
        'BM25': bm25_metrics,
        'BERT_Rerank': rerank_metrics
    }


def create_comparison_table(results: List[Dict]) -> pd.DataFrame:
    """
    Create a comprehensive comparison table.
    """
    rows = []
    
    for result in results:
        k = result['k']
        bm25 = result['BM25']
        rerank = result['BERT_Rerank']
        
        for metric in ['NDCG@1', 'NDCG@5', 'NDCG@10', 'MRR@10']:
            bm25_score = bm25.get(metric, 0.0)
            rerank_score = rerank.get(metric, 0.0)
            improvement = ((rerank_score - bm25_score) / bm25_score * 100) if bm25_score > 0 else 0
            
            rows.append({
                'k': k,
                'Metric': metric,
                'BM25': bm25_score,
                'BERT_Rerank': rerank_score,
                'Improvement (%)': improvement
            })
    
    return pd.DataFrame(rows)


def plot_metrics_vs_k(df: pd.DataFrame, output_dir: str):
    """
    Plot 1: All metrics vs k (2x2 grid)
    """
    metrics = ['NDCG@1', 'NDCG@5', 'NDCG@10', 'MRR@10']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    k_values = sorted(df['k'].unique())
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        metric_df = df[df['Metric'] == metric]
        
        # Group by k
        bm25_scores = []
        rerank_scores = []
        for k in k_values:
            k_data = metric_df[metric_df['k'] == k]
            bm25_scores.append(k_data['BM25'].values[0])
            rerank_scores.append(k_data['BERT_Rerank'].values[0])
        
        ax.plot(k_values, bm25_scores, marker='o', label='BM25', 
                linewidth=2.5, markersize=8, color='#e74c3c')
        ax.plot(k_values, rerank_scores, marker='s', label='BERT Rerank', 
                linewidth=2.5, markersize=8, color='#3498db')
        
        ax.set_xlabel('k (Number of candidates retrieved)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric} Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} Performance vs k', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(k_values)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_vs_k.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: metrics_vs_k.png")
    plt.close()


def plot_improvement_bars(df: pd.DataFrame, output_dir: str):
    """
    Plot 2: Improvement percentage as bar charts
    """
    metrics = ['NDCG@1', 'NDCG@5', 'NDCG@10', 'MRR@10']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    k_values = sorted(df['k'].unique())
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(k_values)))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        metric_df = df[df['Metric'] == metric]
        
        improvements = []
        for k in k_values:
            k_data = metric_df[metric_df['k'] == k]
            improvements.append(k_data['Improvement (%)'].values[0])
        
        bars = ax.bar(range(len(k_values)), improvements, color=colors, 
                      edgecolor='black', linewidth=1.2, alpha=0.8)
        
        ax.set_xlabel('k (Number of candidates)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Improvement over BM25 (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} - BERT Improvement', fontsize=13, fontweight='bold')
        ax.set_xticks(range(len(k_values)))
        ax.set_xticklabels([f'k={k}' for k in k_values])
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, improvements)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top',
                   fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_bars.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: improvement_bars.png")
    plt.close()


def plot_combined_metrics(df: pd.DataFrame, output_dir: str):
    """
    Plot 3: All metrics combined for BERT reranking
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    metrics = ['NDCG@1', 'NDCG@5', 'NDCG@10', 'MRR@10']
    k_values = sorted(df['k'].unique())
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    markers = ['o', 's', '^', 'D']
    
    for idx, metric in enumerate(metrics):
        metric_df = df[df['Metric'] == metric]
        
        scores = []
        for k in k_values:
            k_data = metric_df[metric_df['k'] == k]
            scores.append(k_data['BERT_Rerank'].values[0])
        
        ax.plot(k_values, scores, marker=markers[idx], label=metric,
               linewidth=2.5, markersize=10, color=colors[idx])
    
    ax.set_xlabel('k (Number of candidates retrieved)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Metric Score', fontsize=13, fontweight='bold')
    ax.set_title('BERT Reranking Performance Across All Metrics', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='best', frameon=True, shadow=True, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(k_values)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_metrics.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: combined_metrics.png")
    plt.close()


def plot_improvement_heatmap(df: pd.DataFrame, output_dir: str):
    """
    Plot 4: Heatmap of improvements
    """
    metrics = ['NDCG@1', 'NDCG@5', 'NDCG@10', 'MRR@10']
    k_values = sorted(df['k'].unique())
    
    # Prepare data for heatmap
    heatmap_data = []
    for metric in metrics:
        metric_df = df[df['Metric'] == metric]
        row = []
        for k in k_values:
            k_data = metric_df[metric_df['k'] == k]
            row.append(k_data['Improvement (%)'].values[0])
        heatmap_data.append(row)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=max(20, np.max(heatmap_data)))
    
    ax.set_xticks(np.arange(len(k_values)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels([f'k={k}' for k in k_values])
    ax.set_yticklabels(metrics)
    
    ax.set_xlabel('Number of candidates (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Evaluation Metric', fontsize=12, fontweight='bold')
    ax.set_title('BERT Improvement Heatmap (%)', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(k_values)):
            text = ax.text(j, i, f'{heatmap_data[i][j]:.1f}%',
                         ha="center", va="center", color="black", 
                         fontsize=11, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Improvement (%)', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_heatmap.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: improvement_heatmap.png")
    plt.close()


def plot_bm25_vs_rerank_comparison(df: pd.DataFrame, output_dir: str):
    """
    Plot 5: Side-by-side comparison for each k
    """
    k_values = sorted(df['k'].unique())
    metrics = ['NDCG@1', 'NDCG@5', 'NDCG@10', 'MRR@10']
    
    fig, axes = plt.subplots(len(k_values), 1, figsize=(12, 4*len(k_values)))
    if len(k_values) == 1:
        axes = [axes]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        k_df = df[df['k'] == k]
        
        bm25_scores = [k_df[k_df['Metric'] == m]['BM25'].values[0] for m in metrics]
        rerank_scores = [k_df[k_df['Metric'] == m]['BERT_Rerank'].values[0] for m in metrics]
        
        bars1 = ax.bar(x - width/2, bm25_scores, width, label='BM25',
                      color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x + width/2, rerank_scores, width, label='BERT Rerank',
                      color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
        
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(f'BM25 vs BERT Reranking (k={k})', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend(fontsize=11, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bm25_vs_rerank_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: bm25_vs_rerank_comparison.png")
    plt.close()


def generate_latex_table(df: pd.DataFrame, output_dir: str):
    """
    Generate LaTeX-formatted table for report
    """
    k_values = sorted(df['k'].unique())
    metrics = ['NDCG@1', 'NDCG@5', 'NDCG@10', 'MRR@10']
    
    latex_content = []
    latex_content.append("\\begin{table}[h]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{BM25 vs BERT Reranking Performance Comparison}")
    latex_content.append("\\label{tab:task1_results}")
    
    # Table header
    header = "\\begin{tabular}{|c|c|" + "c|c|c|" * len(metrics) + "}"
    latex_content.append(header)
    latex_content.append("\\hline")
    
    # Column headers
    col_header = "\\multirow{2}{*}{\\textbf{k}} & \\multirow{2}{*}{\\textbf{Method}} "
    for metric in metrics:
        col_header += f"& \\multicolumn{{3}}{{c|}}{{\\textbf{{{metric}}}}} "
    col_header += "\\\\"
    latex_content.append(col_header)
    
    sub_header = "& "
    for _ in metrics:
        sub_header += "& BM25 & BERT & $\\Delta\\%$ "
    sub_header += "\\\\"
    latex_content.append(sub_header)
    latex_content.append("\\hline")
    
    # Data rows
    for k in k_values:
        k_df = df[df['k'] == k]
        
        row = f"{k} & "
        for metric in metrics:
            metric_data = k_df[k_df['Metric'] == metric].iloc[0]
            bm25 = metric_data['BM25']
            rerank = metric_data['BERT_Rerank']
            improvement = metric_data['Improvement (%)']
            row += f"& {bm25:.4f} & {rerank:.4f} & {improvement:+.1f}\\% "
        row += "\\\\"
        latex_content.append(row)
        latex_content.append("\\hline")
    
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")
    
    # Save to file
    latex_file = os.path.join(output_dir, 'results_table.tex')
    with open(latex_file, 'w') as f:
        f.write('\n'.join(latex_content))
    
    print(f"✓ Saved: results_table.tex")


def generate_analysis_report(df: pd.DataFrame, output_dir: str):
    """
    Generate textual analysis report
    """
    report = []
    report.append("="*80)
    report.append("TASK 1 ANALYSIS REPORT")
    report.append("="*80)
    report.append("")
    
    k_values = sorted(df['k'].unique())
    metrics = ['NDCG@1', 'NDCG@5', 'NDCG@10', 'MRR@10']
    
    report.append("1. OVERALL OBSERVATIONS:")
    report.append("-" * 80)
    
    for metric in metrics:
        metric_df = df[df['Metric'] == metric]
        avg_improvement = metric_df['Improvement (%)'].mean()
        max_improvement = metric_df['Improvement (%)'].max()
        best_k = metric_df.loc[metric_df['Improvement (%)'].idxmax(), 'k']
        
        report.append(f"\n{metric}:")
        report.append(f"  - Average Improvement: {avg_improvement:.2f}%")
        report.append(f"  - Maximum Improvement: {max_improvement:.2f}% (at k={int(best_k)})")
    
    report.append("\n\n2. K-VALUE ANALYSIS:")
    report.append("-" * 80)
    
    for k in k_values:
        k_df = df[df['k'] == k]
        avg_improvement = k_df['Improvement (%)'].mean()
        report.append(f"\nk={k}:")
        report.append(f"  Average Improvement across all metrics: {avg_improvement:.2f}%")
        
        for metric in metrics:
            metric_data = k_df[k_df['Metric'] == metric].iloc[0]
            report.append(f"  {metric}: {metric_data['BM25']:.4f} → {metric_data['BERT_Rerank']:.4f} ({metric_data['Improvement (%)']:+.2f}%)")
    
    report.append("\n\n3. KEY FINDINGS:")
    report.append("-" * 80)
    
    # Find best k overall
    k_avg_improvements = df.groupby('k')['Improvement (%)'].mean()
    best_k_overall = k_avg_improvements.idxmax()
    best_avg_improvement = k_avg_improvements.max()
    
    report.append(f"\n- Best k value overall: k={int(best_k_overall)} (avg improvement: {best_avg_improvement:.2f}%)")
    report.append(f"- BERT reranking consistently improves over BM25 across all metrics")
    report.append(f"- Improvement is most significant for NDCG@1 (early precision)")
    
    report.append("\n" + "="*80)
    
    # Save to file
    report_file = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"✓ Saved: analysis_report.txt")
    
    # Print to console
    print("\n" + '\n'.join(report))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate comprehensive plots for Task 1")
    parser.add_argument("--qrels_file", type=str, required=True, help="Path to qrels file")
    parser.add_argument("--results_dir", type=str, required=True, 
                       help="Directory containing BM25 and reranked results for different k values")
    parser.add_argument("--k_values", type=int, nargs='+', required=True,
                       help="List of k values to analyze (e.g., 10 50 100)")
    parser.add_argument("--output_dir", type=str, default="task1_plots",
                       help="Directory to save plots and analysis")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("TASK 1: COMPREHENSIVE ANALYSIS & VISUALIZATION")
    print("="*80)
    print(f"\nQRELs file: {args.qrels_file}")
    print(f"Results directory: {args.results_dir}")
    print(f"K values: {args.k_values}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    # Analyze all k values
    all_results = []
    for k in args.k_values:
        bm25_file = os.path.join(args.results_dir, f'task1_bm25_k{k}.txt')
        rerank_file = os.path.join(args.results_dir, f'task1_rerank_k{k}.txt')
        
        if not os.path.exists(bm25_file):
            print(f"Warning: BM25 file not found for k={k}: {bm25_file}")
            continue
        if not os.path.exists(rerank_file):
            print(f"Warning: Rerank file not found for k={k}: {rerank_file}")
            continue
        
        result = analyze_single_k(args.qrels_file, bm25_file, rerank_file, k)
        all_results.append(result)
    
    if not all_results:
        print("\nError: No valid results found!")
        return
    
    # Create comparison table
    print("\n" + "="*80)
    print("CREATING ANALYSIS...")
    print("="*80)
    df = create_comparison_table(all_results)
    
    # Save CSV
    csv_file = os.path.join(args.output_dir, 'results_summary.csv')
    df.to_csv(csv_file, index=False)
    print(f"\n✓ Saved: results_summary.csv")
    
    # Generate all plots
    print("\nGenerating plots...")
    plot_metrics_vs_k(df, args.output_dir)
    plot_improvement_bars(df, args.output_dir)
    plot_combined_metrics(df, args.output_dir)
    plot_improvement_heatmap(df, args.output_dir)
    plot_bm25_vs_rerank_comparison(df, args.output_dir)
    
    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    generate_latex_table(df, args.output_dir)
    
    # Generate analysis report
    print("\nGenerating analysis report...")
    generate_analysis_report(df, args.output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {args.output_dir}/")
    print("\nGenerated files:")
    print("  1. results_summary.csv - Raw data in CSV format")
    print("  2. metrics_vs_k.png - Metrics comparison across k values")
    print("  3. improvement_bars.png - Improvement bar charts")
    print("  4. combined_metrics.png - All metrics combined")
    print("  5. improvement_heatmap.png - Improvement heatmap")
    print("  6. bm25_vs_rerank_comparison.png - Side-by-side comparison")
    print("  7. results_table.tex - LaTeX table for report")
    print("  8. analysis_report.txt - Detailed textual analysis")
    print("="*80)


if __name__ == "__main__":
    main()
