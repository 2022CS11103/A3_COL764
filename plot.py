import matplotlib.pyplot as plt
import json
import numpy as np

with open('task1_results.json') as f:
    results = json.load(f)


k_values = [10, 20, 50, 100]
ndcg_1 = [results[str(k)]['NDCG@1']['mean'] for k in k_values]
ndcg_5 = [results[str(k)]['NDCG@5']['mean'] for k in k_values]
ndcg_10 = [results[str(k)]['NDCG@10']['mean'] for k in k_values]
mrr_10 = [results[str(k)]['MRR@10']['mean'] for k in k_values]

times = [22.98, 57.62, 159.04, 339.79]

print("Data extracted:")
print(f"k_values: {k_values}")
print(f"NDCG@1: {ndcg_1}")
print(f"NDCG@5: {ndcg_5}")
print(f"NDCG@10: {ndcg_10}")
print(f"MRR@10: {mrr_10}")
print(f"Times: {times}")

# ============================================================================
# Plot 1: NDCG vs k
# ============================================================================
plt.figure(figsize=(12, 7))
plt.plot(k_values, ndcg_1, 'o-', label='NDCG@1', linewidth=2.5, markersize=8, color='#1f77b4')
plt.plot(k_values, ndcg_5, 's-', label='NDCG@5', linewidth=2.5, markersize=8, color='#ff7f0e')
plt.plot(k_values, ndcg_10, '^-', label='NDCG@10', linewidth=2.5, markersize=8, color='#2ca02c')

plt.xlabel('Retrieval Size (k)', fontsize=13, fontweight='bold')
plt.ylabel('NDCG Score', fontsize=13, fontweight='bold')
plt.title('NDCG Performance vs Retrieval Size k\n(Retrieve-and-Rerank with BERT on TREC-DL-Hard)', 
          fontsize=14, fontweight='bold', pad=20)
plt.legend(fontsize=11, loc='lower right')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(k_values, fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.savefig('plot_1_ndcg_vs_k.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: plot_1_ndcg_vs_k.png")
plt.close()

# ============================================================================
# Plot 2: MRR vs k
# ============================================================================
plt.figure(figsize=(12, 7))
plt.plot(k_values, mrr_10, 'o-', color='#d62728', linewidth=2.5, markersize=10)


for i, (k, mrr) in enumerate(zip(k_values, mrr_10)):
    plt.text(k, mrr + 0.01, f'{mrr:.4f}', ha='center', fontsize=10, fontweight='bold')

plt.xlabel('Retrieval Size (k)', fontsize=13, fontweight='bold')
plt.ylabel('MRR@10 Score', fontsize=13, fontweight='bold')
plt.title('MRR@10 Performance vs Retrieval Size k\n(Retrieve-and-Rerank with BERT on TREC-DL-Hard)', 
          fontsize=14, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(k_values, fontsize=11)
plt.yticks(fontsize=11)
plt.ylim([0.58, 0.65])
plt.tight_layout()
plt.savefig('plot_2_mrr_vs_k.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plot_2_mrr_vs_k.png")
plt.close()

# ============================================================================
# Plot 3: Processing Time vs k
# ============================================================================
plt.figure(figsize=(12, 7))
plt.plot(k_values, times, 'o-', color='#9467bd', linewidth=2.5, markersize=10)


for i, (k, t) in enumerate(zip(k_values, times)):
    plt.text(k, t + 10, f'{t:.1f}s', ha='center', fontsize=10, fontweight='bold')

plt.xlabel('Retrieval Size (k)', fontsize=13, fontweight='bold')
plt.ylabel('Processing Time (seconds)', fontsize=13, fontweight='bold')
plt.title('Processing Time vs Retrieval Size k\n(For 50 Queries - Retrieve-and-Rerank with BERT)', 
          fontsize=14, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(k_values, fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.savefig('plot_3_time_vs_k.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plot_3_time_vs_k.png")
plt.close()

# ============================================================================
# Plot 4: All Metrics Comparison (Bonus)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# NDCG@1
axes[0, 0].plot(k_values, ndcg_1, 'o-', color='#1f77b4', linewidth=2.5, markersize=8)
axes[0, 0].set_title('NDCG@1', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Score', fontsize=11)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks(k_values)

# NDCG@5
axes[0, 1].plot(k_values, ndcg_5, 's-', color='#ff7f0e', linewidth=2.5, markersize=8)
axes[0, 1].set_title('NDCG@5', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xticks(k_values)

# NDCG@10
axes[1, 0].plot(k_values, ndcg_10, '^-', color='#2ca02c', linewidth=2.5, markersize=8)
axes[1, 0].set_title('NDCG@10', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Retrieval Size (k)', fontsize=11)
axes[1, 0].set_ylabel('Score', fontsize=11)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xticks(k_values)

# MRR@10
axes[1, 1].plot(k_values, mrr_10, 'o-', color='#d62728', linewidth=2.5, markersize=8)
axes[1, 1].set_title('MRR@10', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Retrieval Size (k)', fontsize=11)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xticks(k_values)

fig.suptitle('All Metrics vs Retrieval Size k (Retrieve-and-Rerank with BERT)', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('plot_4_all_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plot_4_all_metrics.png")
plt.close()

print("\n" + "="*70)
print("RESULTS SUMMARY TABLE")
print("="*70)
print(f"{'k':<5} {'NDCG@1':<10} {'NDCG@5':<10} {'NDCG@10':<10} {'MRR@10':<10} {'Time(s)':<10}")
print("-"*70)
for i, k in enumerate(k_values):
    print(f"{k:<5} {ndcg_1[i]:<10.4f} {ndcg_5[i]:<10.4f} {ndcg_10[i]:<10.4f} {mrr_10[i]:<10.4f} {times[i]:<10.2f}")
print("="*70)


print("\nKEY OBSERVATIONS:")
print("-"*70)
improvement_ndcg10 = ((ndcg_10[-1] - ndcg_10[0]) / ndcg_10[0]) * 100
improvement_mrr = ((mrr_10[-1] - mrr_10[0]) / mrr_10[0]) * 100
print(f"1. NDCG@10 improvement (k=10 to k=100): {improvement_ndcg10:.1f}%")
print(f"   - k=10: {ndcg_10[0]:.4f} → k=100: {ndcg_10[-1]:.4f}")
print(f"\n2. MRR@10 improvement (k=10 to k=100): {improvement_mrr:.1f}%")
print(f"   - k=10: {mrr_10[0]:.4f} → k=100: {mrr_10[-1]:.4f}")
print(f"\n3. Processing time ratio (k=100 vs k=10): {times[-1]/times[0]:.1f}x slower")
print(f"   - k=10: {times[0]:.2f}s → k=100: {times[-1]:.2f}s")
print(f"\n4. Best NDCG@10: k=100 ({ndcg_10[-1]:.4f})")
print(f"   Best MRR@10: k=100 ({mrr_10[-1]:.4f})")
print("="*70)

print("\n✓ All plots generated successfully!")
print("\nFiles created:")
print("  - plot_1_ndcg_vs_k.png")
print("  - plot_2_mrr_vs_k.png")
print("  - plot_3_time_vs_k.png")
print("  - plot_4_all_metrics.png (bonus)")