import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['ResNet18', 'ResNet34', 'ResNet50']
types = ['FP32', 'QAT', 'PTQ XNNPACK', 'QAT XNNPACK']

data = {
    'mAP': {
        'ResNet18': [0.6631, 0.6638, 0.6576, 0.6462],
        'ResNet34': [0.7097, 0.7048, 0.7053, 0.1616],
        'ResNet50': [0.7721, 0.7696, 0.7666, 0.5372],
    },
    'Precision': {
        'ResNet18': [0.7248, 0.7160, 0.7279, 0.7200],
        'ResNet34': [0.7992, 0.7945, 0.7421, 0.2522],
        'ResNet50': [0.8364, 0.8389, 0.8357, 0.6226],
    },
    'Recall': {
        'ResNet18': [0.7194, 0.7291, 0.7155, 0.7031],
        'ResNet34': [0.7088, 0.7054, 0.7589, 0.3177],
        'ResNet50': [0.7638, 0.7644, 0.7614, 0.5573],
    },
    'F1': {
        'ResNet18': [0.6890, 0.6896, 0.6886, 0.6784],
        'ResNet34': [0.7220, 0.7178, 0.7216, 0.2544],
        'ResNet50': [0.7713, 0.7731, 0.7699, 0.5518],
    },
}

colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

# Figure 1: mAP comparison
fig1, ax1 = plt.subplots(figsize=(10, 6))
x = np.arange(len(models))
width = 0.18

for i, t in enumerate(types):
    values = [data['mAP'][m][i] for m in models]
    bars = ax1.bar(x + i * width, values, width, label=t, color=colors[i])
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=8)

ax1.set_xlabel('Model', fontsize=12)
ax1.set_ylabel('mAP', fontsize=12)
ax1.set_title('mAP Comparison by Model and Quantization Method', fontsize=14, fontweight='bold')
ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels(models)
ax1.legend(loc='upper left')
ax1.set_ylim(0, 1.0)
ax1.grid(axis='y', alpha=0.3)
fig1.tight_layout()
fig1.savefig('evaluation_resultses/mAP_comparison.png', dpi=150)
print('Saved: evaluation_resultses/mAP_comparison.png')

# Figure 2: All metrics comparison (4 subplots)
fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
metrics = ['mAP', 'Precision', 'Recall', 'F1']

for idx, (metric, ax) in enumerate(zip(metrics, axes.flatten())):
    x = np.arange(len(models))
    for i, t in enumerate(types):
        values = [data[metric][m][i] for m in models]
        bars = ax.bar(x + i * width, values, width, label=t, color=colors[i])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

fig2.suptitle('Evaluation Metrics Comparison: ResNet Models × Quantization Methods',
              fontsize=15, fontweight='bold')
fig2.tight_layout(rect=[0, 0, 1, 0.96])
fig2.savefig('evaluation_resultses/all_metrics_comparison.png', dpi=150, bbox_inches='tight')
print('Saved: evaluation_resultses/all_metrics_comparison.png')

print('Done!')
