import matplotlib.pyplot as plt
import numpy as np

# 数据
emotions = ['happiness', 'sadness', 'fear', 'anger', 'surprise', 'disgust', 'neutrality']
lfmimv2 = [0.8027, 0.7976, 0.7457, 0.7648, 0.6813, 0.5096, 0.6763]
pmr = [0.8078, 0.784, 0.719, 0.7427, 0.6845, 0.5023, 0.6756]

# 设置图形
plt.figure(figsize=(10, 6))
plt.style.use('tableau-colorblind10')

# 创建柱状图
x = np.arange(len(emotions))
width = 0.35

plt.bar(x - width/2, lfmimv2, width, label='LFMIMv2', color='steelblue')
plt.bar(x + width/2, pmr, width, label='PMR', color='orange')

# 添加标签和标题
plt.xlabel('Emotions', fontsize=12, fontfamily='sans-serif')
plt.ylabel('F1 Score', fontsize=12, fontfamily='sans-serif')
plt.title('F1 Score Comparison between LFMIMv2 and PMR', fontsize=14, fontfamily='sans-serif')
plt.xticks(x, emotions, fontsize=10, fontfamily='sans-serif')
plt.yticks(fontsize=10, fontfamily='sans-serif')

# 添加图例
plt.legend(loc='upper right', fontsize=10)

# 添加数值标签
for i in range(len(emotions)):
    plt.text(i - width/2, lfmimv2[i] + 0.01, f'{lfmimv2[i]:.4f}', ha='center', fontsize=8)
    plt.text(i + width/2, pmr[i] + 0.01, f'{pmr[i]:.4f}', ha='center', fontsize=8)

# 显示网格
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 保存图形
plt.tight_layout()
plt.savefig("f1_score_comparison.png", dpi=300, bbox_inches='tight')

# 显示图形
plt.show()