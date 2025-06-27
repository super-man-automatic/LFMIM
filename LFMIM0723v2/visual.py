import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import os

# 创建保存图片的文件夹
if not os.path.exists('image'):
    os.makedirs('image')

# 数据准备
epochs = np.arange(20)  # 20个训练周期

# 训练损失数据
train_loss_t = [1.8487, 0.2716, 0.2705, 0.2566, 0.2270, 0.1873, 0.1976, 0.1874, 0.1796, 0.1877,
                0.1388, 0.1277, 0.1275, 0.1423, 0.1388, 0.1284, 0.0805, 0.1277, 0.0827, 0.0827]
train_loss_a = [2.1128, 0.4102, 0.3855, 0.4010, 0.4074, 0.4139, 0.4121, 0.4025, 0.4043, 0.4044,
                0.3997, 0.4103, 0.4117, 0.4145, 0.3997, 0.4022, 0.4130, 0.3986, 0.4162, 0.4162]
train_loss_v = [1.9616, 0.4334, 0.4066, 0.3837, 0.3777, 0.3573, 0.3199, 0.3308, 0.3159, 0.2996,
                0.3184, 0.2793, 0.3143, 0.2690, 0.2606, 0.2426, 0.2477, 0.2342, 0.2037, 0.2037]
train_loss_m = [1.9316, 0.9606, 0.9208, 0.8945, 0.8602, 0.8566, 0.7841, 0.7471, 0.7607, 0.7233,
                0.7248, 0.6922, 0.7084, 0.7009, 0.6461, 0.6738, 0.5979, 0.5544, 0.5808, 0.5808]

# 测试损失数据（保留但不绘制2a）
test_loss_t = [1.9043, 1.2758, 1.2984, 1.3238, 1.3411, 1.3574, 1.3814, 1.3990, 1.4235, 1.4407,
               1.4577, 1.4822, 1.4996, 1.5232, 1.5330, 1.5627, 1.5746, 1.5843, 1.6078, 1.6308]
test_loss_a = [2.0881, 0.8402, 0.8187, 0.8154, 0.8156, 0.8163, 0.8168, 0.8179, 0.8177, 0.8185,
               0.8193, 0.8199, 0.8218, 0.8220, 0.8224, 0.8227, 0.8233, 0.8236, 0.8250, 0.8262]
test_loss_v = [1.8992, 1.1212, 1.1110, 1.1176, 1.1409, 1.1582, 1.1657, 1.1807, 1.1976, 1.2053,
               1.2166, 1.2304, 1.2404, 1.2515, 1.2578, 1.2694, 1.2786, 1.2940, 1.2979, 1.3035]
test_loss_m = [1.9322, 0.7955, 0.7761, 0.7686, 0.7627, 0.7621, 0.7616, 0.7605, 0.7633, 0.7646,
               0.7670, 0.7683, 0.7720, 0.7768, 0.7772, 0.7828, 0.7874, 0.7905, 0.7954, 0.7998]

# 测试准确率数据
test_acc_t = [0.2477, 0.6616, 0.6621, 0.6623, 0.6631, 0.6650, 0.6643, 0.6640, 0.6628, 0.6624,
              0.6642, 0.6636, 0.6640, 0.6623, 0.6624, 0.6619, 0.6593, 0.6610, 0.6591, 0.6586]
test_acc_a = [0.0696, 0.7110, 0.7190, 0.7216, 0.7214, 0.7218, 0.7216, 0.7214, 0.7221, 0.7223,
              0.7219, 0.7224, 0.7224, 0.7228, 0.7219, 0.7221, 0.7219, 0.7218, 0.7216, 0.7209]
test_acc_v = [0.2275, 0.6708, 0.6781, 0.6755, 0.6732, 0.6727, 0.6729, 0.6734, 0.6731, 0.6717,
              0.6723, 0.6723, 0.6716, 0.6709, 0.6706, 0.6715, 0.6694, 0.6697, 0.6667, 0.6674]
test_acc_m = [0.1978, 0.7107, 0.7171, 0.7206, 0.7220, 0.7223, 0.7211, 0.7208, 0.7194, 0.7202,
              0.7197, 0.7195, 0.7187, 0.7199, 0.7204, 0.7204, 0.7201, 0.7209, 0.7209, 0.7197]

# LFMIM模型在不同情绪类别上的F1分数（假设数据，需替换为真实值）
emotions = ['happiness', 'sadness', 'fear', 'anger', 'surprise', 'disgust', 'neutrality']
lfmim_f1_scores = [0.7960, 0.7770, 0.7179, 0.7649, 0.6694, 0.4850, 0.0032]  # 示例数据


# 定义绘图函数
def plot_training_test_loss():
    plt.plot(epochs, train_loss_t, label='Train-Text', color='#00A1FF', linestyle='-', linewidth=2)
    plt.plot(epochs, test_loss_t, label='Test-Text', color='#00A1FF', linestyle='--', linewidth=2)
    plt.plot(epochs, train_loss_a, label='Train-Audio', color='#5ed935', linestyle='-', linewidth=2)
    plt.plot(epochs, test_loss_a, label='Test-Audio', color='#5ed935', linestyle='--', linewidth=2)
    plt.plot(epochs, train_loss_v, label='Train-Visual', color='#f8ba00', linestyle='-', linewidth=2)
    plt.plot(epochs, test_loss_v, label='Test-Visual', color='#f8ba00', linestyle='--', linewidth=2)
    plt.plot(epochs, train_loss_m, label='Train-Multimodal', color='#ff2501', linestyle='-', linewidth=2)
    plt.plot(epochs, test_loss_m, label='Test-Multimodal', color='#ff2501', linestyle='--', linewidth=2)
    plt.legend(loc='upper right')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylim(bottom=0)

def plot_modality_accuracy():
    plt.plot(epochs, test_acc_t, label='Text', color='#00A1FF', linewidth=2)
    plt.plot(epochs, test_acc_a, label='Audio', color='#5ed935', linewidth=2)
    plt.plot(epochs, test_acc_v, label='Visual', color='#f8ba00', linewidth=2)
    plt.plot(epochs, test_acc_m, label='Multimodal', color='#ff2501', linewidth=2)
    plt.legend(loc='lower right')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylim(bottom=0, top=1)

def plot_emotion_f1_scores():
    x = np.arange(len(emotions))
    width = 0.4
    plt.bar(x, lfmim_f1_scores, width, color=['#00A1FF', '#5ed935', '#f8ba00', '#ff2501', '#d31876', '#919292', '#58538b'])
    plt.xticks(x, emotions, rotation=45, ha='right')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')


# 绘制并保存图表
def save_figure(fig_title, x_label, y_label, plot_func, filename):
    plt.figure(figsize=(8, 6))
    plot_func()
    plt.title(fig_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'image/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()


# 图1: (a) Training and Test Loss by Modality
save_figure(
    'Training and Test Loss by Modality',
    'Epochs',
    'Loss',
    plot_training_test_loss,
    'figure1a'
)

# 图2: (b) Overall Emotion Recognition Accuracy (与1b相同)
save_figure(
    'Overall Emotion Recognition Accuracy',
    'Epochs',
    'Accuracy',
    plot_modality_accuracy,
    'figure1b'
)

# 图3: (b) Overall Test Accuracy by Modality (移除2a后，原2b改为新图2b，但需求要求与1b相同，故实际保留3图)
# 图4: (c) F1 Scores of LFMIM Model on Different Emotions
save_figure(
    'LFMIM Model F1 Scores by Emotion Category',
    'Emotion Category',
    'F1 Score',
    plot_emotion_f1_scores,
    'figure2c'
)

print("All figures saved to 'image' folder.")