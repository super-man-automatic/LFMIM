import matplotlib.pyplot as plt
import numpy as np

# Epochs
epochs = np.arange(1, 21)

# LFMIM数据
lfmim_test_loss_t = [1.9043, 1.2758, 1.2984, 1.3238, 1.3411, 1.3574, 1.3814, 1.3990, 1.4235, 1.4407,
                    1.4577, 1.4822, 1.4996, 1.5232, 1.5330, 1.5627, 1.5746, 1.5843, 1.6078, 1.6308]
lfmim_test_loss_a = [2.0881, 0.8402, 0.8187, 0.8154, 0.8156, 0.8163, 0.8168, 0.8179, 0.8177, 0.8185,
                    0.8193, 0.8199, 0.8218, 0.8220, 0.8224, 0.8227, 0.8233, 0.8236, 0.8250, 0.8262]
lfmim_test_loss_v = [1.8992, 1.1212, 1.1110, 1.1176, 1.1409, 1.1582, 1.1657, 1.1807, 1.1976, 1.2053,
                    1.2166, 1.2304, 1.2404, 1.2515, 1.2578, 1.2694, 1.2786, 1.2940, 1.2979, 1.3035]
lfmim_test_loss_m = [1.9322, 0.7955, 0.7761, 0.7686, 0.7627, 0.7621, 0.7616, 0.7605, 0.7633, 0.7646,
                    0.7670, 0.7683, 0.7720, 0.7768, 0.7772, 0.7828, 0.7874, 0.7905, 0.7954, 0.7998]

lfmim_test_acc_t = [0.2477, 0.6616, 0.6621, 0.6623, 0.6631, 0.6650, 0.6643, 0.6640, 0.6628, 0.6624,
                   0.6642, 0.6636, 0.6640, 0.6623, 0.6624, 0.6619, 0.6593, 0.6610, 0.6591, 0.6586]
lfmim_test_acc_a = [0.0696, 0.7110, 0.7190, 0.7216, 0.7214, 0.7218, 0.7216, 0.7214, 0.7221, 0.7223,
                   0.7219, 0.7224, 0.7224, 0.7228, 0.7219, 0.7221, 0.7219, 0.7218, 0.7216, 0.7209]
lfmim_test_acc_v = [0.2275, 0.6708, 0.6781, 0.6755, 0.6732, 0.6727, 0.6729, 0.6734, 0.6731, 0.6717,
                   0.6723, 0.6723, 0.6716, 0.6709, 0.6706, 0.6715, 0.6694, 0.6697, 0.6667, 0.6674]
lfmim_test_acc_m = [0.1978, 0.7107, 0.7171, 0.7206, 0.7220, 0.7223, 0.7211, 0.7208, 0.7194, 0.7202,
                   0.7197, 0.7195, 0.7187, 0.7199, 0.7204, 0.7204, 0.7201, 0.7209, 0.7209, 0.7197]

# PMR数据
pmr_test_loss_t = [1.1168, 1.2218, 1.2180, 1.2667, 1.3194, 1.3884, 1.4224, 1.4648, 1.5257, 1.5496,
                  1.5576, 1.5542, 1.5342, 1.5613, 1.5919, 1.5919, 1.5919, 1.5919, 1.5919, 1.5919]
pmr_test_loss_a = [0.8143, 0.8133, 0.8038, 0.8028, 0.8496, 0.8624, 0.8657, 0.8657, 0.9215, 0.9445,
                  0.9773, 0.9773, 0.9773, 0.9773, 1.0015, 1.0015, 1.0015, 1.0015, 1.0015, 1.0015]
pmr_test_loss_v = [0.9349, 0.9706, 0.9753, 1.0179, 1.1235, 1.1590, 1.1635, 1.1635, 1.2216, 1.2735,
                  1.3320, 1.3320, 1.3320, 1.3320, 1.4098, 1.4098, 1.4098, 1.4098, 1.4098, 1.4098]
pmr_test_loss_m = [0.7872, 0.7716, 0.7709, 0.7766, 0.8111, 0.8111, 0.8111, 0.8111, 0.8172, 0.8379,
                  0.8549, 0.8549, 0.8549, 0.8549, 0.8788, 0.8788, 0.8788, 0.8788, 0.8788, 0.8788]

pmr_test_acc_t = [0.6397, 0.6448, 0.6439, 0.6474, 0.6493, 0.6451, 0.6432, 0.6429, 0.6390, 0.6396,
                 0.6396, 0.6418, 0.6418, 0.6418, 0.6378, 0.6378, 0.6378, 0.6378, 0.6378, 0.6378]
pmr_test_acc_a = [0.6874, 0.6996, 0.6972, 0.6960, 0.6942, 0.6942, 0.6932, 0.6934, 0.6934, 0.6939,
                 0.6939, 0.6909, 0.6909, 0.6909, 0.6857, 0.6857, 0.6857, 0.6857, 0.6857, 0.6857]
pmr_test_acc_v = [0.6653, 0.6695, 0.6692, 0.6693, 0.6648, 0.6645, 0.6615, 0.6613, 0.6613, 0.6568,
                 0.6568, 0.6587, 0.6587, 0.6587, 0.6625, 0.6625, 0.6625, 0.6625, 0.6625, 0.6625]
pmr_test_acc_m = [0.6827, 0.6908, 0.6911, 0.6934, 0.6920, 0.6918, 0.6873, 0.6883, 0.6883, 0.6883,
                 0.6883, 0.6876, 0.6876, 0.6876, 0.6824, 0.6824, 0.6824, 0.6824, 0.6824, 0.6824]

# 设置图表样式
plt.style.use('classic')

# 创建第一个图表：测试损失
fig1, ax1 = plt.subplots(figsize=(6, 5))
ax1.plot(epochs, lfmim_test_loss_t, label='t(LFMIM)', color='green', linestyle='-')
ax1.plot(epochs, lfmim_test_loss_a, label='a(LFMIM)', color='gold', linestyle='-')
ax1.plot(epochs, lfmim_test_loss_v, label='v(LFMIM)', color='orange', linestyle='-')
ax1.plot(epochs, lfmim_test_loss_m, label='m(LFMIM)', color='blue', linestyle='-')
ax1.plot(epochs, pmr_test_loss_t, label='t(PMR)', color='green', linestyle='--')
ax1.plot(epochs, pmr_test_loss_a, label='a(PMR)', color='gold', linestyle='--')
ax1.plot(epochs, pmr_test_loss_v, label='v(PMR)', color='orange', linestyle='--')
ax1.plot(epochs, pmr_test_loss_m, label='m(PMR)', color='blue', linestyle='--')

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Test Loss')
ax1.legend(loc='upper right', fontsize='x-small', markerscale=0.7)  # 进一步减小图例字体和标记大小
ax1.grid(True)
fig1.tight_layout()  # 自动调整图表布局
fig1.savefig('test_loss.png', dpi=300, bbox_inches='tight')

# 创建第二个图表：测试准确率
fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.plot(epochs, lfmim_test_acc_t, label='t(LFMIM)', color='green', linestyle='-')
ax2.plot(epochs, lfmim_test_acc_a, label='a(LFMIM)', color='gold', linestyle='-')
ax2.plot(epochs, lfmim_test_acc_v, label='v(LFMIM)', color='orange', linestyle='-')
ax2.plot(epochs, lfmim_test_acc_m, label='m(LFMIM)', color='blue', linestyle='-')
ax2.plot(epochs, pmr_test_acc_t, label='t(PMR)', color='green', linestyle='--')
ax2.plot(epochs, pmr_test_acc_a, label='a(PMR)', color='gold', linestyle='--')
ax2.plot(epochs, pmr_test_acc_v, label='v(PMR)', color='orange', linestyle='--')
ax2.plot(epochs, pmr_test_acc_m, label='m(PMR)', color='blue', linestyle='--')

ax2.set_xlabel('Epochs')
ax2.set_ylabel('Test Accuracy')
ax2.legend(loc='lower right', fontsize='x-small', markerscale=0.7)  # 进一步减小图例字体和标记大小
ax2.grid(True)
fig2.tight_layout()  # 自动调整图表布局
fig2.savefig('test_accuracy.png', dpi=300, bbox_inches='tight')

plt.close(fig1)
plt.close(fig2)
