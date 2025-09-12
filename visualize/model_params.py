from matplotlib import pyplot as plt
import numpy as np
import matplotlib
# 使用Matplotlib自带的字体，避免字体缺失问题
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
figsize = (5, 3)  # PR size

plt.figure(figsize=figsize, dpi=1200)

der_y = [11.2 * i for i in range(1, 11)]
der_std = [0 for i in range(1, 11)]

ours_y = [1.12 * 0.01 for i in range(1, 11)]
ours_std = [0 for i in range(1, 11)]

more_y = [24.11 + i * 0.13 for i in range(1, 11)]
more_std = [0 for i in range(1, 11)]

# 对y值进行对数转换（添加一个小的epsilon防止log(0)）
epsilon = 1e-10  # 防止对0取对数
der_y_log = [np.log10(y + epsilon) for y in der_y]
ours_y_log = [np.log10(y + epsilon) for y in ours_y]
more_y_log = [np.log10(y + epsilon) for y in more_y]

# 对误差条也进行对数转换（如果误差条不为0）
der_std_log = [np.log10(std + epsilon) if std > 0 else 0 for std in der_std]
ours_std_log = [np.log10(std + epsilon) if std > 0 else 0 for std in ours_std]
more_std_log = [np.log10(std + epsilon) if std > 0 else 0 for std in more_std]

l_der = plt.errorbar(x, der_y_log, yerr=der_std_log, fmt='o-', color='green')
l_ours = plt.errorbar(x, ours_y_log, yerr=ours_std_log, fmt='o-', color='red')
l_more = plt.errorbar(x, more_y_log, yerr=more_std_log, fmt='o-', color='blue')

# 设置Y轴为对数坐标
# plt.yscale('log')

plt.grid(ls="--", c='gray', axis='y', linewidth=1)
plt.xlim((9, 101))

# 设置Y轴范围（对数坐标）
plt.ylim((1, 2.2))  # 根据您的数据范围调整

plt.xticks(x, fontsize=10)
# 设置对数坐标的刻度
# plt.yticks(fontsize=10)
y_ticks = np.arange(0, 2, 10)

plt.xlabel('Number of Classes', fontsize=10)
plt.ylabel('Trainable Parameters (Millions, log scale)', fontsize=10)

plt.savefig('./cifar100_b10i10_model_params.png', bbox_inches='tight')
# plt.show()