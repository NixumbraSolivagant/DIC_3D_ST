import os
import re
import numpy as np
import matplotlib.pyplot as plt
import sys

# 用户可在此处设置理论位移（单位与统计文件一致）
theoretical_value = 1.0  # 例如1微米

use_superres = False
if '--superres' in sys.argv:
    use_superres = True

if use_superres:
    stats_dir = 'output_sr/output_3d'
else:
    stats_dir = 'output/output_3d'

pattern = re.compile(r'displacement_stats_(\\d+)\\.txt')

# 收集所有统计文件
txt_files = [f for f in os.listdir(stats_dir) if f.startswith('displacement_stats_') and f.endswith('.txt')]
def extract_num(x):
    nums = re.findall(r'\d+', x)
    return int(nums[0]) if nums else -1
txt_files = [f for f in txt_files if extract_num(f) != -1]
txt_files = sorted(txt_files, key=extract_num)

mean_disps = []
for fname in txt_files:
    with open(os.path.join(stats_dir, fname), 'r') as f:
        lines = f.readlines()
        for line in lines:
            if '平均位移' in line:
                val = float(line.split(':')[1].split()[0])
                mean_disps.append(val)
                break

mean_disps = np.array(mean_disps)
N = len(mean_disps)

if N == 0:
    print("没有有效的平均位移数据，请检查上游流程和统计文件内容。")
    exit(1)

# 计算RMSE和最大误差
rmse = np.sqrt(np.mean((mean_disps - theoretical_value)**2))
max_err = np.max(np.abs(mean_disps - theoretical_value))

# 绘图
plt.figure(figsize=(8,5))
plt.plot(range(1, N+1), mean_disps, marker='o', label='平均位移模长')
plt.axhline(theoretical_value, color='r', linestyle='--', label=f'理论值={theoretical_value}')
plt.xlabel('图像对编号')
plt.ylabel('平均位移 (单位同统计文件)')
plt.title('每组图像平均位移模长与理论值对比')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(stats_dir, 'mean_displacement_vs_theory.png'))
plt.show()

print(f'RMSE: {rmse:.4f}, 最大误差: {max_err:.4f}') 