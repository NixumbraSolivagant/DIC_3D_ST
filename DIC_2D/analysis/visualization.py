import matplotlib.pyplot as plt
import numpy as np
import os
import config

def save_all(results, cfg):
    output_dir = cfg.VISUAL_OUTPUT
    os.makedirs(output_dir, exist_ok=True)
    means = [r['mean'] for r in results]
    stds = [r['std'] for r in results]
    idxs = [r['idx'] for r in results]
    # 主曲线
    plt.figure(figsize=(8, 5))
    plt.plot(idxs, means, 'o-', label='像素位移均值')
    plt.fill_between(idxs, np.array(means)-np.array(stds), np.array(means)+np.array(stds), color='gray', alpha=0.3, label='±1σ')
    plt.xlabel('组编号')
    plt.ylabel('位移 (像素)')
    plt.title('2D平面位移随组变化曲线')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2d_displacement_curve.png'))
    plt.close() 
    # 可选：极线、直方图等可视化（如有需要可扩展） 