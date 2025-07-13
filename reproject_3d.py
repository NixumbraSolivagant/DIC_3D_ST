import cv2
import numpy as np
import os

params = np.load('calib_params.npz')
Q = params['Q']
disp_dir = 'output_disp'
match_dir = 'output_match'
output_dir = 'output_3d'
os.makedirs(output_dir, exist_ok=True)
num_pairs = len([f for f in os.listdir(disp_dir) if f.endswith('.npy')])
for idx in range(1, num_pairs):
    disp_b = np.load(os.path.join(disp_dir, f'disp_{idx}.npy'))
    disp_a = np.load(os.path.join(disp_dir, f'disp_{idx+1}.npy'))
    points_3d_b = cv2.reprojectImageTo3D(disp_b, Q)
    points_3d_a = cv2.reprojectImageTo3D(disp_a, Q)
    match = np.load(os.path.join(match_dir, f'points_{idx}.npz'))
    pts_b = match['pts1']
    pts_a = match['pts2']
    # 调试：输出前20个匹配点的视差和z值分布（只打印第1组）
    if idx == 1:
        print('前20个匹配点的视差和z值：')
        for j, (pt_b, pt_a) in enumerate(zip(pts_b, pts_a)):
            if j >= 20: break
            x_b, y_b = map(int, pt_b)
            x_a, y_a = map(int, pt_a)
            disp_val_b = disp_b[y_b, x_b] if 0 <= y_b < disp_b.shape[0] and 0 <= x_b < disp_b.shape[1] else -1
            disp_val_a = disp_a[y_a, x_a] if 0 <= y_a < disp_a.shape[0] and 0 <= x_a < disp_a.shape[1] else -1
            z_b = points_3d_b[y_b, x_b][2] if 0 <= y_b < points_3d_b.shape[0] and 0 <= x_b < points_3d_b.shape[1] else -1
            z_a = points_3d_a[y_a, x_a][2] if 0 <= y_a < points_3d_a.shape[0] and 0 <= x_a < points_3d_a.shape[1] else -1
            print(f'点{j}: disp_b={disp_val_b:.2f}, disp_a={disp_val_a:.2f}, z_b={z_b:.2f}, z_a={z_a:.2f}')
    # 位移分析与保存（可复用你原有的过滤和统计逻辑）
    # 这里假设 pts_b, pts_a, points_3d_b, points_3d_a 已经准备好
    displacements = []
    mags = []
    for pt_b, pt_a in zip(pts_b, pts_a):
        x_b, y_b = map(int, pt_b)
        x_a, y_a = map(int, pt_a)
        if not (0 <= y_b < points_3d_b.shape[0] and 0 <= x_b < points_3d_b.shape[1] and 0 <= y_a < points_3d_a.shape[0] and 0 <= x_a < points_3d_a.shape[1]):
            continue
        p3d_b = points_3d_b[y_b, x_b]
        p3d_a = points_3d_a[y_a, x_a]
        if not (np.isfinite(p3d_b).all() and np.isfinite(p3d_a).all()):
            continue
        disp = p3d_a - p3d_b
        mag = np.linalg.norm(disp)
        if mag > 1000 or mag == 0:
            continue
        displacements.append((pt_b, pt_a, disp, mag))
        mags.append(mag)
    # 剔除离群点：使用IQR法（四分位数法）过滤异常值
    if mags:
        mags_np = np.array(mags)
        Q1 = np.percentile(mags_np, 25)
        Q3 = np.percentile(mags_np, 75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        filtered = [(pt_b, pt_a, disp) for (pt_b, pt_a, disp, mag) in displacements if lower <= mag <= upper]
    else:
        filtered = []
    # 保存所有三维位移向量（用filtered替换displacements）
    with open(os.path.join(output_dir, f'displacement_vectors_{idx}.txt'), 'w') as f:
        f.write('x_b,y_b,x_a,y_a,dx,dy,dz\n')
        for pt_b, pt_a, disp in filtered:
            f.write(f'{pt_b[0]:.2f},{pt_b[1]:.2f},{pt_a[0]:.2f},{pt_a[1]:.2f},{disp[0]:.4f},{disp[1]:.4f},{disp[2]:.4f}\n')
    # 保存统计信息
    if filtered:
        mags = np.array([np.linalg.norm(disp) for _, _, disp in filtered])
        mean_disp = np.mean([disp for _, _, disp in filtered], axis=0)
        std_disp = np.std([disp for _, _, disp in filtered], axis=0)
        with open(os.path.join(output_dir, f'displacement_stats_{idx}.txt'), 'w') as f:
            f.write(f'有效位移点数: {len(mags)}\n')
            f.write(f'平均位移: {np.mean(mags):.2f} mm\n')
            f.write(f'最小位移: {np.min(mags):.2f} mm\n')
            f.write(f'最大位移: {np.max(mags):.2f} mm\n')
            f.write(f'位移标准差: {np.std(mags):.2f} mm\n')
            f.write(f'三维位移均值: dx={mean_disp[0]:.2f}, dy={mean_disp[1]:.2f}, dz={mean_disp[2]:.2f}\n')
            f.write(f'三维位移标准差: dx={std_disp[0]:.2f}, dy={std_disp[1]:.2f}, dz={std_disp[2]:.2f}\n')
    else:
        with open(os.path.join(output_dir, f'displacement_stats_{idx}.txt'), 'w') as f:
            f.write('无有效位移点\n') 