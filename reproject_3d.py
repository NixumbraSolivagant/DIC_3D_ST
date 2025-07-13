import sys
use_superres = False
if '--superres' in sys.argv:
    use_superres = True
import cv2
import numpy as np
import os

param_dir = 'output_sr' if use_superres else 'output'
params = np.load(os.path.join(param_dir, 'calib_params.npz'))
Q = params['Q']

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

if use_superres:
    disp_dir = 'output_sr/output_disp'
    match_dir = 'output_sr/output_match'
    output_dir = 'output_sr/output_3d'
    ensure_dir('output_sr')
else:
    disp_dir = 'output/output_disp'
    match_dir = 'output/output_match'
    output_dir = 'output/output_3d'
    ensure_dir('output')
ensure_dir(output_dir)
num_pairs = len([f for f in os.listdir(disp_dir) if f.endswith('.npy')])

def bilinear_interp(img, pt):
    x, y = pt
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0+1, img.shape[1]-1), min(y0+1, img.shape[0]-1)
    if not (0 <= x0 < img.shape[1]-1 and 0 <= y0 < img.shape[0]-1):
        return np.array([np.nan, np.nan, np.nan])
    dx, dy = x - x0, y - y0
    val = (img[y0, x0] * (1-dx)*(1-dy) +
           img[y0, x1] * dx*(1-dy) +
           img[y1, x0] * (1-dx)*dy +
           img[y1, x1] * dx*dy)
    return val

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
        z_list = []
        for j, (pt_b, pt_a) in enumerate(zip(pts_b, pts_a)):
            if j >= 20: break
            x_b, y_b = map(int, pt_b)
            x_a, y_a = map(int, pt_a)
            disp_val_b = disp_b[y_b, x_b] if 0 <= y_b < disp_b.shape[0] and 0 <= x_b < disp_b.shape[1] else -1
            disp_val_a = disp_a[y_a, x_a] if 0 <= y_a < disp_a.shape[0] and 0 <= x_a < disp_a.shape[1] else -1
            z_b = points_3d_b[y_b, x_b][2] if 0 <= y_b < points_3d_b.shape[0] and 0 <= x_b < points_3d_b.shape[1] else -1
            z_a = points_3d_a[y_a, x_a][2] if 0 <= y_a < points_3d_a.shape[0] and 0 <= x_a < points_3d_a.shape[1] else -1
            print(f'点{j}: disp_b={disp_val_b:.2f}, disp_a={disp_val_a:.2f}, z_b={z_b:.2f}, z_a={z_a:.2f}')
            z_list.append(z_b)
        if z_list:
            print(f'z_b分布: min={np.min(z_list):.2f}, max={np.max(z_list):.2f}, mean={np.mean(z_list):.2f}')
    # 位移分析与保存
    displacements = []
    mags = []
    for pt_b, pt_a in zip(pts_b, pts_a):
        p3d_b = bilinear_interp(points_3d_b, pt_b)
        p3d_a = bilinear_interp(points_3d_a, pt_a)
        if np.isnan(p3d_b).any() or np.isnan(p3d_a).any():
            continue
        if not (np.isfinite(p3d_b).all() and np.isfinite(p3d_a).all()):
            continue
        if np.isinf(p3d_b[2]) or np.isinf(p3d_a[2]):
            continue
        disp = p3d_a - p3d_b
        mag = np.linalg.norm(disp)
        if mag == 0:
            continue
        displacements.append((pt_b, pt_a, disp, mag))
        mags.append(mag)
    # IQR法剔除离群点
    if mags:
        mags_np = np.array(mags)
        Q1 = np.percentile(mags_np, 25)
        Q3 = np.percentile(mags_np, 75)
        IQR = Q3 - Q1
        lower = Q1 - 0.3 * IQR
        upper = Q3 + 0.3 * IQR
        filtered = [(pt_b, pt_a, disp) for (pt_b, pt_a, disp, mag) in displacements if lower <= mag <= upper]
        # 3σ法进一步剔除
        if filtered:
            mags_f = np.array([np.linalg.norm(disp) for _, _, disp in filtered])
            mean_f = np.mean(mags_f)
            std_f = np.std(mags_f)
            lower_3sigma = mean_f - 1 * std_f
            upper_3sigma = mean_f + 1 * std_f
            filtered = [(pt_b, pt_a, disp) for (pt_b, pt_a, disp) in filtered if lower_3sigma <= np.linalg.norm(disp) <= upper_3sigma]
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
        median_disp = np.median(mags)
        with open(os.path.join(output_dir, f'displacement_stats_{idx}.txt'), 'w') as f:
            f.write(f'有效位移点数: {len(mags)}\n')
            f.write(f'平均位移: {np.mean(mags):.2f} mm\n')
            f.write(f'中位数位移: {median_disp:.2f} mm\n')
            f.write(f'最小位移: {np.min(mags):.2f} mm\n')
            f.write(f'最大位移: {np.max(mags):.2f} mm\n')
            f.write(f'位移标准差: {np.std(mags):.2f} mm\n')
            f.write(f'三维位移均值: dx={mean_disp[0]:.2f}, dy={mean_disp[1]:.2f}, dz={mean_disp[2]:.2f}\n')
            f.write(f'三维位移标准差: dx={std_disp[0]:.2f}, dy={std_disp[1]:.2f}, dz={std_disp[2]:.2f}\n')
    else:
        with open(os.path.join(output_dir, f'displacement_stats_{idx}.txt'), 'w') as f:
            f.write('无有效位移点\n') 