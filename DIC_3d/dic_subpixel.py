import sys
use_superres = False
if '--superres' in sys.argv:
    use_superres = True
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def zncc(template, search):
    template = (template - np.mean(template)) / (np.std(template) + 1e-8)
    search = (search - np.mean(search)) / (np.std(search) + 1e-8)
    return np.sum(template * search)

def parabola_subpixel(scores, coords):
    # 抛物线拟合，返回亚像素峰值位置
    if len(scores) < 3:
        return coords[np.argmax(scores)]
    idx = np.argmax(scores)
    if idx == 0 or idx == len(scores)-1:
        return coords[idx]
    x1, x2, x3 = coords[idx-1], coords[idx], coords[idx+1]
    y1, y2, y3 = scores[idx-1], scores[idx], scores[idx+1]
    denom = (x1-x2)*(x1-x3)*(x2-x3)
    if denom == 0:
        return x2
    A = (x3*(y2-y1) + x2*(y1-y3) + x1*(y3-y2)) / denom
    B = (x3**2*(y1-y2) + x2**2*(y3-y1) + x1**2*(y2-y3)) / denom
    if A == 0:
        return x2
    x_peak = -B/(2*A)
    return x_peak

def dic_subpixel(ref_img, def_img, pt, win_size=21, search_radius=5):
    x, y = int(pt[0]), int(pt[1])
    half = win_size // 2
    template = ref_img[y-half:y+half+1, x-half:x+half+1]
    best_score = -np.inf
    best_pos = (0, 0)
    scores = np.zeros((2*search_radius+1, 2*search_radius+1))
    for dy in range(-search_radius, search_radius+1):
        for dx in range(-search_radius, search_radius+1):
            y2, x2 = y+dy, x+dx
            patch = def_img[y2-half:y2+half+1, x2-half:x2+half+1]
            if patch.shape != template.shape:
                continue
            score = zncc(template, patch)
            scores[dy+search_radius, dx+search_radius] = score
            if score > best_score:
                best_score = score
                best_pos = (dx, dy)
    # 亚像素抛物线拟合（x方向）
    dx0, dy0 = best_pos
    x_scores = scores[search_radius+dy0, max(0,search_radius+dx0-1):min(2*search_radius+1,search_radius+dx0+2)]
    x_coords = np.arange(dx0-1, dx0+2)[:len(x_scores)]
    sub_dx = parabola_subpixel(x_scores, x_coords)
    y_scores = scores[max(0,search_radius+dy0-1):min(2*search_radius+1,search_radius+dy0+2), search_radius+dx0]
    y_coords = np.arange(dy0-1, dy0+2)[:len(y_scores)]
    sub_dy = parabola_subpixel(y_scores, y_coords)
    return (x+sub_dx, y+sub_dy)

def run_dic_grid(ref_img, def_img, grid_step=20, win_size=21, search_radius=5):
    h, w = ref_img.shape
    pts = []
    disps = []
    for y in range(win_size, h-win_size, grid_step):
        for x in range(win_size, w-win_size, grid_step):
            pt = (x, y)
            subpix = dic_subpixel(ref_img, def_img, pt, win_size, search_radius)
            pts.append(pt)
            disps.append((subpix[0]-x, subpix[1]-y))
    return np.array(pts), np.array(disps)

if __name__ == '__main__':
    param_dir = 'output_sr' if use_superres else 'output'
    params = np.load(os.path.join(param_dir, 'calib_params.npz'))
    mapLx, mapLy = params['mapLx'], params['mapLy']
    img1_raw = cv2.imread('../capture/Right/R_008.png', 0)
    img2_raw = cv2.imread('../capture/Right/R_009.png', 0)
    img1 = cv2.remap(img1_raw, mapLx, mapLy, cv2.INTER_LINEAR)
    img2 = cv2.remap(img2_raw, mapLx, mapLy, cv2.INTER_LINEAR)
    pts, disps = run_dic_grid(img1, img2, grid_step=20, win_size=21, search_radius=5)
    # 可视化位移场
    mag = np.linalg.norm(disps, axis=1)
    # 只画ROI区域
    roi_mask = (pts[:,0] > 480) & (pts[:,0] < 1500) & (pts[:,1] > 370) & (pts[:,1] < 1150)
    plt.figure(figsize=(10,8))
    plt.quiver(pts[roi_mask,0], pts[roi_mask,1], disps[roi_mask,0], disps[roi_mask,1], mag[roi_mask], angles='xy', scale_units='xy', scale=0.1, cmap='jet')
    plt.gca().invert_yaxis()
    plt.title('DIC亚像素位移场（ROI）')
    plt.xlabel('x'); plt.ylabel('y')
    plt.colorbar(label='位移模长')
    plt.tight_layout()
    plt.show()
    # 保存结果
    np.savez('output/dic_subpixel_result.npz', pts=pts, disps=disps) 