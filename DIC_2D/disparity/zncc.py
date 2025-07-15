import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import config

def zncc_score(patch1, patch2):
    mean1 = np.mean(patch1)
    mean2 = np.mean(patch2)
    std1 = np.std(patch1)
    std2 = np.std(patch2)
    if std1 == 0 or std2 == 0:
        return -1
    return np.mean((patch1 - mean1) * (patch2 - mean2)) / (std1 * std2)

def compute_disparity(rectify_data, cfg):
    output_dir = cfg.DISPARITY_OUTPUT
    os.makedirs(output_dir, exist_ok=True)
    mapLx = rectify_data['mapLx']
    mapLy = rectify_data['mapLy']
    mapRx = rectify_data['mapRx']
    mapRy = rectify_data['mapRy']
    roi1 = rectify_data['roi1']
    roi2 = rectify_data['roi2']
    left_list = sorted([os.path.join(cfg.LEFT_DIR, f) for f in os.listdir(cfg.LEFT_DIR) if f.endswith('.png')])
    right_list = sorted([os.path.join(cfg.RIGHT_DIR, f) for f in os.listdir(cfg.RIGHT_DIR) if f.endswith('.png')])
    disparities = []
    for idx, (left_path, right_path) in enumerate(zip(left_list, right_list)):
        imgL = cv2.imread(left_path)
        imgR = cv2.imread(right_path)
        rectL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)
        # ROI裁剪
        x1, y1, w1, h1 = [int(x) for x in roi1]
        x2, y2, w2, h2 = [int(x) for x in roi2]
        rectL_roi = rectL[y1:y1+h1, x1:x1+w1]
        rectR_roi = rectR[y2:y2+h2, x2:x2+w2]
        cv2.imwrite(os.path.join(output_dir, f'group{idx+1}_L_zncc.png'), rectL_roi)
        cv2.imwrite(os.path.join(output_dir, f'group{idx+1}_R_zncc.png'), rectR_roi)
        # 灰度
        grayL = cv2.cvtColor(rectL_roi, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectR_roi, cv2.COLOR_BGR2GRAY)
        max_disp = cfg.ZNCC_PARAMS['max_disp']
        win_size = cfg.ZNCC_PARAMS['win_size']
        disp = np.zeros_like(grayL, dtype=np.float32)
        half = win_size // 2
        for y in range(half, grayL.shape[0]-half):
            for x in range(half+max_disp, grayL.shape[1]-half):
                patch1 = grayL[y-half:y+half+1, x-half:x+half+1]
                best_score = -1
                best_d = 0
                for d in range(max_disp):
                    if x-half-d < 0:
                        continue
                    patch2 = grayR[y-half:y+half+1, x-half-d:x+half+1-d]
                    if patch2.shape != patch1.shape:
                        continue
                    score = zncc_score(patch1, patch2)
                    if score > best_score:
                        best_score = score
                        best_d = d
                disp[y, x] = best_d
        np.save(os.path.join(output_dir, f'disp_zncc_{idx+1}.npy'), disp)
        # 可视化视差图
        valid_disp = disp[(disp > 0) & (disp < max_disp)]
        vmax = np.percentile(valid_disp, 98) if valid_disp.size > 0 else max_disp
        plt.figure(figsize=(6, 5))
        plt.imshow(disp, cmap='jet', vmin=0, vmax=vmax)
        plt.title(f'ZNCC视差图-组{idx+1}')
        plt.colorbar()
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'disp_zncc_{idx+1}.png'))
        plt.close()
        disparities.append(disp)
    # 棋盘格标定图像视差图可视化
    calib_left = sorted([os.path.join(cfg.CALIB_LEFT_DIR, f) for f in os.listdir(cfg.CALIB_LEFT_DIR) if f.endswith('.png')])
    calib_right = sorted([os.path.join(cfg.CALIB_RIGHT_DIR, f) for f in os.listdir(cfg.CALIB_RIGHT_DIR) if f.endswith('.png')])
    if calib_left and calib_right:
        imgL = cv2.imread(calib_left[0])
        imgR = cv2.imread(calib_right[0])
        rectL = cv2.remap(cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY), mapLx, mapLy, cv2.INTER_LINEAR)
        rectR = cv2.remap(cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY), mapRx, mapRy, cv2.INTER_LINEAR)
        disp = np.zeros_like(rectL, dtype=np.float32)
        for y in range(half, rectL.shape[0]-half):
            for x in range(half+max_disp, rectL.shape[1]-half):
                patch1 = rectL[y-half:y+half+1, x-half:x+half+1]
                best_score = -1
                best_d = 0
                for d in range(max_disp):
                    if x-half-d < 0:
                        continue
                    patch2 = rectR[y-half:y+half+1, x-half-d:x+half+1-d]
                    if patch2.shape != patch1.shape:
                        continue
                    score = zncc_score(patch1, patch2)
                    if score > best_score:
                        best_score = score
                        best_d = d
                disp[y, x] = best_d
        valid_disp = disp[disp > 0]
        vmax = np.percentile(valid_disp, 98) if valid_disp.size > 0 else max_disp
        plt.figure(figsize=(6, 5))
        plt.imshow(disp, cmap='jet', vmin=0, vmax=vmax)
        plt.title('棋盘格标定图像视差图 (ZNCC)')
        plt.colorbar()
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'chessboard_disparity_map_zncc.png'))
        plt.close()
    return disparities 