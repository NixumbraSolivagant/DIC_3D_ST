import cv2
import numpy as np
import glob
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# 修正matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
import os

# ========== ZNCC+抛物线拟合亚像素位移函数 ==========
def zncc_score(patch1, patch2):
    mean1 = np.mean(patch1)
    mean2 = np.mean(patch2)
    std1 = np.std(patch1)
    std2 = np.std(patch2)
    if std1 == 0 or std2 == 0:
        return -1
    return np.mean((patch1 - mean1) * (patch2 - mean2)) / (std1 * std2)

def parabola(x, a, b, c):
    return a * x**2 + b * x + c

def zncc_subpixel_shift(ref_img, tgt_img, win_size=7, search_range=5):
    h, w = ref_img.shape
    half = win_size // 2
    disp_map = np.full((h, w), np.nan, dtype=np.float32)
    for y in range(half, h-half):
        for x in range(half+search_range, w-half-search_range):
            patch1 = ref_img[y-half:y+half+1, x-half:x+half+1]
            zncc_scores = []
            shifts = np.arange(-search_range, search_range+1)
            for dx in shifts:
                patch2 = tgt_img[y-half:y+half+1, x-half+dx:x+half+1+dx]
                score = zncc_score(patch1, patch2)
                zncc_scores.append(score)
            zncc_scores = np.array(zncc_scores)
            max_idx = np.argmax(zncc_scores)
            if 0 < max_idx < len(shifts)-1:
                x_fit = shifts[max_idx-1:max_idx+2]
                y_fit = zncc_scores[max_idx-1:max_idx+2]
                try:
                    popt, _ = curve_fit(parabola, x_fit, y_fit)
                    a, b, c = popt
                    if a != 0:
                        subpixel_shift = -b/(2*a)
                        disp_map[y, x] = subpixel_shift
                    else:
                        disp_map[y, x] = shifts[max_idx]
                except Exception:
                    disp_map[y, x] = shifts[max_idx]
            else:
                disp_map[y, x] = shifts[max_idx]
    return disp_map

# ========== 1. 相机标定和立体校正 ==========
CHECKERBOARD = (11, 8)
square_size = 5  # mm
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size
objpoints = []
imgpoints_left = []
imgpoints_right = []
output_fig_dir = 'output_figures'
os.makedirs(output_fig_dir, exist_ok=True)
# 修改为SR数据目录
images_left = sorted(glob.glob('../textR18/textL18_SR/*.png'))
images_right = sorted(glob.glob('../textL18_SR/*.png'))
# 1. 标定角点检测更严格，亚像素精度更高
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
for fname_left, fname_right in zip(images_left, images_right):
    imgL = cv2.imread(fname_left)
    imgR = cv2.imread(fname_right)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if retL:
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
    if retR:
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
    if retL and retR:
        objpoints.append(objp)
        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)
# 修正calibrateCamera参数
init_camera = np.eye(3, dtype=np.float64)
init_dist = np.zeros((5, 1), dtype=np.float64)
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpoints_left, grayL.shape[::-1], init_camera.copy(), init_dist.copy())
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpoints_right, grayR.shape[::-1], init_camera.copy(), init_dist.copy())
flags = 0
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtxL, distL, mtxR, distR, grayL.shape[::-1],
    criteria=criteria, flags=flags
)
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtxL, distL, mtxR, distR, grayL.shape[::-1], R, T, alpha=1
)
mapLx, mapLy = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, grayL.shape[::-1], cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, grayR.shape[::-1], cv2.CV_32FC1)
print("相机标定与立体校正完成。")

# ========== Q矩阵与基线自动检验 ==========
print("\n===== Q矩阵与基线参数检验 =====")
print("Q矩阵：\n", Q)
if Q.shape == (4, 4):
    f_q = Q[2, 3]
    inv_baseline = Q[3, 2]
    cx_q = -Q[0, 3]
    cy_q = -Q[1, 3]
    if inv_baseline != 0:
        baseline_q = -1.0 / inv_baseline
    else:
        baseline_q = float('inf')
    print(f"Q[2,3] (焦距f): {f_q:.4f}")
    print(f"Q[3,2] (-1/B): {inv_baseline:.6f}")
    print(f"Q主点(cx, cy): ({cx_q:.2f}, {cy_q:.2f})")
    print(f"Q矩阵推算基线B: {baseline_q:.4f} 单位与标定一致")
    if inv_baseline >= 0:
        print("[警告] Q[3,2] 应为负数，基线B应为正，当前Q[3,2]为正，可能有问题！")
    if not (10 < baseline_q < 1000):
        print("[警告] 基线长度B异常，通常应在10~1000单位之间，请检查标定参数！")
else:
    print("[警告] Q矩阵不是4x4，无法自动检验！")
# 通过T向量直接计算基线
baseline_T = np.linalg.norm(T)
print(f"T向量推算基线B: {baseline_T:.4f} 单位与标定一致")
if abs(baseline_T - baseline_q) > 1e-2:
    print(f"[警告] Q矩阵与T向量推算的基线不一致，差值: {abs(baseline_T - baseline_q):.4f}")
print("===== Q矩阵与基线参数检验结束 =====\n")

# ========== 立体校正评估 ==========
if len(imgpoints_left) > 0 and len(imgpoints_right) > 0:
    # 1. 极线几何误差
    def compute_epipolar_error(F, pts1, pts2):
        pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
        pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])
        l2 = (F @ pts1_h.T).T
        l1 = (F.T @ pts2_h.T).T
        d1 = np.abs(np.sum(l2 * pts2_h, axis=1)) / np.sqrt(l2[:, 0] ** 2 + l2[:, 1] ** 2)
        d2 = np.abs(np.sum(l1 * pts1_h, axis=1)) / np.sqrt(l1[:, 0] ** 2 + l1[:, 1] ** 2)
        return d1, d2


    pts1 = np.vstack([c.reshape(-1, 2) for c in imgpoints_left])
    pts2 = np.vstack([c.reshape(-1, 2) for c in imgpoints_right])
    d1, d2 = compute_epipolar_error(F, pts1, pts2)
    print(f"极线几何误差均值(左到右): {np.mean(d1):.4f} 像素")
    print(f"极线几何误差均值(右到左): {np.mean(d2):.4f} 像素")
    print(f"极线几何误差最大值(左到右): {np.max(d1):.4f} 像素")
    print(f"极线几何误差最大值(右到左): {np.max(d2):.4f} 像素")
    # 2. 校正后左右角点y坐标差
    cornersL_rect = cv2.undistortPoints(imgpoints_left[0], mtxL, distL, R=R1, P=P1).reshape(-1, 2)
    cornersR_rect = cv2.undistortPoints(imgpoints_right[0], mtxR, distR, R=R2, P=P2).reshape(-1, 2)
    y_diff = np.abs(cornersL_rect[:, 1] - cornersR_rect[:, 1])
    mean_y_diff = np.mean(y_diff)
    max_y_diff = np.max(y_diff)
    print(f"校正后左右角点y坐标差均值: {mean_y_diff:.4f} 像素，最大值: {max_y_diff:.4f} 像素")
    print("校正后y坐标差标准：均值 < 2 像素，最大值 < 5 像素为合格")
else:
    print("无法进行立体校正评估，缺少角点数据。")

# ========== 校正后图像SSIM评估与可视化 ==========
# 选取一组标定图像进行校正
# 已去除SSIM分数评估和可视化

# ========== 校正前后棋盘格图像可视化 ==========
if len(images_left) > 0 and len(images_right) > 0:
    imgL = cv2.imread(images_left[0])
    imgR = cv2.imread(images_right[0])
    rectL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)
    # 裁剪到有效区域，去除黑边
    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2
    rectL_cropped = rectL[y1:y1 + h1, x1:x1 + w1]
    rectR_cropped = rectR[y2:y2 + h2, x2:x2 + w2]
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB))
    plt.title('校正前左图')
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB))
    plt.title('校正前右图')
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(rectL_cropped, cv2.COLOR_BGR2RGB))
    plt.title('校正后左图(有效区域)')
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(rectR_cropped, cv2.COLOR_BGR2RGB))
    plt.title('校正后右图(有效区域)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_fig_dir, 'calibration_rectified_images.png'))
    plt.close()

    # 拼合校正后左右图像并画极线
    h1, w1 = rectL_cropped.shape[:2]
    h2, w2 = rectR_cropped.shape[:2]
    h = min(h1, h2)
    rectL_cropped = rectL_cropped[:h, :]
    rectR_cropped = rectR_cropped[:h, :]
    concat_img = np.hstack((rectL_cropped, rectR_cropped))
    interval = 20  # 极线间隔像素
    for y in range(interval, h, interval):
        cv2.line(concat_img, (0, y), (w1 + w2, y), (0, 0, 255), 1)  # 红色极线
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(concat_img, cv2.COLOR_BGR2RGB))
    plt.title('校正后左右图像拼合及极线')
    plt.axis('off')
    plt.savefig(os.path.join(output_fig_dir, 'rectified_images_concat.png'))
    plt.close()

# ========== 棋盘格角点检测可视化 ==========
if len(images_left) > 0 and len(images_right) > 0:
    imgL_vis = cv2.imread(images_left[0]).copy()
    imgR_vis = cv2.imread(images_right[0]).copy()
    grayL_vis = cv2.cvtColor(imgL_vis, cv2.COLOR_BGR2GRAY)
    grayR_vis = cv2.cvtColor(imgR_vis, cv2.COLOR_BGR2GRAY)
    retL_vis, cornersL_vis = cv2.findChessboardCorners(grayL_vis, CHECKERBOARD, None)
    retR_vis, cornersR_vis = cv2.findChessboardCorners(grayR_vis, CHECKERBOARD, None)
    if retL_vis:
        cv2.drawChessboardCorners(imgL_vis, CHECKERBOARD, cornersL_vis, retL_vis)
    if retR_vis:
        cv2.drawChessboardCorners(imgR_vis, CHECKERBOARD, cornersR_vis, retR_vis)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(imgL_vis, cv2.COLOR_BGR2RGB))
    plt.title('左图角点检测')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(imgR_vis, cv2.COLOR_BGR2RGB))
    plt.title('右图角点检测')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_fig_dir, 'chessboard_corner_detection.png'))
    plt.close()

# ========== 重投影RMS误差折线图 ==========
rms_errors = []
for i in range(len(objpoints)):
    imgpoints_proj, _ = cv2.projectPoints(objpoints[i], rvecsL[i], tvecsL[i], mtxL, distL)
    imgpoints_proj = imgpoints_proj.reshape(-1, 2)
    imgpoints_detected = imgpoints_left[i].reshape(-1, 2)
    error = np.linalg.norm(imgpoints_proj - imgpoints_detected, axis=1)
    rms = np.sqrt(np.mean(error ** 2))
    rms_errors.append(rms)
plt.figure(figsize=(8, 4))
plt.plot(rms_errors, 'o-', label='重投影RMS误差')
plt.xlabel('图片编号')
plt.ylabel('重投影RMS误差 (像素)')
plt.title('每张标定图片的重投影RMS误差')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_fig_dir, 'reprojection_rms_errors.png'))
plt.close()

# ========== 校正后极线对齐误差可视化 ==========
if len(imgpoints_left) > 0 and len(imgpoints_right) > 0:
    cornersL_rect = cv2.undistortPoints(imgpoints_left[0], mtxL, distL, R=R1, P=P1).reshape(-1, 2)
    cornersR_rect = cv2.undistortPoints(imgpoints_right[0], mtxR, distR, R=R2, P=P2).reshape(-1, 2)
    yL = cornersL_rect[:, 1]
    yR = cornersR_rect[:, 1]
    plt.figure(figsize=(8, 4))
    plt.plot(yL, 'ro-', label='左图校正后角点y')   # 红色圆点线
    plt.plot(yR, 'bs-', label='右图校正后角点y')   # 蓝色方块线
    plt.plot(np.abs(yL - yR), 'g.-', label='y坐标差')
    plt.legend()
    plt.title('校正后极线对齐误差（y坐标）')
    plt.xlabel('角点序号')
    plt.ylabel('像素')
    plt.tight_layout()
    plt.savefig(os.path.join(output_fig_dir, 'epipolar_alignment_error.png'))
    plt.close()

# 删除auto_fix_path函数和所有input相关、单张图片分析流程
# 保留批量自动读取和分析图片对的主流程

# ========== 自动批量滑动窗口配对图片对 ==========
# 自动批量滑动窗口配对图片对
left_list = sorted(glob.glob('../capture_SR/Left/*.png'))
right_list = sorted(glob.glob('../capture_SR/Right/*.png'))


# ========== 提前输出单个像素的物理尺寸 ==========
if 'mtxL' in locals():
    fx = mtxL[0, 0]
    fy = mtxL[1, 1]
    if len(imgpoints_left) > 0:
        pts = imgpoints_left[0].reshape(-1, 2)
        row0 = pts[:CHECKERBOARD[0], 0]
        px_per_square = np.abs(np.mean(np.diff(row0)))
        pixel_size_mm = square_size / px_per_square
        pixel_size_um = pixel_size_mm * 1000
        print(f'单个像素物理尺寸: {pixel_size_mm:.5f} mm, {pixel_size_um:.2f} μm')
    else:
        pixel_size_mm = None
        print('无法计算像素物理尺寸，缺少角点数据。')
else:
    pixel_size_mm = None
    print('无法计算像素物理尺寸，缺少相机内参。')

# ========== 诊断：可视化每组校正后左右图像（ROI交集） ==========
for idx in range(len(left_list)):
    imgL = cv2.imread(left_list[idx])
    imgR = cv2.imread(right_list[idx])
    if imgL is None or imgR is None:
        print(f'第{idx+1}组有图片无法读取，跳过可视化！')
        continue
    rectL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)
    # ROI交集
    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2
    w = min(w1, w2)
    h = min(h1, h2)
    rectL_cropped = rectL[y1:y1+h, x1:x1+w]
    rectR_cropped = rectR[y2:y2+h, x2:x2+w]


# ========== 棋盘格标定图像视差图可视化（自动优化） ==========
if len(images_left) > 0 and len(images_right) > 0:
    imgL = cv2.imread(images_left[0])
    imgR = cv2.imread(images_right[0])
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    # 校正
    rectL = cv2.remap(grayL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR = cv2.remap(grayR, mapRx, mapRy, cv2.INTER_LINEAR)
    # 自动优化SGBM参数
    min_disp = 0
    num_disp = 128  # 增大视差范围
    block_size = 5  # 更小窗口
    # 兼容低版本OpenCV
    StereoSGBM_create = getattr(cv2, 'StereoSGBM_create', None)
    StereoBM_create = getattr(cv2, 'StereoBM_create', None)
    if StereoSGBM_create is not None:
        sgbm = StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 1 * block_size ** 2,
            P2=32 * 1 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=5,
            speckleWindowSize=50,
            speckleRange=16,
            mode=getattr(cv2, 'STEREO_SGBM_MODE_SGBM_3WAY', 0)
        )
    elif StereoBM_create is not None:
        sgbm = StereoBM_create(numDisparities=num_disp, blockSize=block_size)
    else:
        raise RuntimeError('当前OpenCV不支持SGBM或BM立体匹配')
    disp = sgbm.compute(rectL, rectR).astype(np.float32) / 16.0
    disp[disp < min_disp] = 0
    # 只显示有效视差区间
    valid_disp = disp[disp > 0]
    vmax = np.percentile(valid_disp, 98) if valid_disp.size > 0 else num_disp
    plt.figure(figsize=(6, 5))
    plt.imshow(disp, cmap='jet', vmin=min_disp, vmax=vmax)
    plt.title('棋盘格标定图像视差图 (SGBM, 优化)')
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_fig_dir, 'chessboard_disparity_map.png'))
    plt.close()
    # 自动检测异常
    if valid_disp.size == 0 or np.std(valid_disp) < 1.0:
        print('[警告] 棋盘格视差图无有效数据或过于模糊，请检查标定图像和参数！')

# ========== OpenCV并行加速ZNCC相关法视差计算 ==========
def zncc_disparity_opencv(left, right, max_disp=64, win_size=7, step=4, zncc_thresh=0.1):
    h, w = left.shape
    disp = np.zeros_like(left, dtype=np.float32)
    half = win_size // 2
    tpl_len = w - win_size + 1
    valid_x_start = half + max_disp
    valid_x_end = min(w - half, tpl_len + max_disp)
    for y in range(half, h-half, step):
        tpl = left[y-half:y+half+1, :]
        best_score = np.full(w, -1.0, dtype=np.float32)
        best_d = np.zeros(w, dtype=np.float32)
        for d in range(max_disp):
            shifted = np.zeros_like(right)
            if d > 0:
                shifted[:, d:] = right[:, :-d]
            else:
                shifted = right.copy()
            res = cv2.matchTemplate(shifted[y-half:y+half+1, :], tpl, cv2.TM_CCOEFF_NORMED)
            res_1d = res[0]
            x_range = np.arange(valid_x_start, valid_x_end)
            for i, x in enumerate(x_range):
                if i < len(res_1d) and res_1d[i] > best_score[x]:
                    best_score[x] = res_1d[i]
                    best_d[x] = d
        # 阈值处理：分数低于zncc_thresh的像素视差设为0
        for x in x_range:
            if best_score[x] < zncc_thresh:
                best_d[x] = 0
        disp[y, valid_x_start:valid_x_end] = best_d[valid_x_start:valid_x_end]
    return disp

# ========== SGBM视差计算函数 ==========
def sgbm_disparity(left, right, min_disp=1, num_disp=128, block_size=3):
    StereoSGBM_create = getattr(cv2, 'StereoSGBM_create', None)
    if StereoSGBM_create is not None:
        sgbm = StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 1 * block_size ** 2,
            P2=32 * 1 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=50,
            speckleRange=8,
            mode=getattr(cv2, 'STEREO_SGBM_MODE_SGBM_3WAY', 0)
        )
    else:
        raise RuntimeError('当前OpenCV不支持SGBM立体匹配')
    disp = sgbm.compute(left, right).astype(np.float32) / 16.0
    disp[disp < min_disp] = 0
    disp[disp > (min_disp + num_disp - 1)] = 0
    return disp

min_disp = 0
max_disp = 64

# ========== 优化后的三维位移分析完整流程 ==========
output_fig_dir = 'output_figures'
os.makedirs(output_fig_dir, exist_ok=True)
# 新增输出子文件夹
corrected_dir = os.path.join(output_fig_dir, 'corrected_pairs')
uncertainty_dir = os.path.join(output_fig_dir, 'uncertainty_maps')
os.makedirs(corrected_dir, exist_ok=True)
os.makedirs(uncertainty_dir, exist_ok=True)

# ========== 新增：相机坐标系到物体坐标系的旋转变换 ==========
# 用rvecsL[0]和tvecsL[0]（左相机第1张标定板外参）
R_world2cam, _ = cv2.Rodrigues(rvecsL[0])  # 世界到相机
R_cam2world = R_world2cam.T  # 相机到世界
# 删除物体坐标系x轴在相机坐标系下的方向、夹角的计算和输出

# ========== 新增：用户输入实际物理位移 ===========
# 支持输入x, y, z分量或模长，输入格式如：1.0,0.0,0.0 或 1.0
# 改为命令行输入
actual_disp_str = input('请输入实际物理位移（mm），格式如1.0,0.0,0.0 或 1.0：')
if actual_disp_str is None:
    print('未输入实际物理位移，程序终止。')
    exit()
actual_disp_str = actual_disp_str.strip().replace('，', ',')  # 替换中文逗号为英文逗号
try:
    if ',' in actual_disp_str:
        parts = [x.strip() for x in actual_disp_str.split(',')]
        if len(parts) != 3:
            raise ValueError('输入格式错误，需为3个分量（如1.0,0.0,0.0）')
        actual_disp = np.array([float(x) for x in parts])
        actual_disp_mag = np.linalg.norm(actual_disp)
    else:
        actual_disp = None
        actual_disp_mag = float(actual_disp_str)
except Exception as e:
    print(f'输入格式错误：{e}')
    exit()

# ========== 统计每组位移均值与误差 ===========
est_x_list = []
est_y_list = []
est_z_list = []
est_mag_list = []
err_x_list = []
err_y_list = []
err_z_list = []
err_mag_list = []
group_idx_list = []

mean_disp_list = []  # 自动化输出每组像素位移
mean_disp_mm_list = []  # 自动化输出每组物理位移

for idx in range(len(left_list) - 1):
    left_before = left_list[idx]
    right_before = right_list[idx]
    left_after = left_list[idx + 1]
    right_after = right_list[idx + 1]
    print(f'正在处理第{idx + 1}组:')
    print('左前:', left_before)
    print('右前:', right_before)
    print('左后:', left_after)
    print('右后:', right_after)
    imgL_before = cv2.imread(left_before)
    imgR_before = cv2.imread(right_before)
    imgL_after = cv2.imread(left_after)
    imgR_after = cv2.imread(right_after)
    if any(x is None for x in [imgL_before, imgR_before, imgL_after, imgR_after]):
        print('有图片无法读取，跳过本组！')
        continue

    # ======= 稠密三维位移分析（自动修正，基于稠密光流） =======
    # 1. 立体校正
    rectL_before = cv2.remap(imgL_before, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR_before = cv2.remap(imgR_before, mapRx, mapRy, cv2.INTER_LINEAR)
    rectL_after = cv2.remap(imgL_after, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR_after = cv2.remap(imgR_after, mapRx, mapRy, cv2.INTER_LINEAR)

    # 新增：保存校正后的散斑图（自动裁剪有效ROI去黑边）
    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2
    rectL_before_roi = rectL_before[y1:y1+h1, x1:x1+w1]
    rectR_before_roi = rectR_before[y2:y2+h2, x2:x2+w2]
    rectL_after_roi = rectL_after[y1:y1+h1, x1:x1+w1]
    rectR_after_roi = rectR_after[y2:y2+h2, x2:x2+w2]
    cv2.imwrite(os.path.join(corrected_dir, f'group{idx+1}_L_before.png'), rectL_before_roi)
    cv2.imwrite(os.path.join(corrected_dir, f'group{idx+1}_R_before.png'), rectR_before_roi)
    cv2.imwrite(os.path.join(corrected_dir, f'group{idx+1}_L_after.png'), rectL_after_roi)
    cv2.imwrite(os.path.join(corrected_dir, f'group{idx+1}_R_after.png'), rectR_after_roi)

    # 2. 灰度化
    grayL_b = cv2.cvtColor(rectL_before, cv2.COLOR_BGR2GRAY)
    grayR_b = cv2.cvtColor(rectR_before, cv2.COLOR_BGR2GRAY)
    grayL_a = cv2.cvtColor(rectL_after, cv2.COLOR_BGR2GRAY)
    grayR_a = cv2.cvtColor(rectR_after, cv2.COLOR_BGR2GRAY)

    # 选择视差计算方法：'SGBM'、'BM'、'ZNCC'
    method = 'SGBM'  # 可选 'SGBM', 'BM', 'ZNCC'

    # 3. 计算视差图（自动切换方法）
    if method == 'SGBM':
        sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=256,
            blockSize=3,
            P1=8 * 2 * 5 ** 2,
            P2=32 * 1 * 5 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=150,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        disp_b = sgbm.compute(grayL_b, grayR_b).astype(np.float32) / 16.0
        disp_a = sgbm.compute(grayL_a, grayR_a).astype(np.float32) / 16.0
        valid_b = (disp_b > 0) & (disp_b < 128)
        valid_a = (disp_a > 0) & (disp_a < 128)
        disp_b_norm = disp_b  # SGBM直接用原始视差
        disp_a_norm = disp_a
        vmax = np.percentile(disp_b[valid_b], 98) if np.any(valid_b) else 128
    elif method == 'BM':
        if hasattr(cv2, 'StereoBM_create'):
            bm = cv2.StereoBM_create(numDisparities=16*5, blockSize=21)
        else:
            raise RuntimeError('当前OpenCV不支持StereoBM_create')
        disp_b = bm.compute(grayL_b, grayR_b).astype(np.float32)
        disp_a = bm.compute(grayL_a, grayR_a).astype(np.float32)
        valid_b = (disp_b > 0) & (disp_b < 16*5)
        valid_a = (disp_a > 0) & (disp_a < 16*5)
        disp_b_norm = np.empty_like(disp_b, dtype=np.uint8)
        disp_a_norm = np.empty_like(disp_a, dtype=np.uint8)
        cv2.normalize(disp_b, disp_b_norm, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.normalize(disp_a, disp_a_norm, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        vmax = 255
    elif method == 'ZNCC':
        disp_b = zncc_disparity_opencv(grayL_b, grayR_b, max_disp=64, win_size=7, step=2, zncc_thresh=0.1)
        disp_a = zncc_disparity_opencv(grayL_a, grayR_a, max_disp=64, win_size=7, step=2, zncc_thresh=0.1)
        valid_b = (disp_b > 0) & (disp_b < 64)
        valid_a = (disp_a > 0) & (disp_a < 64)
        disp_b_norm = np.empty_like(disp_b, dtype=np.uint8)
        disp_a_norm = np.empty_like(disp_a, dtype=np.uint8)
        cv2.normalize(disp_b, disp_b_norm, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.normalize(disp_a, disp_a_norm, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        vmax = 255
    else:
        raise ValueError('未知视差计算方法')
    total_px = disp_b.size
    print(f'视差图 before 有效点数: {np.count_nonzero(valid_b)} / {total_px} ({np.count_nonzero(valid_b)/total_px:.2%})')
    print(f'视差图 after  有效点数: {np.count_nonzero(valid_a)} / {total_px} ({np.count_nonzero(valid_a)/total_px:.2%})')

    # 剔除视差图中的异常值，仅保留经验阈值区间
    valid_low = 1
    valid_high = 60
    disp_b_filtered = np.where((disp_b > valid_low) & (disp_b < valid_high), disp_b, np.nan)
    disp_a_filtered = np.where((disp_a > valid_low) & (disp_a < valid_high), disp_a, np.nan)

    # ========== 输出并保存剔除异常值后的散斑视差图 ========== 
    plt.figure(figsize=(6, 5))
    plt.imshow(disp_b_filtered, cmap='jet', vmin=valid_low, vmax=valid_high)
    plt.title(f'经验阈值法剔除异常值后散斑视差图（{method}）- 组{idx+1}')
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_fig_dir, f'filtered_disparity_map_group{idx+1}.png'))
    plt.close()

    # 可视化校正前散斑视差图
    if method == 'SGBM':
        disp_uncorrected = sgbm.compute(grayL_b, grayR_b).astype(np.float32) / 16.0
        disp_uncorrected_norm = disp_uncorrected
        vmax_uncorrected = np.percentile(disp_uncorrected[disp_uncorrected > 0], 98) if np.any(disp_uncorrected > 0) else 128
    elif method == 'BM':
        disp_uncorrected = bm.compute(grayL_b, grayR_b).astype(np.float32)
        disp_uncorrected_norm = np.empty_like(disp_uncorrected, dtype=np.uint8)
        cv2.normalize(disp_uncorrected, disp_uncorrected_norm, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        vmax_uncorrected = 255
    elif method == 'ZNCC':
        disp_uncorrected = zncc_disparity_opencv(grayL_b, grayR_b, max_disp=64, win_size=7, step=2, zncc_thresh=0.1)
        disp_uncorrected_norm = np.empty_like(disp_uncorrected, dtype=np.uint8)
        cv2.normalize(disp_uncorrected, disp_uncorrected_norm, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        vmax_uncorrected = 255
    else:
        raise ValueError('未知视差计算方法')
    plt.figure(figsize=(6, 5))
    plt.imshow(disp_uncorrected_norm, cmap='jet', vmin=0, vmax=vmax_uncorrected)
    plt.title(f'校正前散斑视差图（{method}）- 组{idx+1}')
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_fig_dir, f'uncorrected_disparity_map_group{idx+1}.png'))
    plt.close()

    # 3. Farneback光流参数优化
    print('正在计算ZNCC相关法视差...')
    dx_map = zncc_disparity_opencv(grayL_b, grayL_a, max_disp=5, win_size=7, step=2, zncc_thresh=0.1)
    # dy_map = np.zeros_like(dx_map)  # 如需y方向亚像素位移，可扩展实现
    # dz_map = disp_a - disp_b

    # 自动修正：定义有效掩码
    valid_mask = valid_b  # 以before视差有效区为主
    dx_valid = dx_map[valid_mask]


    # ========== 异常点剔除方法集成 ==========
    def remove_outliers(data, method='iqr', threshold=3):
        data = np.asarray(data)
        if method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            return (data >= lower) & (data <= upper)
        elif method == 'zscore':
            from scipy.stats import zscore
            data_1d = np.asarray(data, dtype=np.float64).flatten()
            valid_mask = np.isfinite(data_1d)
            mask = np.zeros(data_1d.shape, dtype=bool)
            if np.any(valid_mask):
                valid_data = np.asarray(np.copy(data_1d[valid_mask]).astype(np.float64).flatten(), dtype=np.float64)
                valid_z_scores = np.abs(zscore(valid_data, nan_policy='omit'))
                mask_indices = np.where(valid_mask)[0]
                mask[mask_indices] = valid_z_scores < threshold
            if data.shape != data_1d.shape:
                mask = mask.reshape(data.shape)
            return mask
        elif method == 'mad':
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            return np.abs(data - median) < threshold * mad
        elif method == 'percentile':
            lower = np.percentile(data, 2.5)
            upper = np.percentile(data, 97.5)
            return (data >= lower) & (data <= upper)
        else:
            raise ValueError('Unknown method')

    # ========== 主流程中异常点剔除 ========== 
    # 可切换方法：'iqr', 'zscore', 'mad', 'percentile'
    outlier_method = 'zscore'  # 可选: 'iqr', 'zscore', 'mad', 'percentile'
    outlier_threshold = 3   # IQR/zscore/mad的阈值，percentile法忽略


    # 视差直方图
    # 剔除无效点，仅统计有效视差区间
    valid_disp_b = disp_b[(disp_b > valid_low) & (disp_b < valid_high)]
    valid_disp_a = disp_a[(disp_a > valid_low) & (disp_a < valid_high)]
    plt.figure()
    plt.hist(valid_disp_b.ravel(), bins=50, range=(valid_low, valid_high), color='b', alpha=0.5, label='disp_b')
    plt.hist(valid_disp_a.ravel(), bins=50, range=(valid_low, valid_high), color='r', alpha=0.5, label='disp_a')
    plt.title('有效视差分布直方图')
    plt.xlabel('disparity')
    plt.ylabel('count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_fig_dir, f'valid_disparity_hist_group{idx+1}.png'))
    plt.close()

    # 先人工查看分布
    # 经验阈值筛选有效视差
    valid_low = 1    # 可根据实际情况调整
    valid_high = 60  # 可根据实际情况调整
    valid_mask = (disp_b > valid_low) & (disp_b < valid_high) & np.isfinite(disp_b)

    # 计算有效点的误差统计
    valid_disp_values = disp_b[valid_mask]
    if valid_disp_values.size > 0:
        mean_disp = np.mean(valid_disp_values)
        std_disp = np.std(valid_disp_values)
        min_disp = np.min(valid_disp_values)
        max_disp = np.max(valid_disp_values)
        print(f'有效视差点数: {valid_disp_values.size}, 均值: {mean_disp:.4f}, 标准差: {std_disp:.4f}, 最小值: {min_disp:.4f}, 最大值: {max_disp:.4f}')
        # 输出平面2D位移（像素）
        print(f'组{idx+1} 平面2D位移（像素）: {mean_disp:.4f}')
        mean_disp_list.append(mean_disp)
        # 输出平面2D位移（物理量）
        if pixel_size_mm is not None:
            mean_disp_mm = mean_disp * pixel_size_mm
            print(f'组{idx+1} 平面2D位移: {mean_disp_mm:.4f} mm')
            mean_disp_mm_list.append(mean_disp_mm)
        else:
            mean_disp_mm_list.append(np.nan)
    else:
        print('无有效视差点用于误差统计')
        mean_disp_list.append(np.nan)
        mean_disp_mm_list.append(np.nan)

    disp_b_masked = np.full_like(disp_b, np.nan)
    disp_b_masked[valid_mask] = disp_b[valid_mask]
    plt.figure(figsize=(6, 5))
    plt.imshow(disp_b_masked, cmap='jet', vmin=valid_low, vmax=valid_high)
    plt.title(f'有效区域散斑视差图（SGBM）- 组{idx+1}')
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_fig_dir, f'valid_disparity_map_group{idx+1}.png'))
    plt.close()

    print(f'第{idx + 1}组处理完成。')

# ========== 输出单个像素的物理尺寸 ==========
# 取左相机内参mtxL，fx为焦距（像素），假设标定板实际方格尺寸已知
if 'mtxL' in locals():
    fx = mtxL[0, 0]
    fy = mtxL[1, 1]
    # 假设标定板方格实际尺寸 square_size，棋盘格宽度像素数 px_per_square
    # 取第一张标定图像角点x坐标间距
    if len(imgpoints_left) > 0:
        pts = imgpoints_left[0].reshape(-1, 2)
        # 取第一行角点x坐标
        row0 = pts[:CHECKERBOARD[0], 0]
        # 保证递增顺序
        if row0[-1] < row0[0]:
            row0 = row0[::-1]
        px_per_square = np.abs(np.mean(np.diff(row0)))
        pixel_size_mm = square_size / px_per_square
        pixel_size_um = pixel_size_mm * 1000
        print(f'单个像素物理尺寸: {pixel_size_mm:.6f} mm, {pixel_size_um:.2f} μm')
    else:
        print('无法计算像素物理尺寸，缺少角点数据。')
else:
    print('无法计算像素物理尺寸，缺少相机内参。')

# ====== 自动保存2D位移结果并可视化曲线 ======
np.savetxt(os.path.join(output_fig_dir, '2d_displacement_pixel.txt'), mean_disp_list, fmt='%.4f')
np.savetxt(os.path.join(output_fig_dir, '2d_displacement_mm.txt'), mean_disp_mm_list, fmt='%.6f')
plt.figure(figsize=(8, 5))
plt.plot(mean_disp_list, 'o-', label='像素位移')
if pixel_size_mm is not None:
    plt.plot(mean_disp_mm_list, 's-', label='物理位移 (mm)')
plt.xlabel('组编号')
plt.ylabel('位移')
plt.title('2D平面位移随组变化曲线')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_fig_dir, '2d_displacement_curve.png'))
plt.close()
