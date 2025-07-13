import cv2
import numpy as np
import glob
import os
import sys

use_superres = False  # 切换是否用超分

if '--superres' in sys.argv:
    use_superres = True

if use_superres:
    left_list = sorted(glob.glob('../capture_SR/Left/*.png'))
    disp_dir = 'output_sr/output_disp'
    output_dir = 'output_sr/output_match'
    os.makedirs('output_sr', exist_ok=True)
else:
    left_list = sorted(glob.glob('../capture/Left/*.png'))
    disp_dir = 'output/output_disp'
    output_dir = 'output/output_match'
    os.makedirs('output', exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

if hasattr(cv2, 'AKAZE_create'):
    detector = cv2.AKAZE_create()
else:
    print('警告：未检测到AKAZE，自动降级为ORB，建议安装opencv-contrib-python以获得更好特征效果。')
    detector = cv2.ORB_create()

def get_subpixel_value(img, pt):
    x, y = pt
    x0, y0 = int(x), int(y)
    dx, dy = x - x0, y - y0
    if 0 <= x0 < img.shape[1]-1 and 0 <= y0 < img.shape[0]-1:
        val = (img[y0, x0] * (1-dx)*(1-dy) +
               img[y0, x0+1] * dx*(1-dy) +
               img[y0+1, x0] * (1-dx)*dy +
               img[y0+1, x0+1] * dx*dy)
        return val
    else:
        return img[y0, x0]

for idx in range(len(left_list)-1):
    img1 = cv2.imread(left_list[idx], 0)
    img2 = cv2.imread(left_list[idx+1], 0)
    disp1 = np.load(os.path.join(disp_dir, f'disp_{idx+1}.npy'))
    disp2 = np.load(os.path.join(disp_dir, f'disp_{idx+2}.npy'))
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    # 只保留落在有效视差区的特征点
    valid_kp1, valid_des1 = [], []
    for k, d in zip(kp1, des1):
        disp_val = get_subpixel_value(disp1, k.pt)
        if 1 < disp_val < 120:
            valid_kp1.append(k)
            valid_des1.append(d)
    valid_kp2, valid_des2 = [], []
    for k, d in zip(kp2, des2):
        disp_val = get_subpixel_value(disp2, k.pt)
        if 1 < disp_val < 120:
            valid_kp2.append(k)
            valid_des2.append(d)
    if not valid_kp1 or not valid_kp2:
        print(f'第{idx+1}组无有效视差区特征点，跳过。')
        continue
    valid_des1 = np.array(valid_des1)
    valid_des2 = np.array(valid_des2)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(valid_des1, valid_des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    # 修复drawMatches outImg参数
    outImg = np.zeros((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    match_img = cv2.drawMatches(img1, valid_kp1, img2, valid_kp2, good, outImg, flags=2)
    cv2.imwrite(os.path.join(output_dir, f'match_{idx+1}.png'), match_img)
    # 保存匹配点坐标
    pts1 = np.array([valid_kp1[m.queryIdx].pt for m in good], dtype=np.float32)
    pts2 = np.array([valid_kp2[m.trainIdx].pt for m in good], dtype=np.float32)
    np.savez(os.path.join(output_dir, f'points_{idx+1}.npz'), pts1=pts1, pts2=pts2)
print('特征点匹配及坐标已保存到 output_match/') 