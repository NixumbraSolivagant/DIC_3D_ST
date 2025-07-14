import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import zscore
import config

def remove_outliers(data, method='zscore', threshold=3):
    data = np.asarray(data)
    if method == 'iqr':
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        return (data >= lower) & (data <= upper)
    elif method == 'zscore':
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

def analyze(disparities, cfg):
    output_dir = cfg.ANALYSIS_OUTPUT
    os.makedirs(output_dir, exist_ok=True)
    mean_disp_list = []
    mean_disp_mm_list = []
    for idx, disp in enumerate(disparities):
        valid_mask = (disp > cfg.DISP_VALID_LOW) & (disp < cfg.DISP_VALID_HIGH) & np.isfinite(disp)
        valid_disp = disp[valid_mask]
        # 异常点剔除
        outlier_mask = remove_outliers(valid_disp, method=cfg.OUTLIER_METHOD, threshold=cfg.OUTLIER_THRESHOLD)
        filtered_disp = valid_disp[outlier_mask]
        # 统计
        if filtered_disp.size > 0:
            mean_disp = np.mean(filtered_disp)
            std_disp = np.std(filtered_disp)
            min_disp = np.min(filtered_disp)
            max_disp = np.max(filtered_disp)
        else:
            mean_disp = np.nan
            std_disp = np.nan
            min_disp = np.nan
            max_disp = np.nan
        np.savetxt(os.path.join(output_dir, f'disp_stats_{idx+1}.txt'), [mean_disp, std_disp, min_disp, max_disp])
        mean_disp_list.append(mean_disp)
        # 物理位移
        pixel_size_mm = getattr(cfg, 'pixel_size_mm', None)
        if pixel_size_mm is None and hasattr(cfg, 'calib_data'):
            pixel_size_mm = cfg.calib_data.get('pixel_size_mm', None)
        if pixel_size_mm is not None:
            mean_disp_mm = mean_disp * pixel_size_mm
        else:
            mean_disp_mm = np.nan
        mean_disp_mm_list.append(mean_disp_mm)
        # 视差直方图
        plt.figure()
        plt.hist(valid_disp.ravel(), bins=50, range=(cfg.DISP_VALID_LOW, cfg.DISP_VALID_HIGH), color='b', alpha=0.5, label='disp')
        plt.title(f'有效视差分布直方图-组{idx+1}')
        plt.xlabel('disparity')
        plt.ylabel('count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'valid_disparity_hist_group{idx+1}.png'))
        plt.close()
    # 保存所有组的2D位移
    np.savetxt(os.path.join(output_dir, '2d_displacement_pixel.txt'), np.array(mean_disp_list), fmt='%.4f')
    np.savetxt(os.path.join(output_dir, '2d_displacement_mm.txt'), np.array(mean_disp_mm_list), fmt='%.6f')
    # 返回结果
    results = []
    for i, (mean, std) in enumerate(zip(mean_disp_list, mean_disp_mm_list)):
        results.append({'mean': mean, 'std': std, 'idx': i+1})
    return results 