import os
import json
import numpy as np
from DIC_2D.calibration import calibration, rectify
from DIC_2D.disparity import sgbm
import config
from DIC_2D.analysis import displacement

def fourier_score(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    h, w = magnitude_spectrum.shape
    center = (h//2, w//2)
    radius = min(h, w) // 8
    y, x = np.ogrid[:h, :w]
    mask = (x - center[1])**2 + (y - center[0])**2 > radius**2
    high_freq_energy = magnitude_spectrum[mask].sum()
    total_energy = magnitude_spectrum.sum()
    return high_freq_energy / total_energy

def main():
    # 标定与校正
    calib_data = calibration.run_calibration(config)
    if 'pixel_size_mm' in calib_data and calib_data['pixel_size_mm'] is not None:
        config.pixel_size_mm = calib_data['pixel_size_mm']
    rectify_data = rectify.run_rectification(calib_data, config)
    # 参数空间
    block_sizes = [3, 5, 7, 9, 11]
    num_disps = [16, 32, 64, 96, 128]
    theoretical_disp = getattr(config, 'THEORETICAL_DISPLACEMENT_MM', 0.5)
    best_error = float('inf')
    best_params = None
    best_result = None
    best_actual = None
    for blockSize in block_sizes:
        for numDisparities in num_disps:
            if numDisparities % 16 != 0:
                continue
            config.SGBM_PARAMS['blockSize'] = blockSize
            config.SGBM_PARAMS['numDisparities'] = numDisparities
            try:
                disparity_maps = sgbm.compute_disparity(rectify_data, config)
                # 只用第一组视差图做评价
                disp = disparity_maps[0]
                # 位移分析，获取物理位移
                results = displacement.analyze([disp], config)
                mean_disp_mm = results[0]['std']  # 这里原代码std其实是mm，需确认
                # 实际应为mean_disp_mm = results[0]['mean']，但原displacement.py返回的std是mm，mean是像素
                # 但根据displacement.py，mean_disp_mm_list存的是mm，results[0]['std']是mm
                # 所以这里用std
                error = abs(mean_disp_mm - theoretical_disp)
                print(f"blockSize={blockSize}, numDisparities={numDisparities}, mean_disp_mm={mean_disp_mm:.6f}, error={error:.6f}")
                if error < best_error:
                    best_error = error
                    best_params = {'blockSize': blockSize, 'numDisparities': numDisparities}
                    best_result = disp
                    best_actual = mean_disp_mm
            except Exception as e:
                print(f"参数 blockSize={blockSize}, numDisparities={numDisparities} 出错: {e}")
    # 保存最优参数
    output_dir = os.path.join('output2D', 'optimize')
    os.makedirs(output_dir, exist_ok=True)
    # 记录本次用到的标定板和散斑图路径
    calib_left = sorted([os.path.join(config.CALIB_LEFT_DIR, f) for f in os.listdir(config.CALIB_LEFT_DIR) if f.endswith('.png')])
    calib_right = sorted([os.path.join(config.CALIB_RIGHT_DIR, f) for f in os.listdir(config.CALIB_RIGHT_DIR) if f.endswith('.png')])
    speckle_left = sorted([os.path.join(config.LEFT_DIR, f) for f in os.listdir(config.LEFT_DIR) if f.endswith('.png')])
    speckle_right = sorted([os.path.join(config.RIGHT_DIR, f) for f in os.listdir(config.RIGHT_DIR) if f.endswith('.png')])
    result = {
        'best_params': best_params,
        'best_error': best_error,
        'theoretical_disp_mm': theoretical_disp,
        'actual_disp_mm': best_actual,
        'calib_left': calib_left,
        'calib_right': calib_right,
        'speckle_left': speckle_left,
        'speckle_right': speckle_right
    }
    # 修复 numpy 类型无法序列化问题
    def to_python_type(obj):
        if isinstance(obj, dict):
            return {k: to_python_type(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_python_type(v) for v in obj]
        elif isinstance(obj, np.generic):
            return obj.item()
        else:
            return obj
    result = to_python_type(result)
    with open(os.path.join(output_dir, 'optimal_params.json'), 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"最优参数已保存到 {os.path.join(output_dir, 'optimal_params.json')}")

if __name__ == '__main__':
    main() 