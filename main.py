import os
import json
import subprocess
from DIC_2D.calibration import calibration, rectify
from DIC_2D.disparity import sgbm, zncc
from DIC_2D.analysis import displacement, visualization
import config


def load_or_optimize_params():
    opt_path = os.path.join('output2D', 'optimize', 'optimal_params.json')
    if not os.path.exists(opt_path):
        print('未找到最优参数文件，自动运行optimize.py进行调优...')
        subprocess.run(['python', 'optimize.py'], check=True)
    with open(opt_path, 'r') as f:
        result = json.load(f)
    return result['best_params']


def main():
    # 读取或自动优化参数
    best_params = load_or_optimize_params()
    for k, v in best_params.items():
        if k in config.SGBM_PARAMS:
            config.SGBM_PARAMS[k] = v
    # 1. 标定与校正
    calib_data = calibration.run_calibration(config)
    if 'pixel_size_mm' in calib_data and calib_data['pixel_size_mm'] is not None:
        setattr(config, 'pixel_size_mm', calib_data['pixel_size_mm'])
    rectify_data = rectify.run_rectification(calib_data, config)
    # 2. 视差计算
    if config.DISPARITY_METHOD == 'SGBM':
        disparity_maps = sgbm.compute_disparity(rectify_data, config)
    elif config.DISPARITY_METHOD == 'ZNCC':
        disparity_maps = zncc.compute_disparity(rectify_data, config)
    else:
        raise ValueError('未知视差方法')
    # 3. 位移分析
    disp_results = displacement.analyze(disparity_maps, config)
    # 4. 可视化与保存
    visualization.save_all(disp_results, config)

if __name__ == '__main__':
    main()
 