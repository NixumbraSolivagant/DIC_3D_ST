import os
import json
import subprocess
from DIC_2D.calibration import calibration, rectify
from DIC_2D.disparity import sgbm, zncc
from DIC_2D.analysis import displacement, visualization
import config
from tqdm import tqdm


def load_or_optimize_params():
    opt_path = os.path.join('output2D_0.1', 'optimize', 'optimal_params.json')
    if not os.path.exists(opt_path):
        print('未找到最优参数文件，自动运行optimize.py进行调优...')
        subprocess.run(['python', 'optimize.py'], check=True)
    with open(opt_path, 'r') as f:
        result = json.load(f)
    return result['best_params']


def main():
    steps = [
        '参数加载/优化',
        '标定与校正',
        '视差计算',
        '位移分析',
        '可视化与保存'
    ]
    with tqdm(total=len(steps), desc='主流程', ncols=80) as pbar:
        # 1. 参数加载/优化
        best_params = load_or_optimize_params()
        for k, v in best_params.items():
            if k in config.SGBM_PARAMS:
                config.SGBM_PARAMS[k] = v
        pbar.update(1)
        # 2. 标定与校正
        calib_data = calibration.run_calibration(config)
        if 'pixel_size_mm' in calib_data and calib_data['pixel_size_mm'] is not None:
            setattr(config, 'pixel_size_mm', calib_data['pixel_size_mm'])
        rectify_data = rectify.run_rectification(calib_data, config)
        pbar.update(1)
        # 3. 视差计算
        if config.DISPARITY_METHOD == 'SGBM':
            disparity_maps = sgbm.compute_disparity(rectify_data, config)
        elif config.DISPARITY_METHOD == 'ZNCC':
            disparity_maps = zncc.compute_disparity(rectify_data, config)
        else:
            raise ValueError('未知视差方法')
        pbar.update(1)
        # 4. 位移分析
        disp_results = displacement.analyze(disparity_maps, config)
        pbar.update(1)
        # 5. 可视化与保存
        visualization.save_all(disp_results, config)
        pbar.update(1)

if __name__ == '__main__':
    main()
 