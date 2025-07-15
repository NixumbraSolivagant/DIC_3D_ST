import os
import json
import numpy as np
import random
from DIC_2D.calibration import calibration, rectify
from DIC_2D.disparity import sgbm, zncc
from DIC_2D.analysis import displacement
import config
from tqdm import tqdm

def main():
    # 标定与校正
    calib_data = calibration.run_calibration(config)
    if 'pixel_size_mm' in calib_data and calib_data['pixel_size_mm'] is not None:
        config.pixel_size_mm = calib_data['pixel_size_mm']
    rectify_data = rectify.run_rectification(calib_data, config)
    theoretical_disp = getattr(config, 'THEORETICAL_DISPLACEMENT_MM', 1.0)
    best_error = float('inf')
    best_params = None
    best_actual = None
    # 获取所有图片对
    left_list = sorted([os.path.join(config.LEFT_DIR, f) for f in os.listdir(config.LEFT_DIR) if f.endswith('.png')])
    right_list = sorted([os.path.join(config.RIGHT_DIR, f) for f in os.listdir(config.RIGHT_DIR) if f.endswith('.png')])
    all_pairs = list(zip(left_list, right_list))
    if len(all_pairs) < 3:
        raise RuntimeError('图片对数量不足3组，无法进行随机采样调优！')

    if config.DISPARITY_METHOD == 'ZNCC':
        # ZNCC参数空间
        max_disp_list = [32, 48, 64, 80, 96]
        win_size_list = [5, 7, 9, 11]
        step_list = [1, 2, 3]
        zncc_thresh_list = [0.05, 0.1, 0.15, 0.2]
        total = len(max_disp_list) * len(win_size_list) * len(step_list) * len(zncc_thresh_list)
        pbar = tqdm(total=total, desc='ZNCC参数调优')
        for max_disp in max_disp_list:
            for win_size in win_size_list:
                for step in step_list:
                    for zncc_thresh in zncc_thresh_list:
                        print(f"当前参数: max_disp={max_disp}, win_size={win_size}, step={step}, zncc_thresh={zncc_thresh}")
                        config.ZNCC_PARAMS['max_disp'] = max_disp
                        config.ZNCC_PARAMS['win_size'] = win_size
                        config.ZNCC_PARAMS['step'] = step
                        config.ZNCC_PARAMS['zncc_thresh'] = zncc_thresh
                        sample_pairs = random.sample(all_pairs, 3)
                        mean_errors = []
                        for idx, (left_path, right_path) in enumerate(sample_pairs):
                            orig_left = config.LEFT_DIR
                            orig_right = config.RIGHT_DIR
                            config.LEFT_DIR = os.path.dirname(left_path)
                            config.RIGHT_DIR = os.path.dirname(right_path)
                            left_name = os.path.basename(left_path)
                            right_name = os.path.basename(right_path)
                            for f in os.listdir(config.LEFT_DIR):
                                if f != left_name and f.endswith('.png'):
                                    os.rename(os.path.join(config.LEFT_DIR, f), os.path.join(config.LEFT_DIR, f+'.bak'))
                            for f in os.listdir(config.RIGHT_DIR):
                                if f != right_name and f.endswith('.png'):
                                    os.rename(os.path.join(config.RIGHT_DIR, f), os.path.join(config.RIGHT_DIR, f+'.bak'))
                            try:
                                disparity_maps = zncc.compute_disparity(rectify_data, config)
                                disp = disparity_maps[0]
                                results = displacement.analyze([disp], config)
                                mean_disp_mm = results[0]['std']
                                error = abs(mean_disp_mm - theoretical_disp)
                                mean_errors.append(error)
                            except Exception as e:
                                print(f"ZNCC参数 max_disp={max_disp}, win_size={win_size}, step={step}, zncc_thresh={zncc_thresh}，图片组{idx+1}出错: {e}")
                                mean_errors.append(float('inf'))
                            finally:
                                for f in os.listdir(config.LEFT_DIR):
                                    if f.endswith('.bak'):
                                        os.rename(os.path.join(config.LEFT_DIR, f), os.path.join(config.LEFT_DIR, f[:-4]))
                                for f in os.listdir(config.RIGHT_DIR):
                                    if f.endswith('.bak'):
                                        os.rename(os.path.join(config.RIGHT_DIR, f), os.path.join(config.RIGHT_DIR, f[:-4]))
                                config.LEFT_DIR = orig_left
                                config.RIGHT_DIR = orig_right
                        avg_error = np.mean(mean_errors)
                        print(f"ZNCC: max_disp={max_disp}, win_size={win_size}, step={step}, zncc_thresh={zncc_thresh}, avg_error={avg_error:.6f}")
                        if avg_error < best_error:
                            best_error = avg_error
                            best_params = {'max_disp': max_disp, 'win_size': win_size, 'step': step, 'zncc_thresh': zncc_thresh}
                            best_actual = None if not mean_errors else float(np.mean(mean_errors))
                        pbar.update(1)
        pbar.close()
    elif config.DISPARITY_METHOD == 'SGBM':
        # SGBM参数空间
        numDisparities_list = [64, 96, 128]
        blockSize_list = [3, 5, 7]
        uniquenessRatio_list = [5, 10, 15]
        speckleWindowSize_list = [50, 100, 200]
        speckleRange_list = [1, 2, 4]
        total = len(numDisparities_list) * len(blockSize_list) * len(uniquenessRatio_list) * len(speckleWindowSize_list) * len(speckleRange_list)
        pbar = tqdm(total=total, desc='SGBM参数调优')
        for numDisparities in numDisparities_list:
            for blockSize in blockSize_list:
                for uniquenessRatio in uniquenessRatio_list:
                    for speckleWindowSize in speckleWindowSize_list:
                        for speckleRange in speckleRange_list:
                            print(f"当前参数: numDisparities={numDisparities}, blockSize={blockSize}, uniquenessRatio={uniquenessRatio}, speckleWindowSize={speckleWindowSize}, speckleRange={speckleRange}")
                            config.SGBM_PARAMS['numDisparities'] = numDisparities
                            config.SGBM_PARAMS['blockSize'] = blockSize
                            config.SGBM_PARAMS['uniquenessRatio'] = uniquenessRatio
                            config.SGBM_PARAMS['speckleWindowSize'] = speckleWindowSize
                            config.SGBM_PARAMS['speckleRange'] = speckleRange
                            sample_pairs = random.sample(all_pairs, 3)
                            mean_errors = []
                            for idx, (left_path, right_path) in enumerate(sample_pairs):
                                orig_left = config.LEFT_DIR
                                orig_right = config.RIGHT_DIR
                                config.LEFT_DIR = os.path.dirname(left_path)
                                config.RIGHT_DIR = os.path.dirname(right_path)
                                left_name = os.path.basename(left_path)
                                right_name = os.path.basename(right_path)
                                for f in os.listdir(config.LEFT_DIR):
                                    if f != left_name and f.endswith('.png'):
                                        os.rename(os.path.join(config.LEFT_DIR, f), os.path.join(config.LEFT_DIR, f+'.bak'))
                                for f in os.listdir(config.RIGHT_DIR):
                                    if f != right_name and f.endswith('.png'):
                                        os.rename(os.path.join(config.RIGHT_DIR, f), os.path.join(config.RIGHT_DIR, f+'.bak'))
                                try:
                                    disparity_maps = sgbm.compute_disparity(rectify_data, config)
                                    disp = disparity_maps[0]
                                    results = displacement.analyze([disp], config)
                                    mean_disp_mm = results[0]['std']
                                    error = abs(mean_disp_mm - theoretical_disp)
                                    mean_errors.append(error)
                                except Exception as e:
                                    print(f"SGBM参数 numDisparities={numDisparities}, blockSize={blockSize}, uniquenessRatio={uniquenessRatio}, speckleWindowSize={speckleWindowSize}, speckleRange={speckleRange}，图片组{idx+1}出错: {e}")
                                    mean_errors.append(float('inf'))
                                finally:
                                    for f in os.listdir(config.LEFT_DIR):
                                        if f.endswith('.bak'):
                                            os.rename(os.path.join(config.LEFT_DIR, f), os.path.join(config.LEFT_DIR, f[:-4]))
                                    for f in os.listdir(config.RIGHT_DIR):
                                        if f.endswith('.bak'):
                                            os.rename(os.path.join(config.RIGHT_DIR, f), os.path.join(config.RIGHT_DIR, f[:-4]))
                                    config.LEFT_DIR = orig_left
                                    config.RIGHT_DIR = orig_right
                            avg_error = np.mean(mean_errors)
                            print(f"SGBM: numDisparities={numDisparities}, blockSize={blockSize}, uniquenessRatio={uniquenessRatio}, speckleWindowSize={speckleWindowSize}, speckleRange={speckleRange}, avg_error={avg_error:.6f}")
                            if avg_error < best_error:
                                best_error = avg_error
                                best_params = {'numDisparities': numDisparities, 'blockSize': blockSize, 'uniquenessRatio': uniquenessRatio, 'speckleWindowSize': speckleWindowSize, 'speckleRange': speckleRange}
                                best_actual = None if not mean_errors else float(np.mean(mean_errors))
                            pbar.update(1)
        pbar.close()
    else:
        raise ValueError('未知视差方法: ' + str(config.DISPARITY_METHOD))

    # 保存最优参数
    output_dir = os.path.join('output2D_0.1', 'optimize')
    os.makedirs(output_dir, exist_ok=True)
    result = {
        'best_params': best_params,
        'best_error': best_error,
        'theoretical_disp_mm': theoretical_disp,
        'actual_disp_mm': best_actual
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