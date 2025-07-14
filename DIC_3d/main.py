import sys
superres_flag = '--superres' if '--superres' in sys.argv else ''
import os
# os.system(f'python3 stereo_calib.py {superres_flag}')
os.system(f'python3 disparity_param_search.py {superres_flag}')  # 自动参数优化
os.system(f'python3 disparity_compute.py {superres_flag}')
os.system(f'python3 feature_match.py {superres_flag}')
os.system(f'python3 reproject_3d.py {superres_flag}')
os.system(f'python3 plot_displacement_stats.py {superres_flag}')