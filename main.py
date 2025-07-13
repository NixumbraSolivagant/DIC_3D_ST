use_superres = False  # 切换是否用超分
superres_flag = '--superres' if use_superres else ''
import os
os.system(f'python3 stereo_calib.py {superres_flag}')
os.system(f'python3 disparity_compute.py {superres_flag}')
os.system(f'python3 feature_match.py {superres_flag}')
os.system(f'python3 reproject_3d.py {superres_flag}')
os.system(f'python3 plot_displacement_stats.py {superres_flag}')