import os
os.system('python3 stereo_calib.py')
os.system('python3 disparity_compute.py')
os.system('python3 feature_match.py')
os.system('python3 reproject_3d.py') 