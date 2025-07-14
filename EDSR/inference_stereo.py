import torch
import numpy as np
from PIL import Image
import os

# === 1. 参数 ===
class Args:
    def __init__(self):
        self.scale = [2]
        self.rgb_range = 255
        self.n_colors = 1
        self.precision = 'single'
        self.cpu = False
        self.n_resblocks = 8
        self.n_feats = 32
        self.res_scale = 1

args = Args()

from src.model.edsr import EDSR
import src.utility

model = EDSR(args)
state_dict = torch.load('/home/herzog/DIC/DIC/models/model_best.pt', map_location='cpu')
model.load_state_dict(state_dict, strict=True)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
model = model.to(device)

def superres_batch(lr_folder, sr_folder):
    os.makedirs(sr_folder, exist_ok=True)
    for fname in os.listdir(lr_folder):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            lr_path = os.path.join(lr_folder, fname)
            sr_path = os.path.join(sr_folder, fname)
            img = Image.open(lr_path).convert('L')
            img = np.array(img).astype(np.float32)
            img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
            if args.precision == 'half':
                img = img.half()
            img = img.to(device)
            with torch.no_grad():
                sr = model(img)
                sr = src.utility.quantize(sr, args.rgb_range)
            sr_img = sr.squeeze().cpu().numpy().astype(np.uint8)
            Image.fromarray(sr_img, mode='L').save(sr_path)
            print(f'Saved: {sr_path}')

# === 2. 批量处理左相机标定板图片 ===
print('Processing left camera calibration images...')
superres_batch('/home/herzog/DIC/text/Left', '/home/herzog/DIC/text/Left_SR')

# === 3. 批量处理右相机标定板图片 ===
print('Processing right camera calibration images...')
superres_batch('/home/herzog/DIC/text/Right', '/home/herzog/DIC/text/Right_SR') 
