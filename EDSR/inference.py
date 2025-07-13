import torch
import numpy as np
from PIL import Image
import os
import sys

# === 1. 手写参数 ===
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

# === 2. 导入模型和工具 ===
from src.model.edsr import EDSR
import src.utility

# === 3. 加载模型和权重 ===
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
            sr_path = os.path.join(sr_folder, 'SR_' + fname)
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

# === 4. 批量处理左右相机 ===
folders = [
    ('/home/herzog/DIC/capture/Left', '/home/herzog/DIC/capture_SR/Left'),
    ('/home/herzog/DIC/capture/Right', '/home/herzog/DIC/capture_SR/Right')
]
for lr_folder, sr_folder in folders:
    print(f'Processing {lr_folder} ...')
    superres_batch(lr_folder, sr_folder)
