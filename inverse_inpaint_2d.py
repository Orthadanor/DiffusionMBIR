import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import numpy as np
import controllable_generation_TV

from utils import restore_checkpoint, clear, batchfy, patient_wise_min_max, img_wise_min_max
from pathlib import Path
from models import utils as mutils
from models import ncsnpp
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor, LangevinCorrector)
import datasets
import time
from physics.inpainting import Inpainting3D
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from PIL import Image

###############################################
# Configurations
###############################################
problem = 'inpainting_ADMM_TV_2d'
config_name = 'IXI_128_ncsnpp_continuous'
sde = 'VESDE'
num_scales = 2000 # Diffusion steps
ckpt_num = 10 # Use best checkpoint
N = num_scales
resize_to = (128, 128)
slice_range = range(55, 85) # Slice range within the volume for inpainting

vol_name = 'IXI002-Guys-0828_t1.npy'
root = Path(f'./data/IXI/ind')

# Parameters for the inverse problem
mask_rate = 'middle'
size = 128
lamb = 0.04
rho = 10

if sde.lower() == 'vesde':
    from configs.ve import IXI_128_ncsnpp_continuous as configs
    ckpt_filename = f"exp/ve/{config_name}"
    config = configs.get_config()
    config.model.num_scales = N
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sde.N = N
    sampling_eps = 1e-5
predictor = ReverseDiffusionPredictor
corrector = LangevinCorrector
probability_flow = False
snr = 0.16
n_steps = 1

batch_size = 10
config.training.batch_size = batch_size
config.eval.batch_size = batch_size
random_seed = 0

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer, model=score_model, ema=ema)

state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=True, skip_optimizer=True)
ema.copy_to(score_model.parameters())

# Specify save directory for saving generated samples
save_root = Path(f'./results/{config_name}/{problem}/mask{mask_rate}_uniform/rho{rho}/lambda{lamb}')
save_root.mkdir(parents=True, exist_ok=True)

# input - Masked input image (i.e., image with missing pixels)
# recon	- Reconstructed output from the diffusion pipeline
# label	- Ground truth original image
# BP - Back-projected image (pseudo-inverse reconstruction / initialization)
# mask - Binary mask used for inpainting
irl_types = ['input', 'recon', 'label', 'BP', 'mask']
for t in irl_types:
    if t == 'recon':
        save_root_f = save_root / t / 'progress'
        save_root_f.mkdir(exist_ok=True, parents=True)
    else:
        save_root_f = save_root / t
        save_root_f.mkdir(parents=True, exist_ok=True)

# read all data
fname_list = os.listdir(root)
# fname_list = sorted(fname_list, key=lambda x: float(x.split(".")[0]))
fname_list = sorted(fname_list)
print(fname_list)
all_img = []

print("Loading all data")
all_img = []
for fname in tqdm(fname_list):
    just_name = fname.split('.')[0]
    vol = np.load(os.path.join(root, fname), allow_pickle=True)  # shape: (139, 176, 140)
    for idx in slice_range:
        slice_img = vol[:, :, idx]  # shape: (139, 176)
        # Normalize to [0, 1]
        min_val, max_val = slice_img.min(), slice_img.max()
        if max_val > min_val:
            slice_img = (slice_img - min_val) / (max_val - min_val)
        else:
            slice_img = np.zeros_like(slice_img)
        # Resize to (128, 128)
        pil_img = Image.fromarray((slice_img * 255).astype(np.uint8))
        pil_img = pil_img.resize(resize_to, resample=Image.BICUBIC)
        resized = np.array(pil_img).astype(np.float32) / 255.0
        # Add batch and channel dimensions: (1, 1, H, W)
        img_tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0)
        all_img.append(img_tensor)
        # Save label image for visualization
        plt.imsave(os.path.join(save_root, 'label', f'{just_name}_slice{idx:03d}.png'), clear(img_tensor), cmap='gray')
all_img = torch.cat(all_img, dim=0)
print(f"Data loaded shape : {all_img.shape}")
n_slices, _, h, w = all_img.shape

# Inpainting operator
inpaint3d = Inpainting3D(n_slices=n_slices, img_heigth=h, img_width=w, device=config.device)

# Save mask for visualization
# Save each slice's mask as a separate image
for i in range(inpaint3d.masks.shape[0]):
    plt.imsave(os.path.join(save_root, 'mask', f'mask_{i:03d}.png'),
               inpaint3d.masks[i].cpu().numpy(), cmap='gray')
# plt.imsave(os.path.join(save_root, 'mask', 'mask.png'), inpaint3d.mask.cpu().numpy(), cmap='gray')
# Save the bulk masks tensor
torch.save(inpaint3d.masks.cpu(), os.path.join(save_root, 'mask', 'masks.pt'))
# Access masks
masks = torch.load(os.path.join(save_root, 'mask', 'masks.pt'))

img = all_img.to(config.device)
pc_inpaint = controllable_generation_TV.get_pc_radon_ADMM_TV_vol(
    sde, predictor, corrector, inverse_scaler,
    snr=snr, n_steps=n_steps, probability_flow=probability_flow,
    continuous=config.training.continuous, denoise=True,
    radon=inpaint3d,  # Use inpainting operator
    save_progress=True, save_root=save_root, final_consistency=True,
    img_shape=img.shape, lamb_1=lamb, rho=rho,
    recon_batch_size=config.eval.batch_size
)

# Forward model (apply mask)
masked_img = inpaint3d.A(img)

# Adjoint (for iterative solver, same as mask for inpainting)
bp = inpaint3d.A_dagger(masked_img)

# Recon Image
x = pc_inpaint(score_model, scaler(img), measurement=masked_img)
img_cache = x[-1].unsqueeze(0)

count = 0
for i, recon_img in enumerate(x):
    plt.imsave(save_root / 'BP' / f'{count}.png', clear(bp[i]), cmap='gray')
    plt.imsave(save_root / 'label' / f'{count}.png', clear(img[i]), cmap='gray')
    plt.imsave(save_root / 'recon' / f'{count}.png', clear(recon_img), cmap='gray')
    plt.imsave(save_root / 'input' / f'{count}.png', clear(masked_img[i]), cmap='gray')
    count += 1

# Save the reconstructed volume as a numpy file
np.save(os.path.join(save_root, 'recon_volume.npy'), x.cpu().numpy())

print("Inpainting pipeline finished.")