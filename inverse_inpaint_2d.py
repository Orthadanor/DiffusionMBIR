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
from physics.inpainting import Inpainting
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

###############################################
# Configurations
###############################################
problem = 'inpainting_ADMM_TV_2d'
config_name = 'IXI_128_ncsnpp_continuous'
sde = 'VESDE'
num_scales = 2000
ckpt_num = 6
N = num_scales

vol_name = 'L067'
root = Path(f'./data/CT/ind/256_sorted/{vol_name}')

# Parameters for the inverse problem
mask_rate = 0.2
size = 128
lamb = 0.04
rho = 10

if sde.lower() == 'vesde':
    from configs.ve import IXI_128_ncsnpp_continuous as configs
    ckpt_filename = f"exp/ve/{config_name}/checkpoint_{ckpt_num}.pth"
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

batch_size = 12
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
save_root = Path(f'./results/{config_name}/{problem}/mask{mask_rate}/rho{rho}/lambda{lamb}')
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
fname_list = sorted(fname_list, key=lambda x: float(x.split(".")[0]))
print(fname_list)
all_img = []

print("Loading all data")
for fname in tqdm(fname_list):
    just_name = fname.split('.')[0]
    img = torch.from_numpy(np.load(os.path.join(root, fname), allow_pickle=True))
    h, w = img.shape
    img = img.view(1, 1, h, w)
    all_img.append(img)
    plt.imsave(os.path.join(save_root, 'label', f'{just_name}.png'), clear(img), cmap='gray')
all_img = torch.cat(all_img, dim=0)
print(f"Data loaded shape : {all_img.shape}")

# Inpainting operator
inpaint = Inpainting(img_heigth=h, img_width=w, mask_rate=mask_rate, device=config.device)

# Save mask for visualization
plt.imsave(os.path.join(save_root, 'mask', 'mask.png'), inpaint.mask.cpu().numpy(), cmap='gray')

img = all_img.to(config.device)
pc_inpaint = controllable_generation_TV.get_pc_radon_ADMM_TV_vol(
    sde, predictor, corrector, inverse_scaler,
    snr=snr, n_steps=n_steps, probability_flow=probability_flow,
    continuous=config.training.continuous, denoise=True,
    radon=inpaint,  # Use inpainting operator
    save_progress=True, save_root=save_root, final_consistency=True,
    img_shape=img.shape, lamb_1=lamb, rho=rho
)

# Forward model (apply mask)
masked_img = inpaint.A(img)

# Adjoint (for iterative solver, same as mask for inpainting)
bp = inpaint.A_dagger(masked_img)

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

print("Inpainting pipeline finished.")