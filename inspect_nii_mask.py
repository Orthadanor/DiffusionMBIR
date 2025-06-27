import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# === File paths ===
t1_path = "/home/yuchenliu/Dataset/t1/IXI013-HH-1212_t1.npy"
mask_path = "/home/yuchenliu/Dataset/mask/IXI013-HH-1212_mask.npy"

# === Load NIfTI files ===
t1_img = nib.load(t1_path)
mask_img = nib.load(mask_path)

t1_data = t1_img.get_fdata()
mask_data = mask_img.get_fdata()

# === Sanity check ===
print(f"T1 shape: {t1_data.shape}, dtype: {t1_data.dtype}, min: {t1_data.min():.2f}, max: {t1_data.max():.2f}")
print(f"Mask shape: {mask_data.shape}, dtype: {mask_data.dtype}, unique: {np.unique(mask_data)}")

# === Pick axial mid-slice ===
mid_slice = t1_data.shape[2] // 2
t1_slice = t1_data[:, :, mid_slice]
mask_slice = mask_data[:, :, mid_slice]

# === Normalize T1 for display ===
t1_norm = (t1_slice - t1_slice.min()) / (t1_slice.max() - t1_slice.min() + 1e-8)

# === Plotting ===
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(t1_norm.T, cmap='gray', origin='lower')
axs[0].set_title("T1 Image (Axial Slice)")
axs[0].axis('off')

axs[1].imshow(t1_norm.T, cmap='gray', origin='lower')
axs[1].imshow(mask_slice.T > 0.5, cmap='Reds', alpha=0.3, origin='lower')
axs[1].set_title("T1 + Mask Overlay")
axs[1].axis('off')

plt.tight_layout()
output_path = "/home/yuchenliu/Images/raw_imgs/IXI013-HH-1212_axial_overlay.png"
plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
print(f"Saved visualization to: {output_path}")

