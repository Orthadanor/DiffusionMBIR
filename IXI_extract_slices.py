import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image

# Configuration
source_dir = Path("/home/yuchenliu/Dataset/t1_np")
output_root = Path("/home/yuchenliu/Dataset/IXI/t1_np/t1_np_sliced")
resized_root = Path("/home/yuchenliu/Dataset/IXI/t1_np/t1_np_sliced_128")

slice_range = range(55, 85)  # slices 55 to 84 (30 slices)
resize_to = (128, 128)

# Collect slices for splitting
all_slices = []

# Step 1: Normalize and collect slices (without clipping)
for vol_path in sorted(source_dir.glob("*.npy")):
    vol_name = vol_path.stem
    volume = np.load(vol_path).astype(np.float32)

    # Normalize without clipping
    vol_min, vol_max = np.min(volume), np.max(volume)
    volume = (volume - vol_min) / (vol_max - vol_min + 1e-8)

    for idx in slice_range:
        slice_img = volume[:, :, idx]  # shape (H, W)
        slice_fname = f"{vol_name}_slice{idx:03d}.npy"
        all_slices.append((slice_img, slice_fname))

# Step 2: Split
train_data, val_data = train_test_split(all_slices, test_size=0.2, random_state=42)

# Step 3: Save original normalized slices
for split, data in [('train', train_data), ('val', val_data)]:
    split_dir = output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for slice_img, fname in data:
        np.save(split_dir / fname, slice_img)

# Step 4: Resize and save 128×128 versions
for split, data in [('train', train_data), ('val', val_data)]:
    split_dir = resized_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for slice_img, fname in data:
        # Convert to [0,255] uint8 for PIL
        pil_img = Image.fromarray((slice_img * 255).astype(np.uint8))
        pil_img = pil_img.resize(resize_to, resample=Image.BICUBIC)

        # Back to float32 [0,1]
        resized = np.array(pil_img).astype(np.float32) / 255.0
        np.save(split_dir / fname, resized)

print(f"Done. Saved {len(train_data)} slices to 'train/' and {len(val_data)} to 'val/'")
print(f"Done. Normalized and resized slices saved to:")
print(f"- {output_root}")
print(f"- {resized_root}")
