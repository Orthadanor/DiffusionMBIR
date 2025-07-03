import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image

# --- Step 1: Process 3D volumes and save mapped slices ---

# Configuration for extraction
source_dir = Path("/home/yuchenliu/Dataset")
t1_dir = source_dir / "t1_np"
mask_dir = source_dir / "mask_np"
output_root = Path("/home/yuchenliu/Dataset/IXI/t1_np_masked")
slice_range = range(55, 85)  # slices 55 to 84 (30 slices)

output_root.mkdir(parents=True, exist_ok=True)
total_slices_saved = 0

for t1_path in sorted(t1_dir.glob("*.npy")):
    vol_name = t1_path.stem.replace("_t1", "")
    mask_path = mask_dir / f"{vol_name}_mask.npy"
    if not mask_path.exists():
        print(f"Mask not found for {t1_path.name}, skipping.")
        continue

    t1_vol = np.load(t1_path).astype(np.float32)
    mask_vol = np.load(mask_path).astype(np.float32)

    for idx in slice_range:
        t1_slice = t1_vol[:, :, idx]
        mask_slice = mask_vol[:, :, idx]

        # Background voxels: mask == 0.0
        background_mask = (mask_slice == 0.0)
        n_background = np.sum(background_mask)
        # Actual voxels: mask != 0.0
        actual_voxels = t1_slice[~background_mask]
        n_actual = actual_voxels.size

        if n_actual == 0:
            print(f"All voxels are background in {vol_name}_slice{idx:03d}, skipping.")
            continue

        # Map top 2% of actual voxels to the 98th percentile
        p98 = np.percentile(actual_voxels, 98)
        mapped_voxels = actual_voxels > p98
        n_mapped = np.sum(mapped_voxels)
        # Apply mapping
        processed_slice = np.copy(t1_slice)
        processed_slice[~background_mask] = np.where(
            actual_voxels > p98, p98, actual_voxels
        )

        # Save processed slice
        slice_fname = f"{vol_name}_t1_slice{idx:03d}.npy"
        np.save(output_root / slice_fname, processed_slice)
        total_slices_saved += 1

        print(
            f"{slice_fname}: background={n_background}, "
            f"actual={n_actual}, mapped_top2%={n_mapped}"
        )

print(f"Total slices saved: {total_slices_saved}")

# --- Step 2: Normalize, resize, split, and save slices ---

# Configuration for resizing and splitting
resized_root = Path("/home/yuchenliu/Dataset/IXI/t1_np_masked_128")
resize_to = (128, 128)

# Collect all slices
all_slices = []
for npy_path in sorted(output_root.glob("*.npy")):
    slice_img = np.load(npy_path).astype(np.float32)
    # Normalize to [0, 1]
    min_val, max_val = slice_img.min(), slice_img.max()
    if max_val > min_val:
        slice_img = (slice_img - min_val) / (max_val - min_val)
    else:
        slice_img = np.zeros_like(slice_img)
    all_slices.append((slice_img, npy_path.name))

# Train/val split
train_data, val_data = train_test_split(all_slices, test_size=0.2, random_state=42)

# Save resized slices
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

print(f"Done. Saved {len(train_data)} slices to '{resized_root / 'train'}' and {len(val_data)} to '{resized_root / 'val'}'")