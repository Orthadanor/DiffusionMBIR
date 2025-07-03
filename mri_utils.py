import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import subprocess

def print_info(t1_file, mask_file):
    t1 = np.load(t1_file)
    mask = np.load(mask_file)

    print(f"T1 image file: {t1_file}")
    print(f"Shape: {t1.shape}, Dtype: {t1.dtype}")
    print(f"Stats: min={t1.min():.3f}, max={t1.max():.3f}, mean={t1.mean():.3f}")

    print(f"\nMask image file: {mask_file}")
    print(f"Shape: {mask.shape}, Dtype: {mask.dtype}")
    print(f"Stats: min={mask.min():.3f}, max={mask.max():.3f}, mean={mask.mean():.3f}")
    print(f"Unique values in mask: {np.unique(mask)}")

def visualize_slice(t1_file, mask_file, plane, slice_index, output_dir, overlay_mask=True):
    t1 = np.load(t1_file)
    mask = np.load(mask_file)

    if t1.ndim == 2:
        # Already a 2D slice
        t1_slice = t1
        mask_slice = mask
    elif t1.ndim == 3:
        if plane == 'sagittal':
            t1_slice = t1[slice_index, :, :]
            mask_slice = mask[slice_index, :, :]
        elif plane == 'coronal':
            t1_slice = t1[:, slice_index, :]
            mask_slice = mask[:, slice_index, :]
        elif plane == 'axial':
            t1_slice = t1[:, :, slice_index]
            mask_slice = mask[:, :, slice_index]
        else:
            raise ValueError("Plane must be 'sagittal', 'coronal', or 'axial'.")
    else:
        raise ValueError(f"Unsupported input shape: {t1.shape}")

    # Normalize for display
    t1_slice = (t1_slice - np.min(t1_slice)) / (np.max(t1_slice) - np.min(t1_slice) + 1e-8)
    print(f"Visualizing {plane} slice {slice_index} with shape {t1_slice.shape}")

    # Plot
    plt.figure(figsize=(8, 6))

    if overlay_mask:
        t1_masked = np.copy(t1_slice)
        t1_masked[mask_slice > 0.5] = 0.0
        plt.imshow(t1_masked.T, cmap='gray', origin='lower')
    else:
        plt.imshow(t1_slice.T, cmap='gray', origin='lower')

    plt.title(f"{plane.capitalize()} Slice {slice_index}")
    plt.axis('off')

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{plane}_slice_{slice_index}.png")
    plt.savefig(out_file, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved visualization to: {out_file}")
    
def visualize_slice_with_percentile_clip(t1_file, plane, slice_index, output_dir=None):
    """
    Visualize a slice from t1_file, plot its histogram,
    clip values above 98th percentile, and show both original and clipped images.
    """
    t1 = np.load(t1_file)
    # Select slice
    if t1.ndim == 2:
        t1_slice = t1
    elif t1.ndim == 3:
        if plane == 'sagittal':
            t1_slice = t1[slice_index, :, :]
        elif plane == 'coronal':
            t1_slice = t1[:, slice_index, :]
        elif plane == 'axial':
            t1_slice = t1[:, :, slice_index]
        else:
            raise ValueError("Plane must be 'sagittal', 'coronal', or 'axial'.")
    else:
        raise ValueError(f"Unsupported input shape: {t1.shape}")

    # Plot histogram
    plt.figure(figsize=(16, 5))
    plt.subplot(1, 3, 1)
    plt.hist(t1_slice.flatten(), bins=100, color='gray')
    plt.title("Histogram of Slice")
    plt.xlabel("Intensity")
    plt.ylabel("Count")

    # Compute 98th percentile
    p98 = np.percentile(t1_slice, 98)
    # Clip values above 98th percentile
    t1_clipped = np.copy(t1_slice)
    t1_clipped[t1_clipped > p98] = p98

    # Normalize for display
    t1_slice_norm = (t1_slice - np.min(t1_slice)) / (np.max(t1_slice) - np.min(t1_slice) + 1e-8)
    t1_clipped_norm = (t1_clipped - np.min(t1_clipped)) / (np.max(t1_clipped) - np.min(t1_clipped) + 1e-8)

    # Plot original slice
    plt.subplot(1, 3, 2)
    plt.imshow(t1_slice_norm.T, cmap='gray', origin='lower')
    plt.title("Original Slice")
    plt.axis('off')

    # Plot clipped slice
    plt.subplot(1, 3, 3)
    plt.imshow(t1_clipped_norm.T, cmap='gray', origin='lower')
    plt.title("Clipped (Top 2% mapped to 98th percentile)")
    plt.axis('off')

    plt.tight_layout()
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f"cropped_{plane}_slice_{slice_index}_hist.png")
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0.1)
        print(f"Saved visualization to: {out_file}")
        plt.close()
    plt.show()


def launch_fsleyes(t1_file, mask_file, out_dir):
    t1 = np.load(t1_file)
    mask = np.load(mask_file)
    affine = np.eye(4)

    os.makedirs(out_dir, exist_ok=True)
    t1_nii_path = os.path.join(out_dir, "t1.nii.gz")
    mask_nii_path = os.path.join(out_dir, "mask.nii.gz")

    nib.save(nib.Nifti1Image(t1, affine), t1_nii_path)
    nib.save(nib.Nifti1Image(mask, affine), mask_nii_path)

    print(f"Saved NIfTI files:\n  - {t1_nii_path}\n  - {mask_nii_path}")

    # Launch fsleyes
    try:
        subprocess.run(["fsleyes", t1_nii_path, mask_nii_path])
    except FileNotFoundError:
        print("Error: fsleyes not found. Make sure it is installed and available in PATH.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["info", "slice", "fsleyes", "slice_hist"], required=True)
    parser.add_argument("--t1_file", type=str, required=True)
    parser.add_argument("--mask_file", type=str, required=False)
    parser.add_argument("--no_mask", action="store_true", help="If set, disables mask overlay on slice visualization")
    
    # Slice mode args
    parser.add_argument("--plane", type=str, default="axial", choices=["sagittal", "coronal", "axial"])
    parser.add_argument("--slice_index", type=int, default=70)
    parser.add_argument("--output_dir", type=str, default="/home/yuchenliu/Images/raw_imgs")
    
    # FSLeyes mode args
    parser.add_argument("--fsleyes_dir", type=str, default="/home/yuchenliu/Images/nifti")

    args = parser.parse_args()

    if args.mode == "info":
        print_info(args.t1_file, args.mask_file)
    elif args.mode == "slice":
        visualize_slice(
            args.t1_file,
            args.mask_file,
            args.plane,
            args.slice_index,
            args.output_dir,
            overlay_mask=not args.no_mask
        )

    elif args.mode == "fsleyes":
        launch_fsleyes(args.t1_file, args.mask_file, args.fsleyes_dir)
        
    elif args.mode == "slice_hist":
        visualize_slice_with_percentile_clip(
            args.t1_file,
            args.plane,
            args.slice_index,
            args.output_dir
        )

if __name__ == "__main__":
    main()
