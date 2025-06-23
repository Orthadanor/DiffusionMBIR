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

    # Normalize
    t1_slice = (t1_slice - np.min(t1_slice)) / (np.max(t1_slice) - np.min(t1_slice) + 1e-8)

    # Plot
    plt.figure(figsize=(8, 6))
    

    if overlay_mask:
        # plt.imshow(mask_slice.T > 0.5, cmap='Reds', alpha=0.3, origin='lower')
        t1_masked = np.copy(t1_slice)
        t1_masked[mask_slice > 0.5] = 0.0  # Black out masked voxels
        plt.imshow(t1_masked.T, cmap='gray', origin='lower')
    else:
        plt.imshow(t1_slice.T, cmap='gray', origin='lower')
        
    plt.title(f"{plane.capitalize()} Slice {slice_index}")
    plt.axis('off')


    os.makedirs(output_dir, exist_ok=True)
    if overlay_mask:
        out_file = os.path.join(output_dir, f"{plane}_slice_{slice_index}_mask.png")
    else:
        out_file = os.path.join(output_dir, f"{plane}_slice_{slice_index}.png")
    plt.savefig(out_file, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved visualization to: {out_file}")

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
    parser.add_argument("--mode", type=str, choices=["info", "slice", "fsleyes"], required=True)
    parser.add_argument("--t1_file", type=str, required=True)
    parser.add_argument("--mask_file", type=str, required=True)
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

if __name__ == "__main__":
    main()
