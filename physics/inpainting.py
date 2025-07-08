import os
import torch

class Inpainting():
    def __init__(self, img_heigth=512, img_width=512, mode='random', mask_rate=0.3, resize=False, device='cuda:0'):
        mask_path = './physics/inpaint_masks/mask_2d/mask_random{}.pt'.format(mask_rate)
        if os.path.exists(mask_path):
            self.mask = torch.load(mask_path).to(device)
        else:
            self.mask = torch.ones(img_heigth, img_width, device=device)
            self.mask[torch.rand_like(self.mask) > 1 - mask_rate] = 0
            torch.save(self.mask, mask_path)

    def A(self, x):
        return torch.einsum('kl,ijkl->ijkl', self.mask, x)

    def A_dagger(self, x):
        return torch.einsum('kl,ijkl->ijkl', self.mask, x)
    
    def AT(self, y):
        return torch.einsum('kl,ijkl->ijkl', self.mask, y)

class Inpainting3D:
    def __init__(self, n_slices, img_heigth=128, img_width=128, rect_height=32, rect_width=32, device='cuda:0'):
        # Create a central rectangle mask
        mask = torch.ones(img_heigth, img_width, device=device)
        y0 = (img_heigth - rect_height) // 2
        x0 = (img_width - rect_width) // 2
        mask[y0:y0+rect_height, x0:x0+rect_width] = 0  # Zero out the central rectangle

        # Repeat the same mask for all slices
        self.masks = mask.unsqueeze(0).repeat(n_slices, 1, 1)  # (n_slices, H, W)

    def A(self, x):
        # x: (num_slices, 1, H, W)
        return x * self.masks.unsqueeze(1)

    def A_dagger(self, x):
        return x * self.masks.unsqueeze(1)

    def AT(self, y):
        return y * self.masks.unsqueeze(1)
    
class Inpainting3D_mask_noise():
    def __init__(self, num_slices, img_heigth=512, img_width=512, mode='random', mask_rate=0.3, device='cuda:0'):
        self.masks = []
        mask_dir = f'./physics/inpaint_masks/mask_3d/mask_random{mask_rate}'
        os.makedirs(mask_dir, exist_ok=True)
        for i in range(num_slices):
            mask_path = os.path.join(mask_dir, f'mask_{i:03d}.pt')
            if os.path.exists(mask_path):
                mask = torch.load(mask_path).to(device)
            else:
                mask = torch.ones(img_heigth, img_width, device=device)
                mask[torch.rand_like(mask) > 1 - mask_rate] = 0
                torch.save(mask, mask_path)
            self.masks.append(mask)
        # Stack to shape (num_slices, img_heigth, img_width)
        self.masks = torch.stack(self.masks, dim=0)
        # Save the bulk masks tensor
        torch.save(self.masks, os.path.join(mask_dir, 'masks.pt'))

    def A(self, x):
        # x: (num_slices, 1, H, W)
        # self.masks: (num_slices, H, W)
        return x * self.masks.unsqueeze(1)

    def A_dagger(self, x):
        return x * self.masks.unsqueeze(1)

    def AT(self, y):
        return y * self.masks.unsqueeze(1)