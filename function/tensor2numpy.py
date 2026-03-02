import torch
import numpy as np

#反归一化
def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img_tensor * std + mean).clamp(0, 1)
def tensor_to_numpy(img_tensor):
    """
    输入: img_tensor: [3,96,96], float in [0,1]
    输出: arr: [96,96,3], float in [0,1] 或 uint8 in [0,255]
    """
    img = denormalize(img_tensor)        # 还原归一化
    arr = img.permute(1,2,0).cpu().numpy()  # C,H,W -> H,W,C
    # 如果需要 uint8：
    arr_uint8 = (arr * 255).astype(np.uint8)
    return arr_uint8