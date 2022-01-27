import numpy as np
import cv2
import torch


def save_img(img, path):
    """
    Save the input image to the file path.
    Args:
        img:    Image [H x W x C].
        path:   Path to the file where to save the image.

    Returns: None
    """
    if isinstance(img, np.ndarray):
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        img.save(path)


def resize(img, scale):
    H, W, _ = img.shape
    return cv2.resize(img, (scale * H, scale * W))


# def get_size(img):
#     if isinstance(img, (np.ndarray, torch.Tensor)):
#         return img.shape[1::-1]
#     else:
#         return img.size


def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


def rescale_img(imgs):
    """
    Rescale the values of an images from [-1,1] to [0, 255] and permute the channels of the images from [N, C, H, W] to
    [N, H, W, C].
    Args:
        imgs:   Image to rescale.

    Returns: Rescaled images.
    """
    return (imgs.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)


# def prewhiten(x):
#     mean = x.mean()
#     std = x.std()
#     std_adj = std.clamp(min=1.0/(float(x.numel())**0.5))
#     y = (x - mean) / std_adj
#     return y


def one_hot_vector(length, index, device=torch.device('cpu')):
    """
    Create a one-hot-vector of a specific length with a 1 in the given index.
    Args:
        length: Total length of the vector.
        index:  Index of the 1 in the vector.
        device: Torch device (GPU or CPU) to load the vector to.

    Returns: Torch vector of size [1 x length] filled with zeros, and a 1 only at index, loaded to device.
    """
    vector = torch.zeros([1, length]).to(device)
    vector[0, index] = 1.
    return vector
