"""
This file is the entry point of the preprocessing module.

"""

import os
import numpy as np
import nibabel as nib
import cv2


def load_nifti(path):
    """
    Load a NIfTI file and return it as a numpy array.
    """

    nifti = nib.load(path)
    volume = nifti.get_fdata()
    return volume


def normalize_image(image):
    """
    Normalize image using Z-score normalization.

    """
    mean = image.mean()
    std = image.std() + 1e-8
    return (image - mean) / std


def resize_image(image, size, is_mask=False):
    """
    Resize image or mask to a fixed size.
    """
    if is_mask:
        interpolation = cv2.INTER_NEAREST
    else:
        interpolation = cv2.INTER_LINEAR

    return cv2.resize(image, (size, size), interpolation=interpolation)


def extract_slices(volume, mask, target_size=256):
    """
    Convert 3D volume and mask into preprocessed 2D slices.

    Args:
        volume (np.ndarray): 3D image volume
        mask (np.ndarray): 3D mask volume
        target_size (int): output image size

    Returns:
        list of tuples: [(image_slice, mask_slice), ...]
    """
    slices = []

    depth = volume.shape[2]

    for z in range(depth):
        image_slice = volume[:, :, z]
        mask_slice = mask[:, :, z]

        # Skip empty slices (optional but recommended)
        if np.sum(mask_slice) == 0:
            continue

        image_slice = resize_image(image_slice, target_size, is_mask=False)
        mask_slice = resize_image(mask_slice, target_size, is_mask=True)

        image_slice = normalize_image(image_slice)

        # Add channel dimension for U-Net: (C, H, W)
        image_slice = np.expand_dims(image_slice, axis=0)

        slices.append((image_slice, mask_slice))

    return slices


def preprocess_case(image_path, mask_path, target_size=256):
    volume = load_nifti(image_path)
    mask = load_nifti(mask_path)

    return extract_slices(volume, mask, target_size)
