import numpy as np
import albumentations as albu
import torch
from scipy import ndimage
import cv2
from skimage import exposure


def to_dysplastic_vs_non_dysplastic(y, **kwargs):
    """Simplifies the segmentation problem by setting: non-dysplastic (label = 1) vs dysplastic (label = 2).

    Args:
        y: input batch labels (np.array).

    Returns:
        y (np.array): mask is now NDBE vs DYS.
    """
    return np.where(y > 1, 2, y)


def to_tensor_image(x, **kwargs):
    return torch.tensor(x.astype('float32'))


def to_tensor_mask(x, **kwargs):
    return torch.tensor(x.astype('int64'))


def transpose(x, **kwargs):
    # [B, H, W, 3] => [B, 3, H, W]
    return x.transpose(0, 3, 1, 2)


def filter_holes(tissue_w_holes, size_thresh):
    """Filters holes from a tissue mask.

    Args:
        tissue_w_holes: tissue mask with holes
        size_thresh: threshold for size of the removable holes

    Returns:
        tissue mask without holes

    (from: https://github.com/BPdeRooij/barrett_patch_extractor/)
    """
    # filter small objects from mask
    label_objects, _ = ndimage.label(tissue_w_holes)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes > size_thresh
    mask_sizes[0] = 0
    tissue_w_holes = mask_sizes[label_objects]

    # find holes using inverse and filter out large holes
    holes = np.invert(tissue_w_holes)
    label_objects, _ = ndimage.label(holes)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes < size_thresh
    mask_sizes[0] = 0
    holes = mask_sizes[label_objects]

    return np.logical_or(tissue_w_holes, holes)


def tissue_mask_batch(x, y, lum_thresh=0.85, size_thresh=5000):
    """Luminosity based tissue masker.

    Args:
        x: batch of images
            shape: (B, H, W, C)
        y: batch of annotations that has to be tissue masked
            shape: (B, H, W)
        lum_thresh: threshold for luminosity tissue
        size_thresh: threshold for filtering hole sizes

    Returns:
        y_masked: batch of annotations that are tissue masked
            shape: (B, H, W)

    """
    # result array
    y_masked = np.zeros_like(y)

    for i in range(len(x)):

        # get a tissue mask & filter holes
        image, mask = x[i], y[i]
        tissue_mask = get_luminosity_tissue_mask(image, threshold=lum_thresh)
        tissue_mask = filter_holes(tissue_mask, size_thresh=size_thresh)

        # apply the tissue mask
        y_masked[i] = np.where(np.logical_and(tissue_mask, mask), mask, 0)

    return y_masked


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transforms.

    Args:
        preprocessing_fn (callable): data normalization function
            (can be specific for each pretrained neural network).
    Return:
        transform: albumentations.Compose

    """
    _transform = [albu.Lambda(image=preprocessing_fn)] if preprocessing_fn else []

    _transform.extend([
        albu.Lambda(image=transpose),
        albu.Lambda(image=to_tensor_image, mask=to_tensor_mask),
    ])
    return albu.Compose(_transform)


def contrast_enhancer(img, low_p=2, high_p=98):
    """Enhancing contrast of the input image using intensity adjustment.
       This method uses both image low and high percentiles.

    Args:
        img (:class:`numpy.ndarray`): input image used to obtain tissue mask.
            Image should be uint8.
        low_p (scalar): low percentile of image values to be saturated to 0.
        high_p (scalar): high percentile of image values to be saturated to 255.
            high_p should always be greater than low_p.

    Returns:
        img (:class:`numpy.ndarray`):
            Image (uint8) with contrast enhanced.

    Raises:
        AssertionError: Internal errors due to invalid img type.

    """
    # check if image is not uint8
    if not img.dtype == np.uint8:
        raise AssertionError("Image should be uint8.")
    img_out = img.copy()
    p_low, p_high = np.percentile(img_out, (low_p, high_p))
    if p_low >= p_high:
        p_low, p_high = np.min(img_out), np.max(img_out)
    if p_high > p_low:
        img_out = exposure.rescale_intensity(
            img_out, in_range=(p_low, p_high), out_range=(0.0, 255.0)
        )
    return np.uint8(img_out)


def get_luminosity_tissue_mask(img, threshold):
    """Get tissue mask based on the luminosity of the input image.

    Args:
        img (:class:`numpy.ndarray`):
            Input image used to obtain tissue mask.
        threshold (float):
            Luminosity threshold used to determine tissue area.

    Returns:
        tissue_mask (:class:`numpy.ndarray`):
            Binary tissue mask.

    """
    img = img.astype("uint8")  # ensure input image is uint8
    img = contrast_enhancer(img, low_p=2, high_p=98)  # Contrast  enhancement
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_lab = img_lab[:, :, 0] / 255.0  # Convert to range [0,1].
    tissue_mask = l_lab < threshold

    # check it's not empty
    if tissue_mask.sum() == 0:
        raise ValueError("Empty tissue mask computed.")

    return tissue_mask