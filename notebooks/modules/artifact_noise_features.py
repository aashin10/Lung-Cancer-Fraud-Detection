# artifact_noise_features.py

import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
from skimage.morphology import ball
from scipy.fftpack import fft2, fftshift

EPS = 1e-10  # small constant to prevent division by zero

def get_bounding_box(mask):
    """
    Compute the bounding box of a 3D binary mask.
    Returns a tuple of slices for each dimension.
    """
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0) + 1  # include max index
    return tuple(slice(min_coords[i], max_coords[i]) for i in range(3))

def compute_checkerboard_artifact(ct_scan, nodule_mask):
    """
    Detect checkerboard artifacts in the nodule region using Fourier analysis.
    For each axial slice with nodule presence, compute a metric based on the ratio
    of the maximum FFT magnitude (excluding the DC component) to the median FFT magnitude.
    
    Parameters:
        ct_scan (numpy.ndarray): 3D CT scan in HU.
        nodule_mask (numpy.ndarray): 3D binary mask indicating the nodule.
    
    Returns:
        float: Average artifact metric across slices.
    """
    bbox = get_bounding_box(nodule_mask)
    if bbox is None:
        return 0
    ct_bbox = ct_scan[bbox]
    mask_bbox = nodule_mask[bbox]
    artifact_metrics = []
    # Process each axial slice (assumes first dimension is axial)
    for i in range(ct_bbox.shape[0]):
        if np.sum(mask_bbox[i]) == 0:
            continue
        slice_img = ct_bbox[i]
        fft_img = np.abs(fftshift(fft2(slice_img)))
        center = np.array(fft_img.shape) // 2
        fft_img[center[0]-1:center[0]+2, center[1]-1:center[1]+2] = 0
        max_val = fft_img.max()
        median_val = np.median(fft_img) + EPS
        artifact_metrics.append(max_val / median_val)
    return np.mean(artifact_metrics) if artifact_metrics else 0

# The compute_edge_sharpness function has been removed or omitted to avoid memory errors.
# If needed in the future, it can be reintroduced with additional memory management.

def compute_noise_pattern_consistency(ct_scan, nodule_mask, dilation_radius=5):
    """
    Compare noise statistics (variance) inside the nodule and in adjacent lung tissue.
    The adjacent tissue is defined as the dilated nodule region minus the nodule itself.
    
    Parameters:
        ct_scan (numpy.ndarray): 3D CT scan in HU.
        nodule_mask (numpy.ndarray): 3D binary mask of the nodule.
        dilation_radius (int): Radius for dilation (in voxels).
    
    Returns:
        dict: Contains nodule noise variance, adjacent tissue noise variance, and their ratio.
    """
    struct_elem = ball(dilation_radius)
    dilated_mask = binary_dilation(nodule_mask, structure=struct_elem)
    adjacent_mask = dilated_mask & (~nodule_mask)
    
    nodule_values = ct_scan[nodule_mask]
    adjacent_values = ct_scan[adjacent_mask]
    
    var_nodule = np.var(nodule_values) if nodule_values.size > 0 else 0
    var_adjacent = np.var(adjacent_values) if adjacent_values.size > 0 else 0
    noise_variance_ratio = var_nodule / (var_adjacent + EPS)
    
    return {
        'nodule_noise_variance': var_nodule,
        'adjacent_noise_variance': var_adjacent,
        'noise_variance_ratio': noise_variance_ratio
    }

def extract_artifact_noise_features(ct_scan, nodule_mask):
    """
    Extract a combined set of artifact and noise features from the CT scan and nodule mask.
    
    Features include:
      - Checkerboard Artifact Metric.
      - Noise Pattern Consistency (variance-based).
    
    (Note: Edge sharpness is omitted to avoid memory issues.)
    
    Parameters:
        ct_scan (numpy.ndarray): 3D CT scan in HU.
        nodule_mask (numpy.ndarray): 3D binary mask for the nodule.
    
    Returns:
        dict: Dictionary of extracted features.
    """
    features = {}
    features['checkerboard_artifact_metric'] = compute_checkerboard_artifact(ct_scan, nodule_mask)
    noise_feats = compute_noise_pattern_consistency(ct_scan, nodule_mask)
    features.update(noise_feats)
    return features


