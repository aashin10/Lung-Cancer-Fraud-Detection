# nodule_features.py

import numpy as np
from skimage.measure import marching_cubes
from scipy.spatial import ConvexHull
import numpy.linalg as la

def binarize_mask(mask):
    """
    Convert a mask to boolean if it is not already.
    Any nonzero voxel is considered part of the nodule.
    """
    if mask.dtype != bool:
        return mask != 0
    return mask

def compute_volume(mask, spacing=(1, 1, 1)):
    """
    Compute the volume of the nodule.
    :param mask: 3D numpy array (binary) where 1 indicates nodule voxels.
                 If the input mask is not boolean (e.g. derived from HU), it will be binarized.
    :param spacing: Tuple indicating the voxel spacing in each dimension.
    :return: Volume (in physical units if spacing provided).
    """
    mask = binarize_mask(mask)
    voxel_volume = np.prod(spacing)
    volume = np.sum(mask) * voxel_volume
    return volume

def compute_surface_area(mask, spacing=(1, 1, 1)):
    """
    Compute the surface area of the nodule using the marching cubes algorithm.
    :param mask: 3D numpy array (binary) where 1 indicates nodule voxels.
                 If the input mask is not boolean (e.g. derived from HU), it will be binarized.
    :param spacing: Tuple indicating the voxel spacing.
    :return: Estimated surface area.
    """
    mask = binarize_mask(mask)
    # For binary data, level=0.5 is a good choice.
    verts, faces, normals, values = marching_cubes(mask, level=0.5, spacing=spacing)
    surface_area = 0.0
    for face in faces:
        triangle = verts[face]
        # Compute area of the triangle using cross product method.
        a = triangle[1] - triangle[0]
        b = triangle[2] - triangle[0]
        area = 0.5 * np.linalg.norm(np.cross(a, b))
        surface_area += area
    return surface_area

def compute_sphericity(volume, surface_area):
    """
    Compute the sphericity of the nodule.
    Formula: sphericity = (pi^(1/3)*(6*Volume)^(2/3)) / Surface Area
    :param volume: Volume of the nodule.
    :param surface_area: Surface area of the nodule.
    :return: Sphericity measure.
    """
    if surface_area == 0:
        return 0
    sphericity = (np.pi ** (1/3) * (6 * volume) ** (2/3)) / surface_area
    return sphericity

def compute_elongation(mask, spacing=(1, 1, 1)):
    """
    Estimate the elongation and aspect ratio using the voxel coordinates' covariance.
    :param mask: 3D numpy array (binary) where 1 indicates nodule voxels.
                 If the input mask is not boolean (e.g. derived from HU), it will be binarized.
    :param spacing: Tuple indicating the voxel spacing.
    :return: (elongation, aspect_ratio) where:
             - elongation is the ratio of the second largest to the largest eigenvalue (in length scale),
             - aspect_ratio is the ratio of the largest to smallest eigenvalue.
    """
    mask = binarize_mask(mask)
    # Get voxel indices where the mask is True.
    coords = np.argwhere(mask)
    # Convert to physical space using voxel spacing.
    coords = coords * np.array(spacing)
    # Center the coordinates.
    centered = coords - np.mean(coords, axis=0)
    # Compute covariance matrix.
    cov = np.cov(centered, rowvar=False)
    eigenvalues, _ = np.linalg.eig(cov)
    # Sort eigenvalues in descending order.
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    # To convert variance to a length scale, take square roots.
    if eigenvalues[-1] <= 0:
        aspect_ratio = 0
    else:
        aspect_ratio = np.sqrt(eigenvalues[0]) / np.sqrt(eigenvalues[-1])
    
    if eigenvalues[0] <= 0:
        elongation = 0
    else:
        # A common measure: ratio of the second largest to the largest.
        elongation = np.sqrt(eigenvalues[1]) / np.sqrt(eigenvalues[0])
    return elongation, aspect_ratio

def box_counting_fractal_dimension(points, min_box_size, max_box_size, n_boxes=10):
    """
    Estimate the fractal dimension of a point cloud using the box-counting method.
    :param points: N x 3 array of points (e.g. surface vertices).
    :param min_box_size: Minimum box size.
    :param max_box_size: Maximum box size.
    :param n_boxes: Number of box sizes to evaluate.
    :return: Estimated fractal dimension (slope of log-log plot).
    """
    box_sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size), num=n_boxes)
    counts = []
    for box_size in box_sizes:
        # Shift points to positive coordinates.
        min_vals = points.min(axis=0)
        shifted = points - min_vals
        # Compute grid indices for each point.
        grid_indices = np.floor(shifted / box_size)
        unique_boxes = np.unique(grid_indices, axis=0)
        counts.append(len(unique_boxes))
    # Fit a line in log-log space.
    logsizes = np.log(1 / box_sizes)
    logcounts = np.log(counts)
    slope, _ = np.polyfit(logsizes, logcounts, 1)
    return slope

def compute_contour_irregularity(mask, spacing=(1, 1, 1)):
    """
    Estimate the contour irregularity (as a proxy for edge roughness) using fractal dimension.
    :param mask: 3D numpy array (binary) where 1 indicates nodule voxels.
                 If the input mask is not boolean (e.g. derived from HU), it will be binarized.
    :param spacing: Tuple indicating the voxel spacing.
    :return: Estimated fractal dimension of the surface.
    """
    mask = binarize_mask(mask)
    verts, faces, normals, values = marching_cubes(mask, level=0.5, spacing=spacing)
    if verts.size == 0:
        return 0
    # Determine the range of box sizes from the surface vertices.
    ptp = np.ptp(verts, axis=0)  # range in each dimension
    min_box_size = np.min(ptp) / 100.0  # 1% of the smallest range
    max_box_size = np.max(ptp)
    fractal_dim = box_counting_fractal_dimension(verts, min_box_size, max_box_size, n_boxes=10)
    return fractal_dim

def extract_nodule_features(mask, spacing=(1, 1, 1)):
    """
    Extract a set of features from a 3D nodule mask.
    :param mask: 3D numpy array indicating the nodule.
                 If the mask is not binary (e.g. it comes directly from a CT scan in HU),
                 nonzero values are treated as nodule voxels.
    :param spacing: Tuple of voxel spacings (default assumes isotropic voxels of size 1).
    :return: Dictionary of extracted features.
    """
    # Ensure the mask is binary.
    mask = binarize_mask(mask)
    
    volume = compute_volume(mask, spacing)
    surface_area = compute_surface_area(mask, spacing)
    sphericity = compute_sphericity(volume, surface_area)
    elongation, aspect_ratio = compute_elongation(mask, spacing)
    contour_irregularity = compute_contour_irregularity(mask, spacing)
    
    features = {
        'volume': volume,
        'surface_area': surface_area,
        'sphericity': sphericity,
        'elongation': elongation,
        'aspect_ratio': aspect_ratio,
        'contour_irregularity': contour_irregularity
    }
    return features
