"""
lung_segmentation.py

This module performs lung segmentation on a DICOM series.
It uses thresholding, connected component analysis, hole filling, and
automated seeded region growing (ASRG) to extract the lung region.
The resulting binary mask is saved as 'lung_region.npy' and can be used
later (for example, in a template matching workflow).

Usage:
    python lung_segmentation.py
or import its functions in another module.
"""

import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import heapq
from scipy.ndimage import (
    binary_dilation,
    label,
    generate_binary_structure,
    binary_fill_holes
)

def load_dicom_series(folder_path):
    """
    Load a DICOM series from a folder and convert it to a 3D NumPy volume in Hounsfield Units (HU).
    """
    dcm_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith('.dcm')
    ]
    if not dcm_files:
        raise RuntimeError("No DICOM files found in folder: " + folder_path)

    # Read slices that contain SliceLocation (skip localizers)
    slices = []
    for dcm in dcm_files:
        ds = pydicom.dcmread(dcm)
        if hasattr(ds, 'SliceLocation'):
            slices.append(ds)
    # Sort slices based on SliceLocation
    slices.sort(key=lambda x: float(x.SliceLocation))
    
    volume = []
    for s in slices:
        arr = s.pixel_array.astype(np.int16)
        slope = getattr(s, 'RescaleSlope', 1.0)
        intercept = getattr(s, 'RescaleIntercept', 0.0)
        hu_slice = slope * arr + intercept
        volume.append(hu_slice)
    return np.array(volume, dtype=np.int16)

def remove_largest_boundary_component(binary_mask):
    """
    Remove the single largest connected component that touches the volume boundary.
    This helps remove the ambient air outside the patient.
    """
    struct = generate_binary_structure(3, 1)  # 6-connected structure
    labeled, n_features = label(binary_mask, structure=struct)
    if n_features == 0:
        return binary_mask

    zdim, ydim, xdim = binary_mask.shape
    components = []
    for comp_id in range(1, n_features + 1):
        comp_mask = (labeled == comp_id)
        size = np.sum(comp_mask)
        z, y, x = np.where(comp_mask)
        touches = (
            z.min() == 0 or z.max() == zdim - 1 or
            y.min() == 0 or y.max() == ydim - 1 or
            x.min() == 0 or x.max() == xdim - 1
        )
        components.append((comp_id, size, touches))
    
    boundary_components = [(cid, sz) for (cid, sz, tb) in components if tb]
    if not boundary_components:
        return binary_mask
    boundary_components.sort(key=lambda x: x[1], reverse=True)
    largest_boundary_id = boundary_components[0][0]
    cleaned_mask = binary_mask.copy()
    cleaned_mask[labeled == largest_boundary_id] = False
    return cleaned_mask

def apply_xy_bounding_box(binary_mask, frac_min=0.2, frac_max=0.9):
    """
    Zero out everything outside a central bounding box in the Y and X dimensions.
    """
    Z, Y, X = binary_mask.shape
    ymin, ymax = int(frac_min * Y), int(frac_max * Y)
    xmin, xmax = int(frac_min * X), int(frac_max * X)
    masked = binary_mask.copy()
    masked[:, :ymin, :] = False
    masked[:, ymax:, :] = False
    masked[:, :, :xmin] = False
    masked[:, :, xmax:] = False
    return masked

def keep_largest_components(binary_mask, num_keep=2):
    """
    Keep the largest 'num_keep' connected components.
    Typically used to retain the two lung regions.
    """
    struct = generate_binary_structure(3, 1)
    labeled, n_features = label(binary_mask, structure=struct)
    if n_features == 0:
        return binary_mask
    sizes = []
    for comp_id in range(1, n_features + 1):
        comp_size = np.sum(labeled == comp_id)
        sizes.append((comp_id, comp_size))
    sizes.sort(key=lambda x: x[1], reverse=True)
    keep_ids = [comp_id for comp_id, _ in sizes[:num_keep]]
    largest_mask = np.isin(labeled, keep_ids)
    return largest_mask

def expand_mask(mask, iterations=2):
    """
    Expand (dilate) a binary mask using a 3x3x3 structuring element.
    """
    struct = np.ones((3, 3, 3), dtype=np.uint8)
    return binary_dilation(mask, structure=struct, iterations=iterations)

def find_lung_wall(volume):
    """
    Find the lung wall in the middle slice along the x-axis.
    """
    z_mid = volume.shape[0] // 2
    x_mid = volume.shape[2] // 2
    slice_mid = volume[z_mid]
    lung_wall_start = None
    
    # Find non-air region in the middle slice along x-axis
    for y in range(slice_mid.shape[0]):
        if np.any(slice_mid[y, x_mid:] < -400):
            lung_wall_start = y
            break
    
    if lung_wall_start is None:
        raise RuntimeError("Failed to find lung wall on middle slice.")
    
    return lung_wall_start

def apply_dynamic_bounding_box(binary_mask, lung_wall_start):
    """
    Apply a dynamic bounding box based on lung wall detection.
    """
    Z, Y, X = binary_mask.shape
    ymin, ymax = lung_wall_start - 20, lung_wall_start + 20  # Adjust as needed
    ymin = max(0, ymin)
    ymax = min(Y, ymax)
    
    masked = binary_mask.copy()
    masked[:, :ymin, :] = False
    masked[:, ymax:, :] = False
    
    return masked

def find_seeds(volume, bounding_mask=None):
    """
    Automatically select two seed points based on the mean HU value.
    One seed is chosen from voxels with HU below the mean (likely air) and one above (likely tissue).
    Only voxels within the provided bounding_mask are considered.
    """
    mean_val = np.mean(volume)
    shape = volume.shape
    seed_low = None
    seed_high = None
    center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    max_range = min(shape) // 2
    for dz in range(-max_range, max_range, 5):
        for dy in range(-max_range, max_range, 5):
            for dx in range(-max_range, max_range, 5):
                z = center[0] + dz
                y = center[1] + dy
                x = center[2] + dx
                if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
                    if bounding_mask is not None and not bounding_mask[z, y, x]:
                        continue
                    val = volume[z, y, x]
                    if val < mean_val and seed_low is None:
                        seed_low = (z, y, x)
                    elif val > mean_val and seed_high is None:
                        seed_high = (z, y, x)
                    if seed_low is not None and seed_high is not None:
                        return seed_low, seed_high
    return center, center

def automated_SRG_with_mask(volume, seed_low, seed_high, bounding_mask):
    """
    Perform automated seeded region growing (ASRG) on the volume,
    restricted to the provided bounding_mask.
    Two regions are grown (labels 1 and 2) based on the two seeds.
    """
    labels = np.zeros_like(volume, dtype=np.uint8)
    neighbors = [(1, 0, 0), (-1, 0, 0),
                 (0, 1, 0), (0, -1, 0),
                 (0, 0, 1), (0, 0, -1)]
    # Initialize region statistics:
    region_sums = [0.0, float(volume[seed_low]), float(volume[seed_high])]
    region_counts = [0, 1, 1]
    labels[seed_low] = 1
    labels[seed_high] = 2
    pq = []  # Priority queue for region growing

    def push_neighbors(z0, y0, x0, region_label):
        r_mean = region_sums[region_label] / region_counts[region_label]
        for dz, dy, dx in neighbors:
            nz, ny, nx = z0 + dz, y0 + dy, x0 + dx
            if 0 <= nz < volume.shape[0] and 0 <= ny < volume.shape[1] and 0 <= nx < volume.shape[2]:
                if not bounding_mask[nz, ny, nx]:
                    continue
                if labels[nz, ny, nx] == 0:
                    val = volume[nz, ny, nx]
                    cost = abs(val - r_mean)
                    heapq.heappush(pq, (cost, nz, ny, nx, region_label))
    
    push_neighbors(*seed_low, 1)
    push_neighbors(*seed_high, 2)
    total_voxels = np.count_nonzero(bounding_mask)
    assigned = 2  # Already assigned two seeds
    
    while pq and assigned < total_voxels:
        cost, z, y, x, region_label = heapq.heappop(pq)
        if labels[z, y, x] == 0:
            labels[z, y, x] = region_label
            region_sums[region_label] += float(volume[z, y, x])
            region_counts[region_label] += 1
            push_neighbors(z, y, x, region_label)
            assigned += 1
    
    return labels

def get_lung_mask(volume):
    """
    Compute an initial lung mask from the CT volume using simple thresholding,
    hole filling, connected component analysis, and bounding box filtering.
    """
    # Basic thresholding: assume lung regions have HU less than -320
    binary_mask = volume < -320
    # Fill holes slice by slice
    filled_mask = np.zeros_like(binary_mask)
    for i in range(binary_mask.shape[0]):
        filled_mask[i] = binary_fill_holes(binary_mask[i])
    # Remove the largest connected component that touches the boundary (ambient air)
    cleaned_mask = remove_largest_boundary_component(filled_mask)
    # Apply an XY bounding box to focus on the central area
    bounded_mask = apply_xy_bounding_box(cleaned_mask)
    # Keep the largest two connected components (the lungs)
    lung_mask = keep_largest_components(bounded_mask, num_keep=2)
    return lung_mask

def segment_lungs(dicom_folder, output_dir):
    """
    Perform lung segmentation on a DICOM series in 'dicom_folder'.
    The segmentation uses thresholding, connected component analysis, and ASRG.
    The resulting lung region (binary mask) is saved as 'lung_region.npy' in output_dir.
    Also, an axial slice with overlay is displayed.
    """
    os.makedirs(output_dir, exist_ok=True)
    volume = load_dicom_series(dicom_folder)
    # Compute an initial lung mask using thresholding and connected component analysis.
    lung_mask = get_lung_mask(volume)
    # Expand the mask for a more robust region.
    expanded_mask = expand_mask(lung_mask, iterations=2)
    # Find lung wall on the volume's middle slice.
    lung_wall_start = find_lung_wall(volume)
    dynamic_bounding_mask = apply_dynamic_bounding_box(expanded_mask, lung_wall_start)
    # Determine seed points within the expanded mask.
    seed_low, seed_high = find_seeds(volume, bounding_mask=expanded_mask)
    # Perform automated seeded region growing (ASRG) restricted to the expanded mask.
    label_map = automated_SRG_with_mask(volume, seed_low, seed_high, expanded_mask)
    # Define the lung region as the void (air) region grown from the low seed (label 1)
    lung_region = (label_map == 1)
    # Save the lung region as a binary mask (numpy file)
    lung_mask_path = os.path.join(output_dir, "lung_region.npy")
    np.save(lung_mask_path, lung_region)
    print(f"Lung region saved to {lung_mask_path}")
    
    # Display a middle axial slice with segmentation overlay.
    z_mid = volume.shape[0] // 2
    plt.figure(figsize=(10, 6))
    plt.imshow(volume[z_mid], cmap='gray', vmin=-1000, vmax=500)
    overlay = np.zeros((volume.shape[1], volume.shape[2], 3), dtype=np.uint8)
    overlay[label_map[z_mid] == 1] = (255, 0, 0)  # Air region in red
    overlay[label_map[z_mid] == 2] = (0, 255, 0)  # Tissue region in green
    plt.imshow(overlay, alpha=0.4)
    plt.title(f"Axial Slice (z={z_mid}) with ASRG Segmentation\nRed: Air Region, Green: Tissue")
    plt.axis("off")
    plt.show()


