import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, binary_fill_holes
from collections import deque

# ----------------------------------------------------------------
# Global Constants
# ----------------------------------------------------------------
# Path to the combined CSV file (columns: uuid, slice, x, y, class)
CSV_PATH = os.path.join("..", "test", "CombinedCancerList.csv")
# Base directory where processed CT volumes are stored.
# Expected structure: ../data_processed/{class}/{patient_id}/{patient_id}_full_volume.npy
DATA_PROCESSED_ROOT = "../data_processed"

# ----------------------------------------------------------------
# Magic Wand Functions
# ----------------------------------------------------------------
def coord_polar_to_cart(r, theta, center):
    """Converts polar coordinates (r, theta) around a center to Cartesian."""
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]
    return x, y

def coord_cart_to_polar(x, y, center):
    """Converts Cartesian coordinates to polar coordinates relative to a center."""
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    theta = np.arctan2((y - center[1]), (x - center[0]))
    return r, theta

def image_cart_to_polar(image, center, min_radius, max_radius, phase_width, zoom_factor=1):
    """
    Converts an image from Cartesian to polar coordinates around a given center.
    """
    if zoom_factor != 1:
        image = zoom(image, (zoom_factor, zoom_factor), order=4)
        center = (center[0] * zoom_factor + zoom_factor/2, center[1] * zoom_factor + zoom_factor/2)
        min_radius *= zoom_factor
        max_radius *= zoom_factor

    # Pad if necessary
    max_x, max_y = image.shape
    pad_dist_x = np.max([ (center[0] + max_radius) - max_x, -(center[0] - max_radius) ])
    pad_dist_y = np.max([ (center[1] + max_radius) - max_y, -(center[1] - max_radius) ])
    pad_dist = int(np.max([0, pad_dist_x, pad_dist_y]))
    if pad_dist != 0:
        image = np.pad(image, pad_dist, mode='constant')

    theta, r = np.meshgrid(np.linspace(0, 2*np.pi, phase_width),
                           np.arange(min_radius, max_radius))
    x, y = coord_polar_to_cart(r, theta, center)
    x, y = np.round(x).astype(int), np.round(y).astype(int)
    polar = image[x, y]
    polar.reshape((max_radius - min_radius, phase_width))
    return polar

def mask_polar_to_cart(mask, center, min_radius, max_radius, output_shape, zoom_factor=1):
    """
    Converts a polar binary mask back to Cartesian coordinates and embeds it in a zero image.
    """
    if zoom_factor != 1:
        center = (center[0]*zoom_factor + zoom_factor/2, center[1]*zoom_factor + zoom_factor/2)
        min_radius *= zoom_factor
        max_radius *= zoom_factor
        output_shape = tuple([int(a * zoom_factor) for a in output_shape])
    
    image = np.zeros(output_shape, dtype=float)
    theta, r = np.meshgrid(np.linspace(0, 2*np.pi, mask.shape[1]),
                           np.arange(0, max_radius))
    x, y = coord_polar_to_cart(r, theta, center)
    x, y = np.round(x).astype(int), np.round(y).astype(int)
    x = np.clip(x, 0, image.shape[0]-1)
    y = np.clip(y, 0, image.shape[1]-1)
    image[x, y] = mask
    if zoom_factor != 1:
        zf = 1/float(zoom_factor)
        image = zoom(image, (zf, zf), order=4)
    image = (image > 0.5).astype(int)
    image = binary_fill_holes(image)
    return image

def find_edge_2d(polar, min_radius):
    """
    Uses dynamic programming to detect the edge in a 2D polar image.
    Returns both the list of edge points and a binary mask.
    """
    if polar.ndim != 2:
        raise ValueError("Input must be 2D.")
    
    # Create shifted versions for dynamic programming
    values_right_shift      = np.pad(polar, ((0, 0), (0, 1)), mode='constant')[:, 1:]
    values_closeright_shift = np.pad(polar, ((1, 0), (0, 1)), mode='constant')[:-1, 1:]
    values_awayright_shift  = np.pad(polar, ((0, 1), (0, 1)), mode='constant')[1:, 1:]

    values_move = np.zeros((polar.shape[0], polar.shape[1], 3))
    values_move[:, :, 0] = polar + values_awayright_shift
    values_move[:, :, 1] = polar + values_right_shift
    values_move[:, :, 2] = polar + values_closeright_shift

    values = values_move.max(axis=2)
    directions = np.argmax(values_move, axis=2)
    directions = directions - 1
    directions = -directions

    edge = []
    mask = np.zeros(polar.shape, dtype=int)
    
    r_max = 0
    r = 0
    for i, v in enumerate(values[:, 0]):
        if v >= r_max:
            r, r_max = i, v
    edge.append((r + min_radius, 0))
    mask[0:r+1, 0] = 1

    for t in range(1, polar.shape[1]):
        r += directions[r, t-1]
        r = max(0, min(r, directions.shape[0]-1))
        edge.append((r + min_radius, t))
        mask[0:r+1, t] = 1

    new_mask = np.ones((min_radius + mask.shape[0], mask.shape[1]), dtype=int)
    new_mask[min_radius:, :] = mask
    return np.array(edge), new_mask

def edge_polar_to_cart(edge, center):
    """Converts a list of polar edge points to Cartesian edge points."""
    cart_edge = []
    for (r, t) in edge:
        x, y = coord_polar_to_cart(r, t, center)
        cart_edge.append((round(x), round(y)))
    return cart_edge

def cell_magic_wand_single_point(image, center, min_radius, max_radius,
                                 roughness=2, zoom_factor=1):
    """Applies the wand tool for a single seed point."""
    if roughness < 1:
        roughness = 1
    if min_radius < 0:
        min_radius = 0
    if max_radius <= min_radius:
        max_radius = min_radius + 1
    if zoom_factor <= 0:
        zoom_factor = 1
    
    phase_width = int(2 * np.pi * max_radius * roughness)
    polar_image = image_cart_to_polar(image, center, min_radius, max_radius,
                                      phase_width=phase_width, zoom_factor=zoom_factor)
    polar_edge, polar_mask = find_edge_2d(polar_image, min_radius)
    cart_edge = edge_polar_to_cart(polar_edge, center)
    cart_mask = mask_polar_to_cart(polar_mask, center, min_radius, max_radius,
                                   image.shape, zoom_factor=zoom_factor)
    return cart_mask, cart_edge

def cell_magic_wand(image, center, min_radius, max_radius,
                    roughness=2, zoom_factor=1, center_range=2):
    """
    Runs the magic wand tool on multiple perturbed seed points for robust segmentation.
    Returns a final binary mask.
    """
    centers = []
    for i in [-center_range, 0, center_range]:
        for j in [-center_range, 0, center_range]:
            centers.append((center[0] + i, center[1] + j))
    
    masks = np.zeros((image.shape[0], image.shape[1], len(centers)), dtype=float)
    for idx, c in enumerate(centers):
        mask_2d, edge = cell_magic_wand_single_point(
            image, c, min_radius, max_radius,
            roughness=roughness, zoom_factor=zoom_factor
        )
        masks[..., idx] = mask_2d
    mean_mask = np.mean(masks, axis=2)
    final_mask = (mean_mask > 0.5).astype(np.uint8)
    return final_mask

# ----------------------------------------------------------------
# 3D Cancer Mask Extraction via Magic Wand
# ----------------------------------------------------------------
def create_3d_cancer_mask(ct_volume, seed_slice, seed_point, min_radius, max_radius,
                          roughness=2, zoom_factor=1, center_range=2, slices_above_below=5):
    """
    Creates a 3D binary mask for a cancer nodule using Magic Wand segmentation.
    For the seed slice and several slices above/below, it segments the 2D slice.
    """
    num_slices, height, width = ct_volume.shape
    mask_volume = np.zeros_like(ct_volume, dtype=np.uint8)
    
    start_slice = max(0, seed_slice - slices_above_below)
    end_slice = min(num_slices, seed_slice + slices_above_below + 1)
    print(f"Processing slices {start_slice} to {end_slice - 1} for seed slice {seed_slice}.")
    
    for s in range(start_slice, end_slice):
        slice_img = ct_volume[s]
        mask_2d = cell_magic_wand(
            slice_img, seed_point, min_radius, max_radius,
            roughness=roughness, zoom_factor=zoom_factor, center_range=center_range
        )
        mask_volume[s] = mask_2d.astype(np.uint8)
    
    mask_volume = (mask_volume > 0).astype(np.uint8)
    if mask_volume.min() == mask_volume.max():
        print("WARNING: The final 3D mask is uniform.")
    return mask_volume

def display_mask_slice(mask_volume, slice_index):
    """Displays one slice from a 3D mask."""
    plt.figure(figsize=(6, 6))
    plt.imshow(mask_volume[slice_index], cmap='gray')
    plt.title(f"Magic Wand Mask on Slice {slice_index}")
    plt.axis("off")
    plt.show()

# ----------------------------------------------------------------
# Visualization Functions for Extracted 128x128 Region
# ----------------------------------------------------------------
def find_nodule_center(mask_volume):
    """
    Finds the center of the masked nodule region in 3D.
    Returns (z_center, y_center, x_center).
    """
    coords = np.argwhere(mask_volume > 0)
    if len(coords) == 0:
        raise ValueError("No nodule found in mask.")
    z_center, y_center, x_center = np.mean(coords, axis=0).astype(int)
    return z_center, y_center, x_center

def extract_128x128_region(ct_volume, mask_volume):
    """
    Extracts a 128x128 region centered on the nodule in the middle slice.
    Returns the cropped original CT slice, extracted nodule, and corresponding mask.
    """
    z_center, y_center, x_center = find_nodule_center(mask_volume)
    half_size = 64
    y_min, y_max = max(0, y_center - half_size), min(ct_volume.shape[1], y_center + half_size)
    x_min, x_max = max(0, x_center - half_size), min(ct_volume.shape[2], x_center + half_size)
    cropped_original = ct_volume[z_center, y_min:y_max, x_min:x_max]
    cropped_extracted = np.where(mask_volume[z_center, y_min:y_max, x_min:x_max] > 0,
                                 ct_volume[z_center, y_min:y_max, x_min:x_max],
                                 np.min(ct_volume))
    cropped_mask = mask_volume[z_center, y_min:y_max, x_min:x_max]
    return cropped_original, cropped_extracted, cropped_mask

def display_128x128_region(original, extracted, mask):
    """Displays the 128x128 cropped region from the original CT, extracted nodule, and mask."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Original 128x128 Region")
    axes[0].axis("off")
    axes[1].imshow(extracted, cmap='gray')
    axes[1].set_title("Extracted Nodule Region")
    axes[1].axis("off")
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title("Nodule Mask (128x128)")
    axes[2].axis("off")
    plt.show()

# ----------------------------------------------------------------
# Automated Seeded Region Growing (ASRG) Functions
# ----------------------------------------------------------------
def get_seed_point(roi_mask):
    """
    Automatically determine a seed point from the ROI mask
    (computes the centroid of nonzero voxels).
    Returns (z, y, x).
    """
    coords = np.argwhere(roi_mask > 0)
    if coords.size == 0:
        raise ValueError("ROI mask is empty. Cannot determine seed point.")
    centroid = np.mean(coords, axis=0)
    seed = tuple(np.round(centroid).astype(int))
    return seed

def region_growing(ct_volume, roi_mask, seed, threshold=50):
    """
    Performs region growing within the ROI.
    Voxels are added if they are within the ROI mask and their intensity
    difference from the seed intensity is within the threshold.
    """
    shape = ct_volume.shape
    visited = np.zeros(shape, dtype=bool)
    region_mask = np.zeros(shape, dtype=np.uint8)
    seed_intensity = ct_volume[seed]
    neighbor_offsets = [(1, 0, 0), (-1, 0, 0),
                        (0, 1, 0), (0, -1, 0),
                        (0, 0, 1), (0, 0, -1)]
    queue = deque()
    queue.append(seed)
    visited[seed] = True

    while queue:
        current = queue.popleft()
        region_mask[current] = 1
        for offset in neighbor_offsets:
            nz = current[0] + offset[0]
            ny = current[1] + offset[1]
            nx = current[2] + offset[2]
            if nz < 0 or nz >= shape[0] or ny < 0 or ny >= shape[1] or nx < 0 or nx >= shape[2]:
                continue
            if visited[nz, ny, nx]:
                continue
            if roi_mask[nz, ny, nx] == 0:
                continue
            if abs(int(ct_volume[nz, ny, nx]) - int(seed_intensity)) <= threshold:
                queue.append((nz, ny, nx))
                visited[nz, ny, nx] = True
    return region_mask

def display_extracted_region(ct_volume, region_mask, slice_index):
    """
    Displays a CT slice and its corresponding ASRG mask.
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(ct_volume[slice_index], cmap='gray')
    plt.title(f"CT Slice {slice_index}")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(region_mask[slice_index], cmap='gray')
    plt.title(f"ASRG Mask (Slice {slice_index})")
    plt.axis('off')
    plt.show()

# ----------------------------------------------------------------
# Processing Function for Each Patient's Nodules
# ----------------------------------------------------------------
def process_patient_nodules(class_label, patient_id, ct_volume, nodules_df,
                            min_radius=15, max_radius=20, roughness=2, zoom_factor=1,
                            center_range=2, slices_above_below=5, asrg_threshold=200):
    """
    For a given patient, processes each nodule (row in nodules_df) by:
      - Creating a 3D mask using the Magic Wand method.
      - Saving the magic wand mask as [nodule_number]_cmw_mask.npy.
      - Running ASRG on the mask to extract the nodule.
      - Saving the ASRG mask as [nodule_number]_asrg_mask.npy.
      - Displaying the ASRG result.
    """
    patient_dir = os.path.join(DATA_PROCESSED_ROOT, class_label, patient_id)
    for nodule_idx, (_, row) in enumerate(nodules_df.iterrows(), start=1):
        seed_slice = int(row["slice"])
        seed_point = (int(row["y"]), int(row["x"]))  # (row, column) in 2D
        print(f"Processing nodule {nodule_idx} for patient {patient_id}: seed_slice={seed_slice}, seed_point={seed_point}")
        
        # Create 3D mask using Magic Wand segmentation
        cancer_mask = create_3d_cancer_mask(ct_volume, seed_slice, seed_point,
                                            min_radius, max_radius, roughness, zoom_factor,
                                            center_range, slices_above_below)
        # Save Magic Wand mask
        cmw_mask_filename = f"{nodule_idx}_cmw_mask.npy"
        cmw_mask_path = os.path.join(patient_dir, cmw_mask_filename)
        np.save(cmw_mask_path, cancer_mask)
        print(f"Saved CMW mask: {cmw_mask_path}")
        # (Optional) Display one slice of the magic wand mask
        # display_mask_slice(cancer_mask, seed_slice)
        
        # Perform ASRG using the Magic Wand mask as ROI
        try:
            seed_asrg = get_seed_point(cancer_mask)
        except Exception as e:
            print(f"Error obtaining seed for ASRG in nodule {nodule_idx}: {e}")
            continue
        asrg_mask = region_growing(ct_volume, cancer_mask, seed_asrg, threshold=asrg_threshold)
        asrg_mask = (asrg_mask > 0).astype(np.uint8)
        asrg_mask_filename = f"{nodule_idx}_asrg_mask.npy"
        asrg_mask_path = os.path.join(patient_dir, asrg_mask_filename)
        np.save(asrg_mask_path, asrg_mask)
        print(f"Saved ASRG mask: {asrg_mask_path}")
        # Display the ASRG mask on the slice corresponding to the seed’s z-coordinate
        display_extracted_region(ct_volume, asrg_mask, seed_asrg[0])

# ----------------------------------------------------------------
# Main Function (for standalone execution)
# ----------------------------------------------------------------

