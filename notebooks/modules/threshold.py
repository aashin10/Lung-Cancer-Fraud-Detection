#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os


def extract_lung_mask(volume, threshold=-1000, dilation_iterations=1):
    """
    Extracts a lung mask from a 3D CT volume using thresholding,
    connected component analysis (excluding border-touching regions),
    and binary dilation.

    For CT volumes in HU format, air is typically represented by -1000 HU.

    Parameters:
        volume (np.ndarray): 3D CT volume.
        threshold (int or float): Threshold value; voxels below this are considered as air.
        dilation_iterations (int): Number of iterations for binary dilation.

    Returns:
        np.ndarray: Binary mask (uint8) of the extracted lung region.
    """
    # Step 1: Thresholding the volume using the HU value for air.
    binary_mask = volume < threshold

    # Step 2: Connected component analysis using a 26-connected neighborhood.
    structure = np.ones((3, 3, 3), dtype=int)
    labeled_array, num_features = ndimage.label(binary_mask, structure=structure)

    if num_features == 0:
        raise ValueError("No connected components found in the volume with the given threshold.")

    # Step 3: Exclude components that touch the border (assumed to be external air).
    border_labels = set()
    border_labels.update(np.unique(labeled_array[0, :, :]))
    border_labels.update(np.unique(labeled_array[-1, :, :]))
    border_labels.update(np.unique(labeled_array[:, 0, :]))
    border_labels.update(np.unique(labeled_array[:, -1, :]))
    border_labels.update(np.unique(labeled_array[:, :, 0]))
    border_labels.update(np.unique(labeled_array[:, :, -1]))

    # Count the size of each component
    component_sizes = np.bincount(labeled_array.ravel())
    # Zero out the sizes of any component touching the border
    for label in border_labels:
        component_sizes[label] = 0

    if np.all(component_sizes == 0):
        raise ValueError("No non-border connected components found with the given threshold.")

    # Find the largest connected component from the remaining components.
    largest_component_label = component_sizes.argmax()

    # Create a mask for the largest component (assumed to be the lung region)
    lung_mask = (labeled_array == largest_component_label)

    # Step 4: Binary dilation to slightly expand the boundary.
    lung_mask_dilated = ndimage.binary_dilation(lung_mask, iterations=dilation_iterations)

    return lung_mask_dilated.astype(np.uint8)


def display_overlay(ct_slice, mask_slice):
    """
    Displays a 2D CT slice with the corresponding binary mask outlined in green.

    Parameters:
        ct_slice (np.ndarray): 2D array representing the CT image slice.
        mask_slice (np.ndarray): 2D binary array representing the segmented region.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(ct_slice, cmap='gray')
    # Overlay the contour of the mask over the CT image
    plt.contour(mask_slice, levels=[0.5], colors='green', linewidths=1)
    plt.title("CT Middle Axial Slice with Extracted Lung Region")
    plt.axis('off')
    plt.show()


def main():
    # Define file paths based on the patient ID
    patient_id = "3246"
    ct_volume_path = f"../../test_processed/{patient_id}/{patient_id}_full_volume.npy"
    threshold_volume_path = f"../../test_processed/{patient_id}/threshold_volume.npy"

    # Check if the CT volume file exists
    if not os.path.exists(ct_volume_path):
        print(f"CT volume file not found: {ct_volume_path}")
        return

    # Load the CT volume (assumed to be in HU format)
    ct_volume = np.load(ct_volume_path)

    # Extract the lung mask using the HU threshold for air (-1000)
    lung_mask = extract_lung_mask(ct_volume, threshold=-1000, dilation_iterations=1)

    # Save the extracted lung mask as threshold_volume.npy
    np.save(threshold_volume_path, lung_mask)
    print(f"Threshold lung mask saved to {threshold_volume_path}")

    # Display the overlay on the middle axial slice.
    mid_slice_index = ct_volume.shape[0] // 2
    ct_slice = ct_volume[mid_slice_index, :, :]
    mask_slice = lung_mask[mid_slice_index, :, :]
    display_overlay(ct_slice, mask_slice)


if __name__ == '__main__':
    main()
