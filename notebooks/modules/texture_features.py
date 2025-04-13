# texture_features.py

import numpy as np
import pywt
from skimage.feature import graycomatrix, graycoprops

EPS = 1e-10  # small constant to avoid division by zero or log(0)

def quantize_roi(roi, levels=256, hu_range=(-1000, 400)):
    """
    Quantize an ROI from HU values to integer values in [0, levels-1].
    
    Parameters:
        roi (numpy.ndarray): 2D or 3D array of HU values.
        levels (int): Number of gray levels for quantization.
        hu_range (tuple): (hu_min, hu_max) for scaling the HU values.
    
    Returns:
        numpy.ndarray: Quantized ROI with integer values in [0, levels-1].
    """
    hu_min, hu_max = hu_range
    # Clip the values to the specified HU range.
    roi_clipped = np.clip(roi, hu_min, hu_max)
    # Normalize to [0,1] and then scale to [0, levels-1].
    roi_norm = (roi_clipped - hu_min) / (hu_max - hu_min)
    roi_quant = (roi_norm * (levels - 1)).astype(np.int32)
    return roi_quant

#######################
# GLCM Feature Extraction
#######################
def compute_glcm_features(roi, distances=[1], angles=[0], levels=256, hu_range=(-1000,400), symmetric=True, normed=True):
    """
    Compute GLCM texture features (Contrast, Homogeneity, Energy, Entropy, Dissimilarity) 
    from a CT scan ROI in HU.
    
    The ROI is quantized from HU values to integers in [0, levels-1] using the provided hu_range.
    
    Parameters:
        roi (numpy.ndarray): 2D or 3D array containing intensity values in HU.
        distances (list): List of pixel pair distance offsets.
        angles (list): List of angles in radians.
        levels (int): The number of gray levels for quantization.
        hu_range (tuple): (hu_min, hu_max) used for quantization.
        symmetric (bool): If True, the GLCM is symmetric.
        normed (bool): If True, normalize the GLCM.
    
    Returns:
        dict: Dictionary containing averaged GLCM features.
    """
    def _slice_glcm_features(image2d):
        # Quantize the 2D image (in HU) to integer levels.
        image = quantize_roi(image2d, levels=levels, hu_range=hu_range)
        glcm = graycomatrix(image, distances=distances, angles=angles, levels=levels,
                             symmetric=symmetric, normed=normed)
        contrast = graycoprops(glcm, 'contrast').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        # Compute entropy manually from the normalized GLCM.
        glcm_prob = glcm.astype(np.float64)
        glcm_prob /= (glcm_prob.sum() + EPS)
        entropy = -np.sum(glcm_prob * np.log(glcm_prob + EPS))
        return contrast, homogeneity, energy, entropy, dissimilarity

    contrasts, homogeneities, energies, entropies, dissimilarities = [], [], [], [], []
    
    if roi.ndim == 2:
        c, h, e, ent, d = _slice_glcm_features(roi)
        contrasts.append(c)
        homogeneities.append(h)
        energies.append(e)
        entropies.append(ent)
        dissimilarities.append(d)
    elif roi.ndim == 3:
        # Process each axial slice (assumes slices along the first dimension).
        for i in range(roi.shape[0]):
            slice_img = roi[i, :, :]
            c, h, e, ent, d = _slice_glcm_features(slice_img)
            contrasts.append(c)
            homogeneities.append(h)
            energies.append(e)
            entropies.append(ent)
            dissimilarities.append(d)
    else:
        raise ValueError("ROI must be 2D or 3D.")
    
    features = {
        'glcm_contrast': np.mean(contrasts),
        'glcm_homogeneity': np.mean(homogeneities),
        'glcm_energy': np.mean(energies),
        'glcm_entropy': np.mean(entropies),
        'glcm_dissimilarity': np.mean(dissimilarities)
    }
    return features

#######################
# GLRLM Feature Extraction
#######################
def compute_glrlm_features(roi, levels=256, hu_range=(-1000,400)):
    """
    Compute GLRLM features (Short-Run Emphasis, Long-Run Emphasis, Gray Level Non-Uniformity)
    from a CT scan ROI in HU.
    
    The ROI is quantized from HU values to integers in [0, levels-1] using the provided hu_range.
    This implementation computes the run-length matrix in the horizontal direction for each 2D slice
    and then averages the feature values.
    
    Parameters:
        roi (numpy.ndarray): 2D or 3D array containing intensity values in HU.
        levels (int): The number of gray levels for quantization.
        hu_range (tuple): (hu_min, hu_max) used for quantization.
    
    Returns:
        dict: Dictionary containing averaged GLRLM features.
    """
    def compute_glrlm_2d(image):
        """
        Compute the GLRLM for a single 2D image (horizontal runs only).
        Returns the run-length matrix (rows: gray levels, columns: run lengths).
        """
        # Quantize the image from HU to the specified range.
        image = quantize_roi(image, levels=levels, hu_range=hu_range)
        rows, cols = image.shape
        max_run = cols  # Maximum possible run length.
        glrlm = np.zeros((levels, max_run), dtype=np.int32)
        # Process each row.
        for r in range(rows):
            c = 0
            while c < cols:
                current_val = image[r, c]
                run_length = 1
                c_next = c + 1
                while c_next < cols and image[r, c_next] == current_val:
                    run_length += 1
                    c_next += 1
                # Update the run-length matrix (subtract 1 for zero-indexing of run lengths).
                glrlm[current_val, run_length - 1] += 1
                c = c_next
        return glrlm

    def _compute_features_from_glrlm(glrlm):
        total_runs = glrlm.sum() + EPS
        # Create indices for gray levels and run lengths.
        i_indices = np.arange(1, glrlm.shape[0] + 1)  # gray levels (1-indexed).
        j_indices = np.arange(1, glrlm.shape[1] + 1)  # run lengths (1-indexed).
        I, J = np.meshgrid(i_indices, j_indices, indexing='ij')
        # Short-Run Emphasis (SRE).
        sre = np.sum(glrlm / (J**2 + EPS)) / total_runs
        # Long-Run Emphasis (LRE).
        lre = np.sum(glrlm * (J**2)) / total_runs
        # Gray Level Non-Uniformity (GLNU).
        gl_sum = np.sum(glrlm, axis=1)
        glnu = np.sum(gl_sum**2) / total_runs
        return sre, lre, glnu

    sres, lres, glnus = [], [], []
    if roi.ndim == 2:
        glrlm = compute_glrlm_2d(roi)
        sre, lre, glnu = _compute_features_from_glrlm(glrlm)
        sres.append(sre)
        lres.append(lre)
        glnus.append(glnu)
    elif roi.ndim == 3:
        # Process each axial slice (assumes slices along the first dimension).
        for i in range(roi.shape[0]):
            slice_img = roi[i, :, :]
            glrlm = compute_glrlm_2d(slice_img)
            sre, lre, glnu = _compute_features_from_glrlm(glrlm)
            sres.append(sre)
            lres.append(lre)
            glnus.append(glnu)
    else:
        raise ValueError("ROI must be 2D or 3D.")

    features = {
        'glrlm_SRE': np.mean(sres),
        'glrlm_LRE': np.mean(lres),
        'glrlm_GLNU': np.mean(glnus)
    }
    return features

#######################
# Wavelet Feature Extraction
#######################
def compute_wavelet_features(roi, wavelet='db1', level=1):
    """
    Compute wavelet features (energy and entropy at multiple scales) from a 3D ROI.
    Uses an n-dimensional wavelet decomposition (wavedecn) from PyWavelets.
    
    Parameters:
        roi (numpy.ndarray): 3D array containing intensity values (in HU).
        wavelet (str): Wavelet type (e.g., 'db1').
        level (int): Number of decomposition levels.
    
    Returns:
        dict: Dictionary with wavelet energy and entropy for each level (averaged over detail subbands).
    """
    # Ensure ROI is float for decomposition.
    roi = roi.astype(np.float64)
    coeffs = pywt.wavedecn(roi, wavelet=wavelet, level=level)
    wavelet_features = {}
    # Iterate over each level's detail coefficients (levels 1...level).
    for lev in range(1, len(coeffs)):
        details = coeffs[lev]
        energies = []
        entropies = []
        # Each level contains a dictionary with keys for the detail subbands.
        for band, arr in details.items():
            coeff_flat = arr.ravel()
            energy = np.sum(coeff_flat**2)
            energies.append(energy)
            squared = coeff_flat**2
            total = np.sum(squared) + EPS
            p = squared / total
            entropy = -np.sum(p * np.log(p + EPS))
            entropies.append(entropy)
        wavelet_features[f'wavelet_energy_level_{lev}'] = np.mean(energies)
        wavelet_features[f'wavelet_entropy_level_{lev}'] = np.mean(entropies)
    return wavelet_features

#######################
# Combined Feature Extraction
#######################
def extract_texture_features(roi, glcm_params=None, glrlm_levels=256, wavelet='db1', wavelet_level=1, hu_range=(-1000,400)):
    """
    Extract a combined set of texture features (GLCM, GLRLM, Wavelet) from an ROI.
    
    Parameters:
        roi (numpy.ndarray): 2D or 3D array containing intensity values in HU.
        glcm_params (dict): Optional dictionary to override default GLCM parameters.
        glrlm_levels (int): Number of gray levels for GLRLM quantization.
        wavelet (str): Wavelet name for decomposition.
        wavelet_level (int): Decomposition level for wavelet features.
        hu_range (tuple): (hu_min, hu_max) used for quantization.
    
    Returns:
        dict: Dictionary of all extracted texture features.
    """
    # Set default parameters for GLCM if not provided.
    if glcm_params is None:
        glcm_params = {
            'distances': [1],
            'angles': [0],
            'levels': 256,
            'symmetric': True,
            'normed': True
        }
    features = {}
    # GLCM features (computed slice-wise if 3D).
    glcm_feats = compute_glcm_features(roi, **glcm_params, hu_range=hu_range)
    features.update(glcm_feats)
    
    # GLRLM features (computed slice-wise if 3D).
    glrlm_feats = compute_glrlm_features(roi, levels=glrlm_levels, hu_range=hu_range)
    features.update(glrlm_feats)
    
    # Wavelet features (only applicable for 3D ROI in this implementation).
    if roi.ndim == 3:
        wavelet_feats = compute_wavelet_features(roi, wavelet=wavelet, level=wavelet_level)
        features.update(wavelet_feats)
    else:
        # Optionally implement a 2D wavelet transform.
        features.update({})
    
    return features