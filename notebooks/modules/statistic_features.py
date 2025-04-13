# intensity_features.py

import numpy as np
from scipy.stats import skew, kurtosis

def compute_statistical_moments(roi):
    """
    Compute basic statistical moments from a CT scan ROI in HU.
    
    Parameters:
        roi (numpy.ndarray): Region of interest from a CT scan (HU values).
    
    Returns:
        dict: Dictionary with mean, standard deviation, skewness, and kurtosis.
    """
    # Flatten the ROI to a 1D array.
    roi_flat = roi.flatten()
    mean_val = np.mean(roi_flat)
    std_val = np.std(roi_flat)
    skew_val = skew(roi_flat)
    kurtosis_val = kurtosis(roi_flat)
    
    return {
        'intensity_mean': mean_val,
        'intensity_std': std_val,
        'intensity_skewness': skew_val,
        'intensity_kurtosis': kurtosis_val
    }

def compute_intensity_histogram_features(roi, num_bins=50):
    """
    Compute intensity histogram features from a CT scan ROI in HU.
    
    Parameters:
        roi (numpy.ndarray): Region of interest from a CT scan (HU values).
        num_bins (int): Number of bins for the histogram.
    
    Returns:
        dict: Dictionary with histogram peak intensity, spread, and entropy.
    """
    roi_flat = roi.flatten()
    counts, bin_edges = np.histogram(roi_flat, bins=num_bins)
    
    # Determine the peak intensity (center of the bin with highest frequency).
    peak_index = np.argmax(counts)
    peak_value = (bin_edges[peak_index] + bin_edges[peak_index+1]) / 2.0
    
    # Define spread as the difference between the 95th and 5th percentiles.
    lower_percentile = np.percentile(roi_flat, 5)
    upper_percentile = np.percentile(roi_flat, 95)
    spread = upper_percentile - lower_percentile
    
    # Compute histogram entropy as a measure of distribution complexity.
    prob = counts.astype(np.float64) / (np.sum(counts) + 1e-10)
    hist_entropy = -np.sum(prob * np.log(prob + 1e-10))
    
    return {
        'histogram_peak_intensity': peak_value,
        'histogram_spread': spread,
        'histogram_entropy': hist_entropy
    }

def extract_intensity_features(roi, num_bins=50):
    """
    Extract combined intensity-based features from a CT scan ROI (in HU).
    
    Parameters:
        roi (numpy.ndarray): Region of interest from a CT scan (HU values).
        num_bins (int): Number of bins for histogram analysis.
    
    Returns:
        dict: Dictionary of intensity features including statistical moments and histogram features.
    """
    features = {}
    moments = compute_statistical_moments(roi)
    features.update(moments)
    
    hist_features = compute_intensity_histogram_features(roi, num_bins=num_bins)
    features.update(hist_features)
    
    return features
