import os
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import pickle

# Fixed CSV file path
CSV_PATH = os.path.join("..", "test", "CombinedCancerList.csv")

# ----- Default Metadata -----
DEFAULT_METADATA = {
    "slice_thickness": 2.5,       # mm (can be stored if needed)
    "slice_spacing": 2.5,         # mm (not used directly in HU conversion)
    "pixel_spacing": (0.7, 0.7),    # (not used directly in HU conversion)
    "origin": (0.0, 0.0, 0.0),
    "orientation": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    "rescale_intercept": -1024.0,   # Standard air HU value
    "rescale_slope": 1.0          # Default scaling factor
}

# ----- DICOM Loading and HU Conversion Functions -----
def load_scan(dicom_dir):
    """
    Load and sort all DICOM files in a directory by their z-axis location.
    
    Parameters:
        dicom_dir (str): Path to the directory containing DICOM files.
        
    Returns:
        List of sorted pydicom FileDataset objects.
    """
    slices = []
    for f in os.listdir(dicom_dir):
        if f.endswith(".dcm"):
            filepath = os.path.join(dicom_dir, f)
            try:
                ds = pydicom.dcmread(filepath)
                if hasattr(ds, "ImagePositionPatient"):
                    slices.append(ds)
            except Exception as e:
                print(f"Could not read file {filepath}: {e}")
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    return slices

def convert_to_hu(slices, rescale_intercept, rescale_slope):
    """
    Convert raw DICOM pixel data to Hounsfield Units (HU).
    
    Parameters:
        slices (list): Sorted list of DICOM slices.
        rescale_intercept (float): The intercept value.
        rescale_slope (float): The slope value.
        
    Returns:
        A 3D numpy array of the CT volume in HU.
    """
    image = np.stack([s.pixel_array for s in slices], axis=0).astype(np.int16)
    if rescale_slope != 1:
        image = image * rescale_slope
        image = image.astype(np.int16)
    image += np.int16(rescale_intercept)
    return image

def save_volume(output_dir, hu_volume, patient_id):
    """
    Save the processed CT scan volume as a NumPy file.
    
    Parameters:
        output_dir (str): Directory where the volume will be saved.
        hu_volume (numpy.ndarray): The 3D CT volume in HU.
        patient_id (str): Patient identifier used for the filename.
    """
    os.makedirs(output_dir, exist_ok=True)
    npy_filename = f"{patient_id}_full_volume.npy"
    npy_path = os.path.join(output_dir, npy_filename)
    np.save(npy_path, hu_volume)
    print(f"Saved CT scan (HU) for patient {patient_id} at: {npy_path}")

def process_patient_scan(patient_folder, patient_id, output_folder):
    """
    Process a single patient scan by loading DICOM slices, converting them to HU,
    and saving the resulting volume.
    
    Parameters:
        patient_folder (str): Directory containing the DICOM files.
        patient_id (str): Patient identifier.
        output_folder (str): Directory where the processed volume is saved.
    """
    print(f"\nProcessing patient {patient_id} from folder: {patient_folder}")
    slices = load_scan(patient_folder)
    if not slices:
        print(f"No valid DICOM files found in {patient_folder}. Skipping patient {patient_id}.")
        return
    hu_volume = convert_to_hu(
        slices,
        DEFAULT_METADATA["rescale_intercept"],
        DEFAULT_METADATA["rescale_slope"]
    )
    save_volume(output_folder, hu_volume, patient_id)

# ----- Cube Extraction and Display Functions -----
def extract_cube(ct_scan, center, cube_size=(128, 128, 128)):
    """
    Extracts a cube of given size from a 3D CT scan centered at a specified location.
    
    Parameters:
        ct_scan (numpy.ndarray): The 3D CT scan volume.
        center (tuple): (z, y, x) coordinates for the cube center.
        cube_size (tuple): Desired dimensions (depth, height, width) of the cube.
        
    Returns:
        numpy.ndarray: The extracted cube (padded if necessary).
    """
    z, y, x = center
    d, h, w = cube_size

    # Compute extraction boundaries
    z_min, z_max = max(0, z - d // 2), min(ct_scan.shape[0], z + d // 2)
    y_min, y_max = max(0, y - h // 2), min(ct_scan.shape[1], y + h // 2)
    x_min, x_max = max(0, x - w // 2), min(ct_scan.shape[2], x + w // 2)

    cube = ct_scan[z_min:z_max, y_min:y_max, x_min:x_max]

    # Determine required padding if extracted cube is smaller than desired size
    pad_z = d - cube.shape[0]
    pad_y = h - cube.shape[1]
    pad_x = w - cube.shape[2]

    pad_before_z = pad_z // 2
    pad_after_z = pad_z - pad_before_z
    pad_before_y = pad_y // 2
    pad_after_y = pad_y - pad_before_y
    pad_before_x = pad_x // 2
    pad_after_x = pad_x - pad_before_x

    cube = np.pad(
        cube,
        ((pad_before_z, pad_after_z),
         (pad_before_y, pad_after_y),
         (pad_before_x, pad_after_x)),
        mode='constant',
        constant_values=-1000  # pad with air HU value
    )
    return cube

def display_middle_slice(cube, patient_id, nodule_idx, predicted_class):
    """
    Displays the middle slice of an extracted cube along with classification info.
    
    Parameters:
        cube (numpy.ndarray): The extracted cube.
        patient_id (str): Patient identifier.
        nodule_idx (int): Nodule index (or row index from CSV).
        predicted_class (str): Model's prediction.
        ground_truth (str): Ground truth label from CSV.
    """
    middle_slice = cube[cube.shape[0] // 2]
    plt.figure(figsize=(6, 6))
    plt.imshow(middle_slice, cmap='gray')
    plt.colorbar(label="HU")
    plt.title(f"Patient: {patient_id}, Nodule: {nodule_idx}\nPrediction: {predicted_class}")
    plt.axis("off")
    plt.show()

# ----- Feature Extraction and Model Loading -----
def extract_features(cube):
    """
    Extract features from a cube. (This is a placeholder that uses simple statistics.)
    
    Parameters:
        cube (numpy.ndarray): The extracted cube.
        
    Returns:
        numpy.ndarray: Array of features (reshaped for model prediction).
    """
    mean_val = np.mean(cube)
    std_val = np.std(cube)
    min_val = np.min(cube)
    max_val = np.max(cube)
    features = np.array([mean_val, std_val, min_val, max_val])
    return features.reshape(1, -1)

def load_model(model_path):
    """
    Load the pre-trained random forest model from a pickle file.
    
    Parameters:
        model_path (str): Path to the pickle file.
        
    Returns:
        The loaded model.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Loaded model from {model_path}")
    return model

def classify_nodule(model, cube):
    """
    Extract features from a cube and classify the nodule using the provided model.
    
    Parameters:
        model: Pre-trained model.
        cube (numpy.ndarray): The extracted cube.
        
    Returns:
        The predicted class.
    """
    features = extract_features(cube)
    prediction = model.predict(features)[0]
    return prediction

# ----- Utility Function to Get CSV Data -----
def get_nodule_dataframe():
    """
    Reads the fixed CSV file containing nodule information and returns it as a DataFrame.
    
    Returns:
        pandas.DataFrame or None if the file is not found.
    """
    if not os.path.isfile(CSV_PATH):
        print(f"CSV file not found at {CSV_PATH}")
        return None
    return pd.read_csv(CSV_PATH)
