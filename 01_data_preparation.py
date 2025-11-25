import numpy as np
import os
import pandas as pd
from tqdm import tqdm # Used for a simple progress bar during file processing

# --- GLOBAL CONFIGURATION FOR REPOSITORY PORTABILITY ---
# IMPORTANT: The user must download the raw FatigueSet data and place the
# subfolders (e.g., 'Raw EEG samples', 'Alpha EEG samples') inside a folder
# named 'data' in the root directory of this repository.

# Base path points to the root directory where the 'fatigueset' data should reside
BASE_PATH = 'data/fatigueset'

# Output path where the processed 3D NumPy arrays will be saved
# Create a folder named 'processed_data' in your repository root
OUTPUT_PATH = 'processed_data' 
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# --------------------------------------------------------


# --- CORE FUNCTIONS FOR DATA PREPARATION ---

def load_csv_to_numpy(file_path):
    """
    Loads a single .csv file, cleans it, and converts it to a 2D NumPy array.
    This function handles the common issue of an 'Unnamed: 0' index column
    often created when saving DataFrames to CSV.
    """
    try:
        data = pd.read_csv(file_path)
        # Check and drop the unnecessary index column
        if 'Unnamed: 0' in data.columns:
            data = data.drop(columns=['Unnamed: 0'])
        return data.to_numpy()  # Convert DataFrame to NumPy array (Timepoints x Channels)
    except Exception as e:
        print(f"Error loading CSV file {file_path}: {e}")
        return None

def pad_array(array, target_length):
    """
    Pads a 2D NumPy array with zeros along the row axis (Timepoints) to a target length.
    
    Note on Padding: Zero-padding is used to standardize the variable sequence lengths
    for the CNN input. This approach preserves all data and is preferred over trimming,
    which risks discarding important temporal features of the EEG signal.
    """
    current_length = array.shape[0]
    if current_length < target_length:
        padding_needed = target_length - current_length
        # Pad with zeros at the bottom (axis 0)
        array = np.pad(array, ((0, padding_needed), (0, 0)), 
                       mode='constant', constant_values=0)
    # Note: No truncation is performed if current_length > target_length
    return array

def find_max_length(folder_path):
    """
    Determines the maximum number of timepoints (rows) across all CSV files in a folder.
    This length is used as the target for zero-padding.
    """
    max_length = 0
   """
    Determines the maximum number of timepoints (rows) across all CSV files in a folder.
    """
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    print(f"-> Scanning {len(files)} files for max sequence length in {os.path.basename(folder_path)}...")
    
    for file in files:
        file_path = os.path.join(folder_path, file)
        array_2d = load_csv_to_numpy(file_path)
        if array_2d is not None:
            max_length = max(max_length, array_2d.shape[0])
    return max_length

def create_3d_array(folder_path):
    """
    Processes all CSV files in a folder, pads them to the max length,
    and stacks them into a single 3D NumPy array (Samples x Timepoints x Channels).
    """
    # Step 1: Find the maximum length in the current folder
    max_length = find_max_length(folder_path)
    print(f"Max sequence length found: {max_length} timepoints.")
    
    # Step 2: Load, pad, and collect each 2D array
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    array_list = []
    
    # Use tqdm here to monitor the main loading and padding process
    for file in tqdm(files, desc=f"Processing {os.path.basename(folder_path)}"):
        file_path = os.path.join(folder_path, file)
        array_2d = load_csv_to_numpy(file_path)
        
        if array_2d is not None:
            padded_array_2d = pad_array(array_2d, max_length)
            array_list.append(padded_array_2d)

    # Step 3: Stack all 2D arrays into a 3D array
    # Shape will be (Number of Trials, Max Timepoints, Number of Channels)
    if array_list:
        return np.stack(array_list, axis=0)
    else:
        print(f"No valid CSV files found in {folder_path}. Returning None.")
        return None

# --- MAIN EXECUTION BLOCK ---

def process_fatigueset():
    """Defines paths, processes all data bands/levels, and saves the final 3D arrays."""
    
    print("--- Starting FatigueSet Data Preprocessing (Phase 1) ---")
    
  # Defines all 18 combinations of signal type and fatigue intensity. 
    data_sets = []
    bands = ['Raw EEG', 'Alpha EEG', 'Beta EEG', 'Gamma EEG', 'Delta EEG', 'Theta EEG']
    intensities = ['Low Intensity', 'Medium Intensity', 'High Intensity']

    # Dynamically generate all folder paths and their corresponding output names
   for band in bands:
        for intensity in intensities:
            # Constructs the folder path relative to BASE_PATH
            folder_name = f"{band} samples/{intensity}"
            input_folder = os.path.join(BASE_PATH, folder_name)
            
            # Creates a clean output filename (e.g., array_3D_raw_low.npy)
            band_abbr = band.lower().split(' ')[0]
            intensity_abbr = intensity.lower().split(' ')[0]
            output_name = f"array_3D_{band_abbr}_{intensity_abbr}.npy"
            
            data_sets.append({
                'input_path': input_folder,
                'output_name': output_name
            })

    # Process all datasets in the list
    for data_set in data_sets:
        input_folder = data_set['input_path']
        output_name = data_set['output_name']
        output_path = os.path.join(OUTPUT_PATH, output_name)
        
        print(f"\nProcessing: {os.path.basename(input_folder)}")
        
        # Check if the folder exists before attempting to process
        if not os.path.exists(input_folder):
            print(f"WARNING: Input folder not found at {input_folder}. Skipping.")
            continue
        
        # Create the 3D array
        array_3d = create_3d_array(input_folder)
        
        if array_3d is not None:
            # Save the 3D array locally as a NumPy file
            np.save(output_path, array_3d)
            print(f"SUCCESS: Saved {array_3d.shape} to {output_path}")

    print("\n--- FatigueSet Preprocessing Complete. ---")
    print(f"All 3D NumPy arrays saved to the '{OUTPUT_PATH}' directory.")


# Execute the main processing function
if __name__ == "__main__":
    process_fatigueset()
    
