# Mental Fatigue Detection using EEG Signals: A 1D-CNN Approach

**Candidate:** Bhavna Kakkar  
**Degree:** M.A. Clinical Psychology

---

## 1. Project Overview

This repository hosts the code and documentation for a research project focused on the **objective classification of mental fatigue levels** (Low, Medium, High) using electroencephalogram (EEG) signals.

The core methodology is divided into two phases:
1.  **Data Engineering (Phase 1):** Implementing a robust pipeline to standardize raw EEG data.
2.  **Model Development (Phase 2):** Building and training a custom **1-Dimensional Convolutional Neural Network (1D-CNN)** model for automated feature extraction and classification.

The full project proposal, detailing the research objectives, methodology, and proposed architecture, is available in the root directory.

---

## 2. Repository Structure (Local Environment)

The project is structured to separate source code from external and generated data, which is essential for reproducibility and version control.

```
/mental-fatigue-repo
  |- README.md
  |- .gitignore               <-- Ensures data/ and processed_data/ are NOT uploaded to Git.
  |- 01_data_preparation.py   <-- Code for Phase 1 (Data Standardization)
  |- 02_1dcnn_model_training.py <-- Code for Phase 2 (Planned Model Implementation)
  |- /data                    <-- REQUIRED: Local directory for external raw data
     |- /fatigueset           <-- External Content (Must be placed here by user)
        |- /Raw EEG samples
        |  ... (All original sub-folders)
  |- /processed_data            <-- Generated OUTPUT 3D arrays
```

---

## 3. Data Source and Setup

### Data Source

This project utilizes the publicly available **Mental Fatigue Level Detection (FatigueSet)** dataset. The raw data files are **not** included in this repository due to licensing and size restrictions.

### Citing Publication

* **Dataset Link:** [FatigueSet Data Repository](https://www.esense.io/datasets/fatigueset/)
* **Publication:** Manasa Kalanadhabhatta, Chulhong Min, Alessandro Montanari and Fahim Kawsar. *FatigueSet: A Multi-modal Dataset for Modeling Mental Fatigue and Fatigability*. In 15th International Conference on Pervasive Computing Technologies for Healthcare (Pervasive Health), December 6–8, 2021.

### Setup Instructions (Mandatory)

To run the data preparation script, you must manually set up the data environment:

1.  **Dependencies:** Install the required Python libraries:
    ```bash
    pip install numpy pandas tqdm tensorflow
    ```
2.  **Data Placement:** After downloading the FatigueSet data, create a folder named **`data`** in the root of this repository. Then, place all the dataset's sub-folders (e.g., `Raw EEG samples`, `Alpha EEG samples`, etc.) into a sub-folder named **`fatigueset`** within the new `data` directory.

---

## 4. Codebase Breakdown

### `01_data_preparation.py`

**Status:** Phase 1: Complete

**Description:** Complete data preprocessing pipeline for the FatigueSet EEG data. Implements **zero-padding** to convert variable-length trials into the standardized **3D tensor input** required for the 1D-CNN model, ensuring full project reproducibility.

**Execution:**

```bash
python 01_data_preparation.py

```

**Outcome:** The script saves 18 standardized 3D NumPy arrays into the `/processed_data` folder, making the data model-ready.

### `02_1dcnn_model_training.py`

**Status:** Phase 2: Planned

**Description:** This script will implement the proposed sequential **1D-CNN architecture**. It will be responsible for loading the processed 3D arrays, managing data partitioning, handling class labels, and executing the model training and formal evaluation against standard machine learning metrics. 
