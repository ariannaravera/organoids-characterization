# Organoids Characterization

This repository contains **four distinct Python pipelines** for analyzing microscopy images, including cell quantification, organoid morphology, Zo1 rosette segmentation, and ROI-based analysis with Napari.

---

## Table of Contents
 
1. [Installation](#installation)  
2. [Usage](#usage)  
3. [Sholl and Cell Analysis Pipeline](#sholl-and-cell-analysis-pipeline)  
4. [Organoid Morphology Analysis Pipeline](#organoid-morphology-analysis-pipeline)  
5. [Zo1 Rosette Segmentation and Analysis Pipeline](#zo1-rosette-segmentation-and-analysis-pipeline)  
6. [ROI-based Cell Analysis Pipeline with Napari](#roi-based-cell-analysis-pipeline-with-napari) 

---

## 📖 Installation Guide


1. Clone the repository
```bash
git clone https://github.com/your-organization/your-repository.git
cd your-repository
```

2. Create and activate a virtual environment

It is strongly recommended to use a virtual environment to avoid conflicts with other Python projects.

Option 1: Using pip + venv
```bash
python3 -m venv organoids_characterization
source organoids_characterization/bin/activate   # on macOS/Linux
organoids_characterization\Scripts\activate      # on Windows
```
Option 2: Using Conda
```bash
conda create -n organoids_characterization python=3.11
conda activate organoids_characterization
```

3. Install dependencies
   
Once the virtual environment is active, install all required packages:
```bash
pip install -r requirements.txt
```

4. Run the code
   
Now you can run the Python scripts or launch the main program depending on your project:

---
**Purpose:**  
Quantifies cell distributions around a center of mass and performs **Sholl analysis** on multi-channel microscopy images. Generates visual outputs with Sholl circles and contours.

**Key Features:**  
- Compute center of mass of cells.  
- Count Caspase3-positive cells in concentric rings.  
- Generate visualizations with cell contours and Sholl rings.  
- Process folders of `.tif` images automatically.

**Usage Example:**
```python
from sholl_pipeline import cell_analysis_estimation

input_path = "/path/to/tif_images"
output_path = "/path/to/results"
cell_analysis_estimation(input_path, output_path)
```
