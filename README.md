# Organoids Characterization

This repository contains **four distinct Python pipelines** for analyzing microscopy images, including cell quantification, organoid morphology, Zo1 rosette segmentation, and ROI-based analysis with Napari.

## 1. Necrotic Core Quantification
**Purpose:**
Quantifies cell distributions around a center of mass and performs **Sholl analysis** on multi-channel microscopy images. Generates visual outputs with Sholl circles and contours.

**Key Features:**
- Compute center of mass of cells.  
- Count Caspase3-positive cells in concentric rings.  
- Generate visualizations with cell contours and Sholl rings.  
- Process folders of `.tif` images automatically.

## 2. Organoid Morphology
**Purpose:**
Analyzes grayscale images of organoids, extracting morphological features including area, perimeter, circularity, curvature, and Dirichlet Normal Energy (DNE). Saves results to Excel and generates contour visualizations.

**Key Features:**
-	Segment organoids using adaptive thresholding and morphological operations.
-	Compute features: area, perimeter, average radius, circularity, roundness, Feret diameters, curvature, DNE, and transparency.
-	Smooth contours with Gaussian filters.
-	Output results to Excel per image.

## 3. Lumen Zo1 Quantification
**Purpose:**
Converts .lif microscopy files to TIFF, segments Zo1 rosettes, allows manual corrections in Napari, and quantifies rosettes including PAX6-positive ones.

**Key Features:**
-	Convert .lif files to TIFF images with normalization.
-	Clean binary masks using morphological operations.
-	Napari-based manual correction of rosettes.
-	Quantify number, area, and distance of rosettes to DAPI center.
-	Save results to CSV.

## 4. Histological Quantification
**Purpose:**
Interactive pipeline to draw Regions of Interest (ROIs), segment channels, estimate cell size, and quantify cells and marker expression per ROI.

**Key Features:**
-	Napari-based manual cell size estimation.
-	Segment multiple channels (DAPI, Ki67, NeuroD2, PAX6).
-	Draw ROIs interactively and save masks.
-	Quantify single and double-positive cells per ROI.
-	Save results to CSV.

---

## Getting Started

### Dependencies

Before installing, ensure you have the following prerequisites:

- Operating System: Linux (Ubuntu 20.04 or later) / macOS / Windows 10+
- Python 3.12.4
- [Git](https://git-scm.com/downloads)
- (Optional) [Anaconda/Miniconda](https://docs.conda.io/en/latest/miniconda.html) for environment management

Main Python libraries used include:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- jupyterlab

(see full list in `requirements.txt`)

---

### Installing
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
   
Now you can run the Python scripts, eg.:
```bash
python histological_quantification.py
```
---

## Authors

[Arianna Ravera](ariannaravera22@gmail.com)
