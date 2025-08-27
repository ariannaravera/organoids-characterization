"""
Lumen Zo1 Quantification

This script allows converting LIF files to TIFF, cleaning and manually
correcting Zo1 rosette masks, and quantifying rosettes in microscopy images.

Author: Arianna Ravera
Date: 2025
"""

import os
import cv2
import csv
import napari
import tifffile
import numpy as np
from qtpy.QtCore import Qt
from readlif.reader import LifFile
from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout, QSpinBox
from skimage.measure import regionprops, label
from scipy.ndimage import binary_dilation, binary_erosion, center_of_mass
from skimage.morphology import remove_small_objects, remove_small_holes, disk


# ---------------------------------------------------------------------------- #
# Helper Functions
# ---------------------------------------------------------------------------- #


def convert_lif_tif(input_path: str, output_path: str):
    """
    Convert all .lif microscopy files in a folder to TIFF images,
    normalized and saved in an ImageJ-compatible format.

    Parameters
    ----------
    input_path : str
        Folder containing .lif files.
    output_path : str
        Folder to save the converted .tif images.
    """
    # Prepare CSV for conversion metadata
    with open(os.path.join(output_path, 'conversions.csv'), "a") as file:
        writer = csv.writer(file)
        writer.writerow(['image', 'value (px/value = micron)'])
                
    # Loop over .lif files
    for image_name in [x for x in os.listdir(input_path) if x.endswith('.lif') and 'lifext' not in x and 'xml' not in x]:
        save_name = image_name
        print(f'Converting image {save_name}\n')

        lif_file = LifFile(os.path.join(input_path, image_name))
        for i in range(lif_file.num_images):
            lif_image = lif_file.get_image(i)

            if 'Merged' in lif_image.name:
                pixel_size = lif_image.info['scale']

                with open(os.path.join(output_path, 'conversions.csv'), "a") as file:
                    writer = csv.writer(file)
                    writer.writerow([save_name.replace(' ', '').replace('_.lif','') + '-' + lif_image.name.replace(' ', '').replace('/', '').replace('TileScan','').replace('Merged','') + '.tif', pixel_size])

                # Extract channels
                image = np.zeros((lif_image.info['channels'], lif_image.info['dims'].y, lif_image.info['dims'].x), dtype=np.uint8)
                for c in range(lif_image.info['channels']):
                    image[c,:,:] = lif_image.get_frame(c=int(c))

                # Normalize intensity to 0–255
                normalized_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

                # Save as TIFF
                save_path = os.path.join(output_path, save_name.replace(' ', '').replace('_.lif','') + '-' + lif_image.name.replace(' ', '').replace('/', '').replace('TileScan','').replace('Merged','') + '.tif')
                tifffile.imwrite(save_path, normalized_image, metadata={'axes': 'CYX'}, imagej=True)


def clean_binary_mask(mask: np.ndarray, min_size=100, hole_size=100, iterations=2) -> np.ndarray:
    """
    Clean a binary mask by keeping core areas, filling holes, and removing small noise.

    Parameters
    ----------
    mask : np.ndarray
        Input binary mask (2D, dtype=bool or 0/1).
    min_size : int
        Minimum object size to keep.
    hole_size : int
        Maximum hole size to fill.
    iterations : int
        Number of dilation/erosion steps.

    Returns
    -------
    np.ndarray
        Cleaned binary mask (0/1).
    """
    mask = mask.astype(bool)
    eroded = binary_erosion(mask, structure=disk(1), iterations=iterations)
    dilated = binary_dilation(eroded, structure=disk(1), iterations=iterations)
    filled = remove_small_holes(dilated, area_threshold=hole_size)
    cleaned = remove_small_objects(filled, min_size=min_size)
    return cleaned.astype(np.uint8)


def manualcorrection_mask(blur: np.ndarray, pax6: np.ndarray, mask: np.ndarray):
    """
    Open Napari for manual correction of Zo1 rosettes with user input for positive rosettes.

    Parameters
    ----------
    blur : np.ndarray
        Blurred Zo1 image.
    pax6 : np.ndarray
        PAX6 image for reference.
    mask : np.ndarray
        Initial binary mask.

    Returns
    -------
    corrected_mask : np.ndarray
        Corrected binary mask (1 where label == 2).
    n_positive : int
        User-entered number of PAX6-positive rosettes.
    """
    result = {"value": -1}

    viewer = napari.Viewer()
    viewer.add_image(blur, name='Zo1')
    viewer.add_image(pax6, name='PAX6', visible=False)

    label_layer = viewer.add_labels(mask, name='mask')
    label_layer.mode = 'paint'
    label_layer.selected_label = 2

    # Instructions dock widget
    label_widget = QLabel("🖌️ Paint and correct all the rosette with brush of value 2")
    label_widget.setAlignment(Qt.AlignCenter)
    label_widget.setStyleSheet("color: white; font-size: 14pt; background-color: rgba(0,0,0,150); padding: 6px; border-radius: 6px;")
    viewer.window.add_dock_widget(label_widget, area='top', name='Instructions')

    # Input widget for number of positive rosettes
    input_widget = QWidget()
    layout = QVBoxLayout()
    value_label = QLabel("Enter the number of PAX6-positive rosettes:")
    value_label.setAlignment(Qt.AlignLeft)

    spinbox = QSpinBox()
    spinbox.setMinimum(0)
    spinbox.setMaximum(9999)
    spinbox.setValue(0)

    layout.addWidget(value_label)
    layout.addWidget(spinbox)
    input_widget.setLayout(layout)
    viewer.window.add_dock_widget(input_widget, area='right', name='Parameter Input')

    # Capture spinbox value on close
    def on_close(event=None):
        result["value"] = spinbox.value()
    viewer.window._qt_window.closeEvent = on_close

    napari.run()

    updated_mask = label_layer.data
    corrected_mask = (updated_mask == 2).astype(np.uint8)
    return corrected_mask, result["value"]


def analyze_mask_and_region(dapi: np.ndarray, mask: np.ndarray, npositive: int, image_name: str, output_path: str):
    """
    Analyze rosettes: compute number, distance to DAPI center, area, and save results.

    Parameters
    ----------
    dapi : np.ndarray
        DAPI channel image.
    mask : np.ndarray
        Binary mask of rosettes.
    npositive : int
        Number of PAX6-positive rosettes.
    image_name : str
        Name of the image.
    output_path : str
        Folder to save results.
    """
    results_file = os.path.join(output_path, 'results.csv')
    if not os.path.exists(results_file):
        with open(results_file, "w") as file:
            writer = csv.writer(file)
            writer.writerow(['image', '#rosette', '#positive', 'avg distance', 'avg area', 'distances', 'areas'])

    ch = cv2.convertScaleAbs(dapi, alpha=3, beta=0)
    blur = cv2.GaussianBlur(ch, (9, 9), 0)
    _, DAPI_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    labeled_mask = label(mask)
    dapi_center = np.array(center_of_mass(DAPI_mask))
    regions = regionprops(labeled_mask)

    region_distances = []
    region_areas = []
    for region in regions:
        centroid = np.array(region.centroid)
        region_distances.append(np.linalg.norm(centroid - dapi_center))
        region_areas.append(int(region.area))

    with open(results_file, "a") as file:
        writer = csv.writer(file)
        writer.writerow([
            image_name, len(regions), npositive,
            np.round(np.average(region_distances), 2),
            np.round(np.average(region_areas), 2),
            region_distances, region_areas
        ])


def zo1_segmentation(input_path: str, output_path: str, masks_path: str):
    """
    Segment Zo1 rosettes in TIFF images and perform analysis.

    Parameters
    ----------
    input_path : str
        Folder with TIFF images.
    output_path : str
        Folder to save results.
    masks_path : str
        Folder to save masks.
    """
    for image_name in [x for x in os.listdir(input_path) if x.endswith('.tif')]:
        output_mask_path = os.path.join(masks_path, image_name.replace('.tif', '-mask.tif'))
        print(f'Segmenting image {image_name}\n')

        image = tifffile.imread(os.path.join(input_path, image_name)).astype('uint8')
        if not os.path.exists(output_mask_path):
            ch = cv2.convertScaleAbs(image[1], alpha=3, beta=0)  # Zo1 channel
            blur = cv2.GaussianBlur(ch, (9, 9), 0)
            avg = blur[np.nonzero(blur)].mean()
            _, mask = cv2.threshold(blur, min(int(avg*6), 200), 255, cv2.THRESH_BINARY)
            mask = clean_binary_mask(mask, min_size=20, hole_size=10, iterations=2)
            rosettes_mask, positive_rosette = manualcorrection_mask(blur, image[2], mask)
            tifffile.imwrite(output_mask_path, rosettes_mask)
            analyze_mask_and_region(image[1], rosettes_mask, positive_rosette, image_name, output_path)


# ---------------------------------------------------------------------------- #
# Script Entry Point
# ---------------------------------------------------------------------------- #

if __name__ == '__main__':
    # Define base directories (update these paths as needed)
    main_dir = '/path/to/project'
    original_data_path = os.path.join(main_dir, 'data_Zo1')          # Folder with original .lif files
    tif_data_path = os.path.join(main_dir, 'tif_version')            # Folder to save converted TIFFs
    masks_path = os.path.join(main_dir, 'results/masks_zo1')         # Folder to save rosette masks
    output_analysis_path = os.path.join(main_dir, 'results')         # Folder to save analysis results
    
    # Create directories if they do not exist
    os.makedirs(tif_data_path, exist_ok=True)
    os.makedirs(output_analysis_path, exist_ok=True)
    os.makedirs(masks_path, exist_ok=True)

    # STEP 1: Convert .lif files to TIFF images
    convert_lif_tif(original_data_path, tif_data_path)

    # STEP 2: Segment and analyze Zo1 rosettes
    zo1_segmentation(tif_data_path, output_analysis_path, masks_path)