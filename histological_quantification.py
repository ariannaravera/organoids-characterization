"""
ROI-based Cell Analysis Pipeline with Napari

This script provides an interactive pipeline to estimate cell size, 
segment channels, draw regions of interest (ROIs), and quantify cells 
and marker expression in microscopy images.

Author: Arianna Ravera
Date: 2025
"""

import os
import cv2
import csv
import napari
import tifffile
import numpy as np

from skimage.measure import label
from skimage.draw import polygon2mask
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QWidget, QMessageBox, QLineEdit, QLabel


# ---------------------------------------------------------------------------- #
# Helper Functions
# ---------------------------------------------------------------------------- #

def cellsizewindow(image: np.ndarray) -> int:
    """
    Open a Napari window to manually draw shapes representing typical cell sizes.
    Estimate the average cell area based on user-defined polygons.

    Parameters
    ----------
    image : np.ndarray
        2D microscopy image.

    Returns
    -------
    int
        Estimated average cell area (in pixels).
    """
    viewer = napari.Viewer()
    viewer.title = 'Set cell size'
    viewer.add_image(image, name='image')
    shape_layer = viewer.add_shapes(name='cells size')
    napari.run()

    areas = []
    mask = np.zeros(image.shape, dtype=np.uint8)

    for shape in shape_layer.data:
        shape = np.array(shape)
        if shape.ndim == 3:  # unwrap if extra dimension
            shape = shape[0]

        poly_mask = polygon2mask(image.shape, shape)
        areas.append(poly_mask.sum())
        mask |= poly_mask

    return int(np.mean(areas)) if areas else 0


class SaveButton(QWidget):
    """
    Custom PyQt widget to apply thresholding interactively
    and save masks from Napari.
    """
    def __init__(self, viewer: napari.Viewer, mask: np.ndarray, output_path: str):
        super().__init__()
        self.viewer = viewer
        self.output_path = output_path
        self.newmask = mask
        self.th_value = None
        # UI Elements
        label = QLabel("Insert the threshold value:")
        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText("Enter number")
        self.button = QPushButton('Save layer')
        self.button.clicked.connect(self.save_layer)
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def save_layer(self):
        """
        Apply thresholding based on user input and save result.
        """
        try:
            value = float(self.line_edit.text())
            image = self.viewer.layers['image'].data
            _, self.newmask = cv2.threshold(image, int(value), 255, cv2.THRESH_BINARY)
            self.viewer.add_labels(self.newmask, name='mask')

            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Mask saved successfully!")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

        except ValueError:
            print("Please enter a valid number.")

      
def manualcheck_mask(img: np.ndarray, name: str, output_path: str) -> np.ndarray:
    """
    Open Napari to manually correct a mask for a given channel.

    Parameters
    ----------
    img : np.ndarray
        Input image (single channel).
    name : str
        Channel name.
    output_path : str
        Path where the mask can be saved.

    Returns
    -------
    np.ndarray
        Corrected binary mask.
    """
    viewer = napari.Viewer()
    viewer.title = f'Segment {name} channel'
    viewer.add_image(img, name='image')

    # Initial rough threshold
    _, mask = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)
    viewer.add_labels(mask, name=name)

    # Add Save button
    save_button = SaveButton(viewer, mask, output_path)
    viewer.window.add_dock_widget(save_button, name='Save', area='left')
    viewer.show(block=True)

    return save_button.newmask


def ROIwindow(image: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    """
    Open Napari to draw Regions of Interest (ROIs).

    Parameters
    ----------
    image : np.ndarray
        Reference image (typically DAPI).
    masks : list[np.ndarray]
        Channel masks for visualization.

    Returns
    -------
    np.ndarray
        ROI mask (binary).
    """
    names = ['Ki67', 'NeuroD2', 'PAX6']

    viewer = napari.Viewer()
    viewer.add_image(image, name='image')

    for i, m in enumerate(masks):
        if len(m.shape) > 2:
            viewer.add_labels(m[:, :, 0], name=names[i], visible=False)
        else:
            viewer.add_labels(m, name=names[i], visible=False)

    shape_layer = viewer.add_shapes(name='ROI')
    napari.run()

    # Convert polygons to a binary ROI mask
    mask = np.zeros(image.shape, dtype=np.uint8)
    for shape in shape_layer.data:
        shape = np.array(shape)
        if shape.ndim == 3:
            shape = shape[0]
        poly_mask = polygon2mask(image.shape, shape)
        mask |= (poly_mask.astype(np.uint8) * 255)

    return mask


# ---------------------------------------------------------------------------- #
# Main Analysis
# ---------------------------------------------------------------------------- #

def analysis(input_path: str, output_path: str, ROI_outpath: str):
    """
    Run the full analysis pipeline on all .tif images in a folder.

    Parameters
    ----------
    input_path : str
        Path containing .tif microscopy images.
    output_path : str
        Directory to save results and masks.
    ROI_outpath : str
        Directory to save ROI masks.
    """
    # Prepare CSV results file
    results_file = os.path.join(output_path, 'results-ROI.csv')
    if not os.path.exists(results_file):
        with open(results_file, "w") as file:
            writer = csv.writer(file)
            writer.writerow(['image', 'ROI id', '#cells', '#Ki67', '#NeuroD2', '#PAX6', '#Ki67&PAX6'])

    # Loop over images
    for image_name in [x for x in os.listdir(input_path) if x.endswith('.tif')]:
        print(f'Analyzing image {image_name}')

        image = tifffile.imread(os.path.join(input_path, image_name))

        # Output paths
        output_mask_path = os.path.join(ROI_outpath, image_name.replace('.tif', '-ROI.tif'))

        if not os.path.exists(output_mask_path):
            # Estimate cell size
            estimated_area = cellsizewindow(image[0])
            print(f'Estimated cell size: {estimated_area} px²')

            # Segment channels
            channels_masks = np.zeros_like(image, dtype=np.uint8)
            channel_names = ['DAPI', 'Ki67', 'NeuroD2', 'PAX6']

            for i, ch in enumerate(image):
                if i == 0:  # DAPI
                    ch = cv2.convertScaleAbs(ch, alpha=5, beta=0)
                    blur = cv2.GaussianBlur(ch, (9, 9), 0)
                    _, channels_masks[i] = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                else:  # Other channels
                    blur = cv2.GaussianBlur(ch, (9, 9), 0)
                    channels_masks[i] = manualcheck_mask(blur, channel_names[i], output_path)

            # Save channel masks
            os.makedirs(os.path.join(output_path, 'masks_estimation'), exist_ok=True)
            tifffile.imwrite(os.path.join(output_path, 'masks_estimation', f'masks-{image_name}'), channels_masks.astype('uint8'), metadata={'axes': 'CYX'}, imagej=True)

            # Define ROIs
            ROIs_masks = ROIwindow(image[0], channels_masks[1:])
            ROIs_masks[ROIs_masks != 0] = 255

            # Quantification
            print('Running quantification...')
            labeledROIs_masks = label(ROIs_masks)

            for ROI_id in [x for x in np.unique(labeledROIs_masks) if x != 0]:
                ROI_mask = (labeledROIs_masks == ROI_id).astype(np.uint8) * 255

                results = [image_name.replace('.tif', ''), ROI_id]

                # Count cells per channel
                for i, _ in enumerate(image):
                    masked_mask = cv2.bitwise_and(channels_masks[i], channels_masks[i], mask=ROI_mask)
                    n_cells = int(np.count_nonzero(masked_mask) / estimated_area)
                    results.append(n_cells)

                # Double-positive cells (Ki67 & PAX6)
                ki67mask = cv2.bitwise_and(channels_masks[1], channels_masks[1], mask=ROI_mask)
                pax6mask = cv2.bitwise_and(channels_masks[3], channels_masks[3], mask=ROI_mask)
                maskcoloc = (ki67mask & pax6mask).astype(np.uint8)

                n_cells_coloc = int(np.count_nonzero(maskcoloc) / estimated_area)
                results.append(n_cells_coloc)

                # Save results
                with open(results_file, "a") as file:
                    writer = csv.writer(file)
                    writer.writerow(results)

            # Save ROI mask
            tifffile.imwrite(output_mask_path, ROIs_masks.astype(np.uint8))


# ---------------------------------------------------------------------------- #
# Script Entry Point
# ---------------------------------------------------------------------------- #

if __name__ == '__main__':
    main_dir = '/path/to/project'
    tif_data_path = os.path.join(main_dir, f'data/tif_version')
    output_analysis_path = os.path.join(main_dir, f'results')
    ROI_path = os.path.join(output_analysis_path, 'ROIs')
    # Create directories if they do not exist
    os.makedirs(output_analysis_path, exist_ok=True)
    os.makedirs(ROI_path, exist_ok=True)

    analysis(tif_data_path, output_analysis_path, ROI_path)