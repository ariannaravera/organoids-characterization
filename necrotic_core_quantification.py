"""
Necrotic Core Quantification

This script performs Sholl analysis and cell quantification 
on multi-channel microscopy images. It estimates cell distributions 
around a center of mass and generates visual outputs.

Author: Arianna Ravera
Date: 2025
"""

import os
import cv2
import csv
import tifffile
import numpy as np
from tqdm import tqdm
from skimage.measure import label
from scipy.ndimage import center_of_mass

# ---------------------------------------------------------------------------- #
# Helper Functions
# ---------------------------------------------------------------------------- #


def sholl_analysis(mask: np.ndarray, mask_cs3: np.ndarray, radius_step_micron: float, output_path: str, image_name: str):
    """
    Perform Sholl analysis on a binary mask of cells.

    Parameters
    ----------
    mask : np.ndarray
        Labeled mask of cells (main channel).
    mask_cs3 : np.ndarray
        Labeled mask of Caspase3-positive cells.
    radius_step_micron : float
        Step of concentric rings in microns.
    output_path : str
        Path to save intermediate results (currently unused).
    image_name : str
        Name of the image (for output logging).

    Returns
    -------
    radii : np.ndarray
        Array of radii used for Sholl rings.
    cy, cx : float
        Coordinates of the center of mass.
    """
    avg_area = 220  # average cell area (px²), can be estimated from data

    # Compute center of mass
    binary_mask = (mask > 0).astype(np.uint8)
    cy, cx = center_of_mass(binary_mask)

    # Create distance map from center
    yy, xx = np.indices(mask.shape)
    dist_map = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    # Convert radius step from micron to pixels
    radius_step_px = 2.2 * radius_step_micron

    # Determine maximum radius for rings
    max_radius = int((mask.shape[0] / 2) // radius_step_px * radius_step_px)

    # Define concentric ring edges
    radii = np.arange(0, max_radius, radius_step_px)

    # Iterate over rings and count cells
    for i in range(len(radii) - 1):
        r_in, r_out = radii[i], radii[i + 1]

        # Binary mask for this ring
        ring_mask = np.logical_and(dist_map >= r_in, dist_map < r_out)

        # Mask the channels inside the ring
        masked = cv2.bitwise_and(mask, mask, mask=ring_mask.astype('uint8'))
        masked_cs3 = cv2.bitwise_and(mask_cs3, mask_cs3, mask=ring_mask.astype('uint8'))

        # Count Caspase3-positive cells based on area thresholds
        if avg_area > 0:
            unique_labels_cs3 = [x for x in np.unique(masked_cs3) if x != 0]
            ncs3 = 0
            min_th = 20
            max_th = 0.7 * avg_area
            for label_id in unique_labels_cs3:
                area = np.sum(masked_cs3 == label_id)
                if area < min_th:
                    continue
                elif area < max_th:
                    ncs3 += 1
                else:
                    ncs3 += round(area / max_th)
        else:
            ncs3 = 0

        # Results writing is currently commented out
        with open(os.path.join(output_path, 'results_necrotioncore.csv'), 'a') as file:
            writer = csv.writer(file)
            writer.writerow([image_name.replace('.tif',''), i, np.sum(ring_mask), round(np.sum(masked!=0)/avg_area), ncs3])

    return radii, cy, cx


def cell_analysis_estimation(input_path: str, output_path: str):
    """
    Process a folder of .tif images: segment cells, perform Sholl analysis,
    and generate visualization images.

    Parameters
    ----------
    input_path : str
        Path to input .tif images.
    output_path : str
        Path to save results and Sholl visualizations.
    """
    os.makedirs(os.path.join(output_path, 'sholl_new'), exist_ok=True)

    for image_name in tqdm([x for x in os.listdir(input_path) if x.endswith('.tif')]):
        sholl_outfile = os.path.join(output_path, 'sholl_new', image_name)
        if os.path.exists(sholl_outfile):
            continue  # skip already processed images

        # Load image
        image = tifffile.imread(os.path.join(input_path, image_name))

        # Segment main channel (DAPI)
        _, masks = cv2.threshold(cv2.GaussianBlur(image[0], (9, 9), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Segment Caspase3 channel
        _, mask_cs3 = cv2.threshold(cv2.GaussianBlur(image[1], (9, 9), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask_cs3 = label(mask_cs3)

        # Define Sholl step in microns
        radius_step_micron = 50

        # Perform Sholl analysis
        radii, cy, cx = sholl_analysis(masks, mask_cs3, radius_step_micron, output_path, image_name)

        # Create visualization images
        final_image = np.zeros((image.shape[0], image.shape[1], image.shape[2], 3), dtype=np.uint8)
        sholl_image = np.zeros((image.shape[1], image.shape[2]), dtype=np.uint8)

        for i, ch in enumerate(image):
            final_image[i, :, :, 0] = ch
            final_image[i, :, :, 1] = ch
            final_image[i, :, :, 2] = ch

            # Draw contours for first two channels
            mask_to_use = masks if i == 0 else mask_cs3
            if i in [0, 1]:
                contours, _ = cv2.findContours(mask_to_use.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(final_image[i], contours, -1, (255, 255, 0), 1)

            # Draw Sholl circles
            for r in radii:
                cv2.circle(final_image[i], (int(cx), int(cy)), int(r), (255, 0, 0), 1)
                cv2.circle(sholl_image, (int(cx), int(cy)), int(r), 255, 3)

            # Draw center of mass
            cv2.circle(final_image[i], (int(cx), int(cy)), 2, (0, 255, 0), -1)
            cv2.circle(sholl_image, (int(cx), int(cy)), 2, 200, -1)

        # Save Sholl visualization
        tifffile.imwrite(os.path.join(output_path, 'sholl_new', f'shollcircles-{image_name}'), sholl_image.astype('uint8'), imagej=True)


# ---------------------------------------------------------------------------- #
# Script Entry Point
# ---------------------------------------------------------------------------- #

if __name__ == '__main__':
    # Define base directories (update these paths as needed)
    main_dir = '/path/to/project'
    original_data_path = os.path.join(main_dir, 'data_necrotic_core')
    tif_data_path = os.path.join(original_data_path, 'tif_version')
    output_analysis_path = os.path.join(main_dir, 'results_necrotic_core')

    # Ensure output directories exist
    os.makedirs(tif_data_path, exist_ok=True)
    os.makedirs(output_analysis_path, exist_ok=True)

    # Run cell analysis
    cell_analysis_estimation(tif_data_path, output_analysis_path)