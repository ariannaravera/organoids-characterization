"""
Organoid Morphology

This script processes grayscale images of organoids, segments them, computes
morphological features including area, perimeter, circularity, curvature, 
Dirichlet Normal Energy (DNE), and saves results in an Excel file along 
with visualization images showing organoid contours and IDs.

Author: Arianna Ravera
Date: 2025
"""

import os
import cv2
import math
import feret
import xlsxwriter
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import measure


# ---------------------------------------------------------------------------- #
# Helper Functions
# ---------------------------------------------------------------------------- #

def curv_helper(x0, y0, x1, y1, x2, y2):
    """
    Compute curvature using Heron's formula.
    """
    xa, ya = x0 - x1, y0 - y1
    xb, yb = x2 - x1, y2 - y1
    xc, yc = x2 - x0, y2 - y0
    a = np.sqrt(xa**2 + ya**2)
    b = np.sqrt(xb**2 + yb**2)
    c = np.sqrt(xc**2 + yc**2)
    delta = (xb * ya - xa * yb) / 2.0
    kappa = 4.0 * delta / (a * b * c)
    return kappa


def get_contour_curvature_geo(coords):
    """
    Compute curvature of a contour with periodic boundary conditions.
    """
    kappa = np.zeros(len(coords))
    for i in range(len(coords)):
        ip = (i - 1) % len(coords)
        in_ = (i + 1) % len(coords)
        kappa[i] = curv_helper(coords[ip][0], coords[ip][1],
                               coords[i][0], coords[i][1],
                               coords[in_][0], coords[in_][1])
    return kappa


def LoG_filter_opencv(image, sigma, size=None):
    """
    Compute Laplacian of Gaussian filter on the image.
    """
    if size is None:
        size = int(6 * sigma + 1) if sigma >= 1 else 7
    if size % 2 == 0:
        size += 1
    x, y = np.meshgrid(np.arange(-size//2+1, size//2+1),
                       np.arange(-size//2+1, size//2+1))
    kernel = -(1/(np.pi * sigma**4)) * (1 - ((x**2 + y**2) / (2 * sigma**2))) * \
             np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / np.sum(np.abs(kernel))
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image


def compute_dne(contours_coords):
    """
    Compute Dirichlet Normal Energy (DNE) for a contour.
    """
    dy = np.gradient([i[0] for i in contours_coords])
    dx = np.gradient([i[1] for i in contours_coords])
    normals = np.column_stack((dy, -dx))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    tangents = np.column_stack((dx, dy))
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    grad_nx = np.gradient(normals[:, 0])
    grad_ny = np.gradient(normals[:, 1])
    projections = grad_nx * tangents[:, 0] + grad_ny * tangents[:, 1]
    variation_squared = np.sum([x**2 for x in projections if not math.isnan(x)])
    dne = np.log(variation_squared + np.finfo(float).eps)
    return dne


def set_scale(typology, image_name):
    """
    Set conversion scale based on image typology.
    """
    if typology == '2x' or '2x' in image_name:
        return 1230/478
    elif typology == '4x' or '4x' in image_name:
        return 670/480
    else:
        raise ValueError("Typology must be '2x' or '4x'")


def segment_image(image):
    """
    Segment organoids in a grayscale image and return labeled mask.
    """
    val = np.average(image) + np.std(image)/2
    median = cv2.medianBlur(image, 5)
    _, mask = cv2.threshold(median, int(val), 255, cv2.THRESH_BINARY)
    mask = np.invert(mask)
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel, iterations=5)
    mask = ndimage.binary_fill_holes(mask).astype(int)
    mask = measure.label(mask, background=0)
    mask[mask == mask[0][0]] = 0
    mask[mask == mask[-1][-1]] = 0
    return mask.astype('uint16')


def smooth_contours(contours, result_image):
    """
    Smooth contour coordinates using Gaussian filter.
    """
    sig = 7
    coords = []
    for cnt in contours:
        x = ndimage.gaussian_filter(cnt[:, 0, 0], sig, mode='wrap')
        y = ndimage.gaussian_filter(cnt[:, 0, 1], sig, mode='wrap')
        for i in range(len(x)):
            coords.append((y[i], x[i]))
        cnt[:, 0, 0] = x
        cnt[:, 0, 1] = y
        cv2.drawContours(result_image, cnt, -1, (255, 0, 0), 3)
    return coords, result_image


def save_output_values(cell_area, coords, scale, radius, org_mask, masked_image, image_name, new_id, row, worksheet, result_image):
    """
    Save computed morphological features in Excel sheet.
    """
    area = cell_area * scale**2
    perimeter = np.sum([math.sqrt((coords[i][0]-coords[i+1][0])**2 + (coords[i][1]-coords[i+1][1])**2)
                        for i in range(len(coords)-1)]) * scale
    avg_radius = radius * scale
    roundness = 4 * (cell_area / (np.pi * (2*radius)**2))
    circularity = (4 * np.pi * cell_area)/np.power(perimeter, 2)
    maxf, minf, minf90, maxf90 = feret.all(org_mask)
    curvature = get_contour_curvature_geo(coords)
    mean_curvature = np.mean([x for x in curvature if not math.isnan(x)]) / scale
    std_curvature = np.std([x for x in curvature if not math.isnan(x)]) / scale
    std_curvature_norm = std_curvature * avg_radius
    dne = compute_dne(coords)
    transparency = np.count_nonzero(LoG_filter_opencv(masked_image, 1))/np.count_nonzero(masked_image)

    features = [image_name, new_id, area, perimeter, avg_radius,
                roundness*100, 100-(abs(1-circularity))*100,
                maxf*scale, minf*scale, transparency*100,
                mean_curvature, std_curvature, std_curvature_norm, dne]

    for i, val in enumerate(features):
        worksheet.write(row, i, val)
    row += 1
    return result_image, row


# ---------------------------------------------------------------------------- #
# Main Analysis
# ---------------------------------------------------------------------------- #


def main(input_path, output_path, typology):
    folder_name = os.path.basename(input_path)
    workbook = xlsxwriter.Workbook(os.path.join(output_path, folder_name+'.xlsx'))
    worksheet = workbook.add_worksheet()
    headers = ['image', 'id', 'area[um2]', 'perimeter[um]', 'avg_radius[um]', 'roundness[%]',
               'circularity[%]', 'maxFeret[um]', 'minFeret[um]', 'transparency[%]',
               'Mean curvature[um-1]', 'Std curvature[um-1]', 'Std curvature x R0', 'DNE']
    for i, h in enumerate(headers):
        worksheet.write(0, i, h)
    row = 1

    for image_name in [f for f in os.listdir(input_path) if f.endswith('.jpg')]:
        print(f'# Analyzing image {image_name}')
        image = cv2.imread(os.path.join(input_path, image_name), cv2.IMREAD_GRAYSCALE)
        image = image.astype('uint8')
        scale = set_scale(typology, image_name)

        organoid_ids = []
        iteration = 0
        while len(organoid_ids) == 0 and iteration < 3:
            mask = segment_image(image)
            organoid_ids = list(np.unique(mask))
            organoid_ids.remove(0)
            iteration += 1

        result_image = np.dstack([image]*3)
        new_id = 1
        for organoid_id in organoid_ids:
            cell_area = np.count_nonzero(mask == organoid_id)
            if cell_area > 5000:
                org_mask = (mask == organoid_id).astype(np.uint8)
                masked_image = cv2.bitwise_and(image, image, mask=org_mask)
                contours, _ = cv2.findContours(org_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                coords, result_image = smooth_contours(contours, result_image)
                ((x, y), radius) = cv2.minEnclosingCircle(contours[0])
                result_image = cv2.putText(result_image, str(new_id), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 6)
                result_image, row = save_output_values(cell_area, coords, scale, radius, org_mask, masked_image, image_name, new_id, row, worksheet, result_image)
                new_id += 1

        Image.fromarray(result_image).save(os.path.join(output_path, image_name))
    workbook.close()


# ---------------------------------------------------------------------------- #
# Script Entry Point
# ---------------------------------------------------------------------------- #


if __name__ == '__main__':
    # Define base directories (update these paths as needed)
    input_path = '/path/to/project'
    output_path = '/path/to/results'
    # Typology of your images ('2x' or '4x')
    typology = '2x'

    # Ensure output directories exist
    os.makedirs(output_path, exist_ok=True)

    # Run cell analysis
    main(input_path, output_path, typology)