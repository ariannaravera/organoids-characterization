#!/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2024, Arianna Ravera <Arianna.Ravera@unil.ch> DCSR, UNIL.
# All rights reserved.

import os
import cv2
import math
import feret
import xlsxwriter
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import measure


### PARAMETERS TO CHANGE ###

# Set the folder path where the jpg images are, eg. /Users/my_name/Documents/my_data_folder
input_path = '/Users/my_name/Documents/my_data_folder'

# Set the folder path where to same results, eg. /Users/my_name/Documents/my_results_folder
output_path = '/Users/my_name/Documents/my_results_folder'

# Set the type of your data to define the convertion scale, it can be '2x' or '4x'
typology = '2x'

################################################################

def curv_helper(x0, y0, x1, y1, x2, y2):
    """
    Use Heron's formula to compute curvature
    as the radius of the osculating circle using a geometric approach.
    """
    xa = x0 - x1
    ya = y0 - y1
    
    xb = x2 - x1
    yb = y2 - y1
    
    xc = x2 - x0
    yc = y2 - y0
    
    a = np.sqrt(xa * xa + ya * ya)
    b = np.sqrt(xb * xb + yb * yb)
    c = np.sqrt(xc * xc + yc * yc)
    
    Delta = (xb * ya - xa * yb) / 2.0
    Kappa = 4.0 * Delta / (a * b * c)
    
    return Kappa

def get_contour_curvature_geo(coords):
    """
    Compute curvature with periodic boundary conditions using a geometric method.
    """
    kappa = np.zeros(len(coords))
    
    for i in range(len(coords)):
        ip = ((i - 1) + len(coords)) % len(coords)
        in_ = ((i + 1) + len(coords)) % len(coords)
        kappa[i] = curv_helper(coords[ip][0], coords[ip][1], coords[i][0], coords[i][1], coords[in_][0], coords[in_][1])
        
    return kappa

def get_contour_inflection_points(coord, res_image, curvature):
    """
    Return list of indices where the curvature is more than 2 times the minor axis.
    """
    # Smooth the curvature data
    tmp = ndimage.gaussian_filter1d(curvature, 2)
    
    # Set the threshold
    threshold = 0.05
    min_distance = 30
    
    # Find inflection points
    inflx = []
    for i in range(1, len(tmp) - 1):
        if tmp[i - 1] < tmp[i] > tmp[i + 1] and tmp[i] > threshold:
            if len(inflx) == 0 or i - inflx[-1] > min_distance:
                inflx.append(i)
                res_image = cv2.circle(res_image, coord[i], 15, (255,0,0), -1)
                res_image = cv2.circle(res_image, coord[i], 10, (255,255,0), -1)
    
    return len(inflx), res_image

########################################################################

def LoG_filter_opencv(image, sigma, size=None):
    """
    Calculate LoG filter of the image

    Input:
        - image: 2D numpy array
    Output:
        - filtered_image: 2D numpy array
    """
    # Generate LoG kernel
    if size is None:
        size = int(6 * sigma + 1) if sigma >= 1 else 7
    if size % 2 == 0:
        size += 1

    x, y = np.meshgrid(np.arange(-size//2+1, size//2+1), np.arange(-size//2+1, size//2+1))
    kernel = -(1/(np.pi * sigma**4)) * (1 - ((x**2 + y**2) / (2 * sigma**2))) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / np.sum(np.abs(kernel))

    # Perform convolution using OpenCV filter2D
    filtered_image = cv2.filter2D(image, -1, kernel)

    return filtered_image

########################################################################

def compute_dne(contours_coords):
    """
    Dirichlet Normal Energy (DNE):
    logarithm of the square of the variation of the normal n=(dy,-dx) of the contour projected on its tangent t=(dx,dy)
    where dx and dy are the first derivative in x and y
    
    Input:
        - contour (np.array): A 2D numpy array of shape (N, 2) representing the x and y coordinates of the contour.

    Output:
        - log_projection_squared: float DNE value.
    """
    dy = np.gradient([i[0] for i in contours_coords])
    dx = np.gradient([i[1] for i in contours_coords])
    
    # Compute the normal vectors n = (dy, -dx)
    normals = np.column_stack((dy, -dx))
    normals_norm = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    
    # Compute the tangent vectors t = (dx, dy)
    tangents = np.column_stack((dx, dy))
    tangents_norm = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)
    
    # Compute the gradient of the normals
    normal_gradient_dx = np.gradient(normals_norm[:, 0])
    normal_gradient_dy = np.gradient(normals_norm[:, 1])
    
    # Project the gradient of the normal onto the tangent
    projections = normal_gradient_dx * tangents_norm[:, 0] + normal_gradient_dy * tangents_norm[:, 1]
    
    # Compute the square of the variation
    variation_squared = np.sum([x for x in np.power(projections, 2) if not math.isnan(x)])

    # Compute the logarithm of the square of the projection
    dne = np.log(variation_squared + np.finfo(float).eps)

    return dne

def set_scale(typology, image_name):
    """
    Set the conversion scale based on the type of the image that can be 2x or 4x.
    """
    if typology == '2x':
        scale = 1230/478
    elif typology == '4x':
        scale = 670/480
    else:
        if '2x' in image_name:
            scale = 1230/478
        elif '4x' in image_name:
            scale = 670/480
        else:
            print("!ERROR! : wrong typology, expected '2x' or '4x'")
            quit()
    return scale

def segment_image(image):
    """
    Segment the image generating its binary mask
    """
    # Create the mask of the image
    val = np.average(image)+np.std(image)/2
    median = cv2.medianBlur(image,5)

    ## If you want to change the threshold parameter ##
    # 1. print the val variable to see what's the used one
    # 2. change it based on the image content, and use this
    #_, mask = cv2.threshold(median, YOUR VALUE HERE, 255, cv2.THRESH_BINARY)
    # instead of:
    _, mask = cv2.threshold(median, int(val), 255, cv2.THRESH_BINARY)
    
    mask = np.invert(mask)

    # Filling holes
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel, iterations=5)
    mask = ndimage.binary_fill_holes(mask).astype(int)

    mask = measure.label(mask, background=0)
    mask[mask == mask[0][0]] = 0
    mask[mask == mask[mask.shape[0]-1][mask.shape[1]-1]] = 0
    mask = mask.astype('uint16')

    return mask

def smooth_contours(contours, result_image):
    """
    Smooth the contours of the image applying a gaussian filter.
    """
    sig = 7
    coords = []
    for cnt in contours:
        originalX = np.array(cnt[:, 0, 1])
        originalY = np.array(cnt[:, 0, 0])
        
        #Smooth points
        xscipy = ndimage.gaussian_filter(originalX, sig, mode='wrap')
        yscipy = ndimage.gaussian_filter(originalY, sig, mode='wrap')
        for i in range(len(xscipy)):
            coords.append((yscipy[i], xscipy[i]))
        cnt[:, 0, 0] = yscipy
        cnt[:, 0, 1] = xscipy
        cv2.drawContours(result_image, cnt, -1, (255,0,0), 3)
    return coords, result_image

def save_output_values(cell_area, coords, scale, radius, org_mask, masked_image, image_name, new_id, row, worksheet, result_image):
    """
    Save results in xlsx file
    """
    # Area in micrometers
    area = cell_area * scale * scale
    # Perimeter in micrometers
    perimeter = 0
    for i in range(len(coords)-1):
        perimeter += math.sqrt((coords[i][0] - coords[(i + 1)][0]) ** 2 + (coords[i][1] - coords[(i + 1)][1]) ** 2)
    # Average radius in micrometers
    avg_radius = radius * scale
    # Roundness = 4 x [Area/π(Major axis)^2]
    roundness = 4 * (cell_area / (np.pi * np.power((radius*2), 2)))
    #Circularity = (4π x Area)/Perimeter^2
    circularity = (4 * np.pi * cell_area)/np.power(perimeter, 2)
    # Feret values
    maxf, minf, minf90, maxf90 = feret.all(org_mask)
    # Curvature
    curvature = get_contour_curvature_geo(coords)
    # Inflection points
    #infl_points, result_image = get_contour_inflection_points(coords, result_image, curvature)
    # Trasparency
    transparency = np.count_nonzero(LoG_filter_opencv(masked_image, 1))/np.count_nonzero(masked_image)
    # Mean curvature in micrometers
    mean_curvature = np.average([x for x in curvature if not math.isnan(x)]) / scale
    # Standard deviation of curvature in micrometers
    std_curvature = np.std([x for x in curvature if not math.isnan(x)]) / scale
    # Standard deviation of curvature normalized with average radius
    std_curvature_norm = std_curvature * avg_radius
    # DNE
    dne = compute_dne(coords)
    # Write outputs in the final xlsx
    for i, el in enumerate([image_name, new_id, area, perimeter * scale, avg_radius, roundness*100, 100 - (abs(1-circularity))*100, maxf * scale, minf * scale, transparency*100, mean_curvature, std_curvature, std_curvature_norm, dne]):
        worksheet.write(row, i, el)
    row += 1

    return result_image, row

########################################################################

def main():
    folder_name = os.path.basename(input_path)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    # Initialize the xlsx result file
    workbook = xlsxwriter.Workbook(os.path.join(output_path, folder_name+'.xlsx'))
    worksheet = workbook.add_worksheet()
    for i, el in enumerate(['image', 'id', 'area[um2]', 'perimeter[um]', 'avg_radius[um]', 'roundness[%]', 'circularity[%]', 'maxFeret[um]', 'minFeret[um]', 'transparency[%]', 'Mean curvature[um-1]', 'Std curvature[um-1]', 'Std curvature x R0', 'DNE']):
        worksheet.write(0, i, el)
    row = 1

    # Iterate over all the jpg image in the input_path folder
    for image_name in [f for f in os.listdir(input_path) if '.jpg' in f]:
        print('# Analyzing image '+image_name)
        # Read the image
        image = cv2.imread(os.path.join(input_path, image_name), cv2.IMREAD_GRAYSCALE).astype('uint8')
        # Set the scale based on the typology chosen
        scale = set_scale(typology, image_name)
        
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        organoid_ids = []
        iteration = 0
        while len(organoid_ids) == 0 and iteration < 3:
            # Filter the mask and assign one value to each area found
            mask = segment_image(image)
            # Get the masks unique values
            organoid_ids = list(np.unique(mask))
            organoid_ids.remove(0)
            iteration += 1
        
        # Set organoid id starting from 1
        new_id = 1
        result_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        result_image = np.dstack([image, image, image])
        
        # Iterate over organoids found
        for organoid_id in organoid_ids:
            # Calculate organoid area
            cell_area = np.count_nonzero(mask == organoid_id)
            # Filter organoids based on their area
            if cell_area > 5000:
                # Create the mask of the single organoid
                org_mask = np.zeros(mask.shape, dtype=np.uint8)
                org_mask[mask == organoid_id] = 1
                # Masked the original image in this area
                masked_image = cv2.bitwise_and(image, image, mask=org_mask)
                # Get contours of the organoid mask
                contours, _ = cv2.findContours(org_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # Smooth the contours
                coords, result_image = smooth_contours(contours, result_image)
                
                # Find minum enclosing circle around this area
                ((x, y), radius) = cv2.minEnclosingCircle(contours[0])

                # Write organoid id on the result image
                result_image = cv2.putText(result_image, '{}'.format(new_id), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 6)
                
                ## Calculate output values
                result_image, row = save_output_values(cell_area, coords, scale, radius, org_mask, masked_image, image_name, new_id, row, worksheet, result_image)
                
                # Increment the organoid id
                new_id += 1
        
        # Save the output image with organoid ids drawn
        Image.fromarray(result_image).save(os.path.join(output_path, image_name))
        
    workbook.close()
        

if __name__ == '__main__':
    main()
