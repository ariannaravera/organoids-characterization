import os
import cv2
import csv
from tqdm import tqdm
import tifffile
import numpy as np
from scipy.ndimage import center_of_mass
from skimage.measure import label, regionprops


def sholl_analysis(mask, mask_cs3, radius_step_micron, output_path, image_name):

    avg_area = 220

    # Create a binary version just to compute center of mass
    binary_mask = (mask > 0).astype(np.uint8)
    # Compute the global center of mass
    cy, cx = center_of_mass(binary_mask)
    # Create a distance map from center of mass
    yy, xx = np.indices(mask.shape)
    dist_map = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius_step_px = 2.2 * radius_step_micron
    # Determine maximum radius
    max_radius = mask.shape[0] / radius_step_px
    max_radius = int(max_radius / 2 * radius_step_px)
    # Create rings and count unique cell labels inside each
    radii = np.arange(0, max_radius, radius_step_px)
    
    for i in range(len(radii) - 1):
        r_in, r_out = radii[i], radii[i + 1]
        # Binary mask for this ring
        ring_mask = np.logical_and(dist_map >= r_in, dist_map < r_out)
        # Apply ring mask to the labeled mask
        masked = cv2.bitwise_and(mask, mask, mask=ring_mask.astype('uint8'))
        masked_cs3 = cv2.bitwise_and(mask_cs3, mask_cs3, mask=ring_mask.astype('uint8'))
        #masked_pax6 = cv2.bitwise_and(mask_pax6, mask_pax6, mask=ring_mask.astype('uint8'))
        """# Get unique labels excluding background (0)
        unique_cells = [x for x in np.unique(masked) if x != 0]
        # Compute cell area only inside the ring (not total area of the cell)
        areas = []
        for label_id in unique_cells:
            area = np.sum(masked == label_id)
            areas.append(area)
        avg_area = np.mean(areas) if areas else 0"""
        
        # Count cs3 cells based on their area
        """
        area < 20 -> delete
        20 < area < 0.7*avg_area -> 1
        area > 0.7*avg_area -> x based on x = area/(0.7*avg_area) -> approssima eccesso
        """
        if avg_area > 0:
            unique_labels_cs3 = [x for x in np.unique(masked_cs3) if x != 0]
            ncs3 = 0
            min_th = 20
            max_th = 0.7 * avg_area
            for label_id in unique_labels_cs3:
                area = np.sum(masked_cs3 == label_id)
                if area < min_th: continue
                elif area < max_th: ncs3 += 1
                else:
                    x = area / max_th
                    ncs3 += round(x)
        else:
            ncs3 = 0

        # total_area_pax6 = np.sum(masked_pax6 > 0)
        # npax6 = total_area_pax6 / avg_area if avg_area > 0 else 0

        # with open(os.path.join(output_path,  'results_caspase.csv'), "a") as file:
        #     writer = csv.writer(file)
        #     writer.writerow([image_name.replace('.tif', ''), i, np.sum(ring_mask > 0), round(np.sum(masked!=0)/avg_area), ncs3])#, npax6
        
    return radii, cy, cx
      

def cell_analysis_estimation(input_path, masks_path, output_path):
    
    # with open(os.path.join(output_path,  'results_caspase.csv'), "w") as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['image', 'sholl ID', 'area sholl', '#cells', '#caspase3'])#, '#PAX6'
    
    for image_name in tqdm([x for x in os.listdir(input_path) if '.tif' in x]):
        if not os.path.exists(os.path.join(output_path, 'sholl_new', image_name)):
            # Open the image and the mask
            image = tifffile.imread(os.path.join(input_path, image_name))
            _, masks = cv2.threshold(cv2.GaussianBlur(image[0], (9, 9), 0), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            #tifffile.imread(os.path.join(masks_path, image_name.replace('.tif','-maskSAM.tif')))
            # Set the sholl radius in microns
            radius_step_micron = 50

            # Segment the other 2 channels
            _, mask_cs3 = cv2.threshold(cv2.GaussianBlur(image[1], (9, 9), 0), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            mask_cs3 = label(mask_cs3)
            #_, mask_pax6 = cv2.threshold(cv2.GaussianBlur(image[2], (9, 9), 0), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            # Compute sholl
            radii, cy, cx = sholl_analysis(masks, mask_cs3, radius_step_micron, output_path, image_name)

            # Make sholl plots
            final_image = np.zeros((image.shape[0], image.shape[1], image.shape[2], 3), dtype=np.uint8)
            sholl_image = np.zeros((image.shape[1], image.shape[2]), dtype=np.uint8)
            
            for i, ch in enumerate(image):
                final_image[i, :, :, 0] = ch
                final_image[i, :, :, 1] = ch
                final_image[i, :, :, 2] = ch
                if i in [0,1]:
                    mask = masks if i == 0 else mask_cs3
                    contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(final_image[i], contours, -1, (255, 255, 0), 1)
                # Draw Sholl circles
                for r in radii:
                    cv2.circle(final_image[i], (int(cx), int(cy)), int(r), (255, 0, 0), 1)  # red circle
                    cv2.circle(sholl_image, (int(cx), int(cy)), int(r), 255, 3)  # red circle
                    
                cv2.circle(final_image[i], (int(cx), int(cy)), 2, (0, 255, 0), -1)  # green dot for center
                cv2.circle(sholl_image, (int(cx), int(cy)), 2, 200, -1)  # red circle
                
            
            tifffile.imwrite(os.path.join(output_path, 'sholl_new', 'shollcircles-'+image_name), sholl_image.astype('uint8'), imagej=True)
            # os.makedirs(os.path.join(output_path, 'sholl_new'), exist_ok=True)
            # tifffile.imwrite(os.path.join(output_path, 'sholl_new', image_name), final_image.astype('uint8'), photometric='rgb', metadata={'axes': 'CYXS'}, imagej=True)


if __name__ == '__main__':
    #px_to_mm = 11.077456

    ###############################################################
    # TO DO: set your path
    ###############################################################
    main_dir = '/Users/aravera/Documents/PROJECTS/DNF_Bagni/Giuseppe_s55/'
    original_data_path = os.path.join(main_dir, 'data_caspase')
    tif_data_path = os.path.join(main_dir, 'data_caspase/tif_version/todo')
    output_analysis_path = os.path.join(main_dir, 'results_caspase/')
    masks_path = os.path.join(main_dir, 'results_caspase/cells_mask')
    ###############################################################
    
    # Create output directory if does not exist
    os.makedirs(tif_data_path, exist_ok=True)
    os.makedirs(output_analysis_path, exist_ok=True)
    os.makedirs(masks_path, exist_ok=True)

    ###############################################################
    # TO DO: comment the steps you don't want to run, if any (comment adding an "#" at the beginneng of the line)
    ###############################################################

    # for folder1 in ['OS', 'RC']:
    #     for folder2 in [x for x in os.listdir(os.path.join(original_data_path, folder1)) if x != '.DS_Store']:
    #         input_path = os.path.join(original_data_path, folder1, folder2)
    #         name = folder1+'-'+folder2
    #         

    #         # STEP 1. Convert your lif file into tif images
    #         convert_lif_tif(input_path, tif_data_path, name)

    cell_analysis_estimation(tif_data_path, masks_path, output_analysis_path)


    ###############################################################