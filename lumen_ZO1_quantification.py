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


def convert_lif_tif(input_path, output_path, name):
    with open(os.path.join(output_path,  'conversions.csv'), "a") as file:
        writer = csv.writer(file)
        writer.writerow(['image', 'value (px/value = micron)'])
                
    # Loop over .lif files in the input directory (excluding lifext and xml files)
    for image_name in [x for x in os.listdir(input_path) if '.lif' in x and 'lifext' not in x and 'xml' not in x]:
        save_name = name+'-'+image_name
        print('Converting image '+save_name)
        print()
        # Load the .lif file
        lif_file = LifFile(os.path.join(input_path, image_name))
        # Loop over all images in the .lif file
        for i in range(lif_file.num_images):
            # Create an empty array to store the image data (z, y, x)
            lif_image = lif_file.get_image(i)
            if 'Merged' in lif_image.name:
                pixel_size = lif_image.info['scale']
                with open(os.path.join(output_path,  'conversions.csv'), "a") as file:
                    writer = csv.writer(file)
                    writer.writerow([save_name.replace(' ','').replace('_.lif','')+'-'+lif_image.name.replace(' ','').replace('/','').replace('TileScan','').replace('Merged','')+'.tif', pixel_size])
                
                # px/value = micron
                image = np.zeros((lif_image.info['channels'], lif_image.info['dims'].y, lif_image.info['dims'].x), dtype=np.uint8)
                # Extract the z-slice for the current frame and store in the array
                for c in range(lif_image.info['channels']):
                    image[c, :,:] = lif_image.get_frame(c=int(c))
                # Normalize the pixel intensity to the range [0, 255]
                normalized_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                # Save the normalized image as a .tif file (ImageJ-compatible)
                tifffile.imwrite(os.path.join(output_path, save_name.replace(' ','').replace('_.lif','')+'-'+lif_image.name.replace(' ','').replace('/','').replace('TileScan','').replace('Merged','')+'.tif'), normalized_image, metadata={'axes': 'CYX'}, imagej=True)

                
def clean_binary_mask(mask, min_size=100, hole_size=100, iterations=2):
    """
    Clean a binary mask by keeping stable core areas and removing noise.
    
    Parameters:
        mask (np.ndarray): Input binary mask (2D, dtype=bool or 0/1).
        min_size (int): Minimum area of objects to keep.
        hole_size (int): Maximum area of holes to fill.
        iterations (int): Number of dilation/erosion steps.
    
    Returns:
        np.ndarray: Cleaned binary mask.
    """
    # Ensure boolean mask
    mask = mask.astype(bool)
    # Morphological opening (erosion followed by dilation)
    eroded = binary_erosion(mask, structure=disk(1), iterations=iterations)
    dilated = binary_dilation(eroded, structure=disk(1), iterations=iterations)
    # Fill small holes in the remaining regions
    filled = remove_small_holes(dilated, area_threshold=hole_size)
    # Remove small objects
    cleaned = remove_small_objects(filled, min_size=min_size)

    return cleaned.astype(np.uint8)


def manualcorrection_mask(blur, pax6, mask):
    result = {"value": -1}  # valore iniziale che verrà aggiornato

    viewer = napari.Viewer()
    viewer.add_image(blur, name='Zo1')
    viewer.add_image(pax6, name='PAX6', visible=False)

    label_layer = viewer.add_labels(mask, name='mask')
    label_layer.mode = 'paint'
    label_layer.selected_label = 2

    # Istruzioni
    label = QLabel("🖌️ Paint and correct all the rosette with brush of value 2")
    label.setAlignment(Qt.AlignCenter)
    label.setStyleSheet("color: white; font-size: 14pt; background-color: rgba(0,0,0,150); padding: 6px; border-radius: 6px;")
    viewer.window.add_dock_widget(label, area='top', name='Instructions')

    # Input utente
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

    # Funzione che salva il valore prima di chiudere la finestra
    def on_close(event=None):
        result["value"] = spinbox.value()

    # Collegamento alla chiusura della finestra
    viewer.window._qt_window.closeEvent = on_close

    napari.run()

    updated_mask = label_layer.data
    corrected_mask = (updated_mask == 2).astype(np.uint8)

    return corrected_mask, result["value"]


def analyze_mask_and_region(dapi, mask, npositive, image_name, output_path):
    """
    Analyze the region around the corrected mask in another image.
    
    Parameters:
        mask (np.ndarray): Binary mask (1 where label == 2).
        other_channel (np.ndarray): Image to analyze in the masked region.
        dilation_radius (int): Radius of dilation to enlarge the region.
        
    Returns:
        dict: Dictionary with analysis results.
    """
    if not os.path.exists(os.path.join(output_path,  'results.csv')):
        with open(os.path.join(output_path,  'results.csv'), "w") as file:
            writer = csv.writer(file)
            writer.writerow(['image', '#rosette', '#positive', 'avg distance', 'avg area', 'distances', 'areas'])

    ch = cv2.convertScaleAbs(dapi, alpha=3, beta=0) # zo1 = image[1]
    blur = cv2.GaussianBlur(ch, (9, 9), 0)
    _, DAPI_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Compute location of the mask and compare to image center
    labeled_mask = label(mask)
    dapi_center = np.array(center_of_mass(DAPI_mask))  # returns (row, col)
    regions = regionprops(labeled_mask)
    # Compute distance from each region centroid to the DAPI center
    region_distances = []
    region_areas = []
    
    for region in regions:
        centroid = np.array(region.centroid)  # (row, col)
        region_distances.append(np.linalg.norm(centroid - dapi_center))
        region_areas.append(int(region.area))

    with open(os.path.join(output_path,  'results.csv'), "a") as file:
            writer = csv.writer(file)
            writer.writerow([image_name, len(regions), npositive, np.round(np.average(region_distances), 2), np.round(np.average(region_areas), 2), region_distances, region_areas])
    
    
def zo1_segmentation(input_path, output_path, masks_path):
    # Loop over .tif files in the input directory
    for image_name in [x for x in os.listdir(input_path) if '.tif' in x]:
        # Define the path to save the segmentation mask
        output_mask_path = os.path.join(masks_path, image_name.replace('.tif', '-mask.tif'))
        print('Segmenting image '+image_name)
        print()
        # Load the image as an 8-bit unsigned integer array
        image = tifffile.imread(os.path.join(input_path, image_name)).astype('uint8')
        if not os.path.exists(output_mask_path):
            ch = cv2.convertScaleAbs(image[1], alpha=3, beta=0) # zo1 = image[1]
            blur = cv2.GaussianBlur(ch, (9, 9), 0)
            # final_image[i, :, :, 0] = blur
            # final_image[i, :, :, 1] = blur
            # final_image[i, :, :, 2] = blur
            avg = blur[np.nonzero(blur)].mean()
            _, mask = cv2.threshold(blur, np.min([int(avg*6), 200]), 255, cv2.THRESH_BINARY)
            mask = clean_binary_mask(mask, min_size=20, hole_size=10, iterations=2)
            rosettes_mask, positive_rosette = manualcorrection_mask(blur, image[2], mask)
            # Save the segmentation mask as a .tif file
            tifffile.imwrite(output_mask_path, rosettes_mask)
            analyze_mask_and_region(image[1], rosettes_mask, positive_rosette, image_name, output_path)        


if __name__ == '__main__':
    ###############################################################
    # TO DO: set your path
    ###############################################################
    main_dir = '/Users/aravera/Documents/PROJECTS/DNF_Bagni/Giuseppe_s55/'
    original_data_path = os.path.join(main_dir, 'data_Zo1')
    tif_data_path = os.path.join(main_dir, 'data_Zo1/tif_version')
    masks_path = os.path.join(main_dir, 'results_Zo1/rosette_masks')
    output_analysis_path = os.path.join(main_dir, 'results_Zo1')
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

    #         # STEP 1. Convert your lif file into tif images
    #         convert_lif_tif(input_path, tif_data_path, name)

    # STEP 2. Segment and analyse zo1
    zo1_segmentation(tif_data_path, output_analysis_path, masks_path)
    
    ###############################################################