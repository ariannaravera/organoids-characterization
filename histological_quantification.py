import os
import cv2
import csv
import napari
import tifffile
import numpy as np
from skimage.measure import label
from skimage.draw import polygon2mask
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QWidget, QMessageBox, QLineEdit, QLabel


def cellsizewindow(image):
    viewer = napari.Viewer()
    viewer.title = 'Set cell size'
    viewer.add_image(image, name='image')
    shape_layer = viewer.add_shapes(name='cells size')
    napari.run()

    
    # Initialize ROI mask and area list
    mask = np.zeros(image.shape, dtype=np.uint8)
    areas = []

    for shape in shape_layer.data:
        shape = np.array(shape)
        if shape.ndim == 3:
            shape = shape[0]  # unwrap if needed

        poly_mask = polygon2mask(image.shape, shape)
        area = poly_mask.sum()
        areas.append(area)
        mask |= poly_mask

    average_area = np.mean(areas) if areas else 0

    return int(average_area)


class SaveButton(QWidget):
    def __init__(self, viewer: napari.Viewer, mask, output_path):
        super().__init__()
        self.viewer = viewer

        label = QLabel("Insert the threshold value:")
        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText("Enter number")

        self.button = QPushButton('Save layer')
        self.button.clicked.connect(self.save_layer)

        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.button)
        self.setLayout(layout)

        self.output_path = output_path
        self.newmask = mask
        self.th_value = None
    
    def save_layer(self):
        try:
            value = float(self.line_edit.text())
            image = self.viewer.layers['image'].data
            _, self.newmask = cv2.threshold(image, int(value), 255, cv2.THRESH_BINARY)
            self.viewer.add_labels(self.newmask, name='mask')
            msg = QMessageBox() 
            msg.setIcon(QMessageBox.Information)
            msg.setText("Saved!")
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            msg.exec_()
        except ValueError:
            print("Please enter a valid number.")
           

def manualcheck_mask(img, name, output_path):
    # Manually modify ChAT mask with napari
    viewer = napari.Viewer()
    viewer.title = 'Segment '+name+' channel'
    viewer.add_image(img, name='image')
    _, mask = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)
    viewer.add_labels(mask, name=name)
    save_button = SaveButton(viewer, mask, output_path)
    viewer.window.add_dock_widget(save_button, name='Save', area='left')
    viewer.show(block=True)
    return save_button.newmask


def ROIwindow(image, masks):
    names = ['Ki67', 'NeuroD2', 'PAX6']
    viewer = napari.Viewer()
    viewer.add_image(image, name='image')
    for i, m in enumerate(masks):
        if len(m.shape) > 2:
            viewer.add_labels(m[:,:,0], name=names[i], visible=False)
        else:
            viewer.add_labels(m, name=names[i], visible=False)
    shape_layer = viewer.add_shapes(name='ROI')
    napari.run()

    
    mask = np.zeros(image.shape, dtype=np.uint8)
    for shape in shape_layer.data:
        shape = np.array(shape)
        if shape.ndim == 3:
            shape = shape[0]  # unwrap if needed
        poly_mask = polygon2mask(image.shape, shape)
        poly_mask[poly_mask != 0] = 255
        mask |= poly_mask

    return mask.astype(np.uint8)


def analysis(input_path, output_path, ROI_outpath):
    if not os.path.exists(os.path.join(output_path,  'results-ROI.csv')):
        with open(os.path.join(output_path,  'results-ROI.csv'), "w") as file:
            writer = csv.writer(file)
            writer.writerow(['image', 'ROI id', '#cells', '#Ki67', '#NeuroD2', '#PAX6', '#Ki67&PAX6'])

    # Loop over .tif files in the input directory
    for image_name in [x for x in os.listdir(input_path) if '.tif' in x]:
        print('Analysing image '+image_name)
        print()

        # Load the image as an 8-bit unsigned integer array
        image = tifffile.imread(os.path.join(input_path, image_name))
        
        # Define the path to save the segmentation mask
        output_mask_path = os.path.join(ROI_outpath, image_name.replace('.tif', '-ROI.tif'))
        if not os.path.exists(output_mask_path):
            """
            Estimate cell size
            """
            estimated_area = cellsizewindow(image[0])
            print('Estimated cells size '+str(estimated_area))

            """
            Make channels mask
            """
            channels_masks = np.zeros_like(image, dtype=np.uint8)
            names = ['DAPI', 'Ki67', 'NeuroD2', 'PAX6']
            for i, ch in enumerate(image):
                if i == 0:
                    ch = cv2.convertScaleAbs(ch, alpha=5, beta=0)
                    blur = cv2.GaussianBlur(ch, (9, 9), 0)
                    _, channels_masks[i] = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                else:
                    blur = cv2.GaussianBlur(ch, (9, 9), 0)
                    channels_masks[i] = manualcheck_mask(blur,  names[i], output_path)
            
            os.makedirs(os.path.join(output_path, 'masks_estimation'), exist_ok=True)
            tifffile.imwrite(os.path.join(output_path, 'masks_estimation', 'masks-'+image_name), channels_masks.astype('uint8'), metadata={'axes': 'CYX'}, imagej=True)

            """
            Make ROIs
            """
            ROIs_masks = ROIwindow(image[0], channels_masks[1:])
            # Save the segmentation mask as a .tif file
            ROIs_masks[ROIs_masks!=0] = 255
            
            """
            Estimation anslysis
            """
            results =[image_name.replace('.tif','')]
            print('Analysis...')
            labeledROIs_masks = label(ROIs_masks)
            for ROI_id in [x for x in np.unique(labeledROIs_masks) if x!=0]:
                ROI_mask = np.zeros(labeledROIs_masks.shape, dtype=np.uint8)
                ROI_mask[labeledROIs_masks == ROI_id] = 255
                # Compute centroid to place the text
                coords = np.argwhere(labeledROIs_masks == ROI_id)
                if coords.size == 0:
                    continue  # skip empty region

                centroid_yx = coords.mean(axis=0)
                position = (int(centroid_yx[1]), int(centroid_yx[0]))

                # Put text (ROI ID) on the ROI_mask
                cv2.putText(ROIs_masks, str(ROI_id), position, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=4, color=100, thickness=3, lineType=cv2.LINE_AA)

                results =[image_name.replace('.tif',''), ROI_id]
                
                for i, _ in enumerate(image):
                    masked_mask = cv2.bitwise_and(channels_masks[i], channels_masks[i], mask=ROI_mask.astype('uint8'))
                    tot_area = np.count_nonzero(masked_mask)
                    n_cells = int(tot_area / estimated_area)
                    results.append(n_cells)
                
                ki67mask = cv2.bitwise_and(channels_masks[1], channels_masks[1], mask=ROI_mask.astype('uint8'))
                pax6mask = cv2.bitwise_and(channels_masks[3], channels_masks[3], mask=ROI_mask.astype('uint8'))

                maskcoloc = (ki67mask & pax6mask).astype(np.uint8)
                tot_area = np.count_nonzero(maskcoloc)
                n_cells_coloc = int(tot_area / estimated_area)
                results.append(n_cells_coloc)

                with open(os.path.join(output_path,  'results-ROI.csv'), "a") as file:
                    writer = csv.writer(file)
                    writer.writerow(results)
                
                tifffile.imwrite(output_mask_path, ROIs_masks.astype(np.uint8))


if __name__ == '__main__':
    
    for type in ['RC', 'OS']:
        main_dir = '/Users/aravera/Documents/PROJECTS/DNF_Bagni/Giuseppe_s55/'
        tif_data_path = os.path.join(main_dir, 'data/tif_version-'+type)
        output_analysis_path = os.path.join(main_dir, 'results/'+type)
        ROI_path = os.path.join(main_dir, 'results/'+type+'/ROIs')
        ###############################################################
        
        # Create output directory if does not exist
        os.makedirs(output_analysis_path, exist_ok=True)
        os.makedirs(ROI_path, exist_ok=True)

        analysis(tif_data_path, output_analysis_path, ROI_path)
        
        ###############################################################