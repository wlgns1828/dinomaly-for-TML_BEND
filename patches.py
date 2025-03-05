import torch
import cv2
import numpy as np
import time


# TODO 1: Import your model here. Using YOLO as example
from colorama import Fore, Style
import pathlib
import os
# TODO 2:Import any other packages required by your code


class MODEL:
    """
    A class that encapsulates the model loading, warm-up, and inference operations.
    This approach ensures the model is loaded into memory once and used for multiple inferences.
    """
    def __init__(self, template_path='', method="TM_CCOEFF_NORMED", warmup_runs=1, device="cuda:0"):  #TODO 3: Update the model_path, Add the parameters you need which can be modified from GUI
        """
        Initialize the model.

        Args:
            template_path (str): Path of the template.
            method: method of template matching.
            warmup_runs (int): Number of dummy inference runs to warm up the model.
        """

        self.template = cv2.imread(template_path,0)
        self.method = getattr(cv2, method)
        dummy_image = cv2.cvtColor(np.zeros((640, 640,3), dtype=np.uint8),    cv2.COLOR_BGR2GRAY)
        for _ in range(warmup_runs):
            _ = cv2.matchTemplate(dummy_image, self.template, self.method)




    
    def pre_process(self, image: np.ndarray, crop_coord=[600,400,400,0]):
        """
        Pre-process the input image before running inference.

        Args:
            image (np.ndarray): The input image array.

        Returns:
            image (np.ndarray): The pre-processed image array.
        """
    
        
        if len(image.shape)==3:h,w,_ = image.shape
        else: h,w = image.shape
        
        x1 = crop_coord[2]
        x2 = w - crop_coord[3]
        y1 = crop_coord[0]
        y2 = h - crop_coord[1]
        
        return image[y1:y2, x1:x2]
    
    
    def GET_ANGLE_NAME(self,image_name):
        """
        Helper function to extract angle name from 
        image file. 
        
        Input: Image Name
        Returns: Camera Angle
        """
        
        if "Top-pin-2nd_auto_0" in image_name:
            return "Top-pin-2nd_auto_0"
        elif "Top-pin-2nd_auto_1" in image_name:
            return "Top-pin-2nd_auto_1"
        elif "Top-pin_auto_0" in image_name:
            return "Top-pin_auto_0"
        elif "Top-pin_auto_1" in image_name:
            return "Top-pin_auto_1"
        elif "Bottom-pin_auto_1" in image_name:
            return "Bottom-pin_auto_1"
        elif "Bottom-pin_auto_2" in image_name:
            return "Bottom-pin_auto_2"
        else: return None
        
    
    def CLEAVE_v5(self, img, patch_size=(256,256), overlap=0.25):
        '''
        This function will slice the images into the given patch sizes
        Input:  Image, patch_size, overlap
        Return: Size of each Patch (y, x) --> (h,w )
        '''

        array_image = np.asarray(img)
        image_height =  array_image.shape[0]
        image_width  =  array_image.shape[1]
        patch_height =  patch_size[0]
        patch_width  =  patch_size[1]
        patches = []
        y_max = y_min = 0
        y_overlap = int(patch_size[0]*overlap)
        x_overlap = int(patch_size[1]*overlap)

        i = 0
        shifting_measures = {}
        overlap_measures = {}
        
        while y_max < image_height:
            x_min = x_max = 0
            y_max = y_min + patch_height
            while x_max < image_width:
                x_max = x_min + patch_width 
                if y_max > image_height or x_max > image_width:
                    xmax = min(image_width, x_max)
                    ymax = min(image_height, y_max)
                    xmin = max(0, xmax - patch_width)
                    ymin = max(0, ymax - patch_height)
                    patches.append(array_image[ymin:ymax, xmin:xmax])
                    shifting_measures[f'Patch_{i}'] = (xmin, ymin)
                    #overlap_measures[f'Patch_{i}'] = [x_min+x_overlap, y_min, xmax, ymax]
                    if y_max > image_height and x_max < image_width:
                        if xmin ==0:
                            overlap_measures[f'Patch_{i}'] = [xmin, y_min + y_overlap, xmax-x_overlap, ymax]
                        else:
                            overlap_measures[f'Patch_{i}'] = [x_min+x_overlap, y_min + y_overlap, xmax-x_overlap, ymax]
                    elif x_max > image_width and y_max < image_height:
                        if ymin == 0:
                            overlap_measures[f'Patch_{i}'] = [x_min+x_overlap, ymin, xmax, y_max-y_overlap]
                        else:
                            overlap_measures[f'Patch_{i}'] = [x_min+x_overlap, ymin+y_overlap, xmax, y_max-y_overlap]    
                    elif y_max > image_height and x_max > image_width:
                        overlap_measures[f'Patch_{i}'] = [x_min+x_overlap, y_min+y_overlap, xmax, ymax]

                    i+=1
                else:
                    patches.append(array_image[y_min:y_max, x_min:x_max])
                    shifting_measures[f'Patch_{i}'] = (x_min, y_min)
                    if y_min == 0:
                        if x_min ==0:
                            top_thres_x = 0
                            top_thres_y = 0
                            bot_thres_x = x_max-x_overlap
                            bot_thres_y = y_max-y_overlap
                            overlap_measures[f'Patch_{i}'] = [top_thres_x, top_thres_y, bot_thres_x, bot_thres_y]
                        else:
                            top_thres_x = x_min + x_overlap
                            top_thres_y = 0
                            bot_thres_x = x_max-x_overlap
                            bot_thres_y = y_max-y_overlap
                            overlap_measures[f'Patch_{i}'] = [top_thres_x, top_thres_y, bot_thres_x, bot_thres_y] 
                    else:
                        if x_min==0:
                            top_thres_x = 0
                            top_thres_y = y_min + y_overlap
                            bot_thres_x = x_max-x_overlap
                            bot_thres_y = y_max-y_overlap
                            overlap_measures[f'Patch_{i}'] = [top_thres_x, top_thres_y, bot_thres_x, bot_thres_y]  
                        else:
                            top_thres_x = x_min + x_overlap
                            top_thres_y = y_min + y_overlap
                            bot_thres_x = x_max-x_overlap
                            bot_thres_y = y_max-y_overlap
                            overlap_measures[f'Patch_{i}'] = [top_thres_x, top_thres_y, bot_thres_x, bot_thres_y]         
                    
                    #overlap_measures[f'Patch_{i}'] = [x_min, y_min, x_max, y_max]
                    i+=1
                x_min = x_max - x_overlap
            y_min = y_max - y_overlap
        
        return patches, shifting_measures, overlap_measures
    
    def infer(self, image: np.ndarray):
        """
        This function performs the template matching. 
        """
        start_time = time.time()
        result = cv2.matchTemplate(image, self.template, self.method)
        inference_time = time.time() - start_time
        return result, inference_time

        
    def get_Template_results(self, result):
        
        template_width, template_height = self.template.shape[::-1]
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        x1 = max_loc[0] 
        y1 = max_loc[1]
        x2 = max_loc[0] + template_width
        y2 = max_loc[1] + template_height
        
        return [x1, y1, x2, y2]
    
    
    def get_ROI(self, img, offset_list, xyxy_coordinates):
        """
        This is a helper function that converts template matching coordinates to full ROI 
        of your choice.
        """
        
        ROI_x1 = xyxy_coordinates[2] + offset_list[0] # bottom right x of the xyxy_coordinates 
        ROI_y1 = xyxy_coordinates[1] + offset_list[1] # bottom right y of template
        ROI_x2 = xyxy_coordinates[2] + offset_list[2]
        ROI_y2 = xyxy_coordinates[3] + offset_list[3]
        
        ROI_xyxy  = [ROI_x1, ROI_y1, ROI_x2, ROI_y2]
        ROI_image = img[ROI_y1:ROI_y2, ROI_x1:ROI_x2] 
        
        return ROI_image, ROI_xyxy
    

    def main(self, image: np.ndarray, previous_detections = [], info='', patch_mode = "", binarize=False):  #TODO 7: You will receive image and the detection results from previous model (if this model is not the first one)
        """
        Run inference on the provided image using the pre-loaded YOLO model.

        Args:
            image (np.ndarray): The input image array.
            previous_detections (list): A list of detection results from the previous model.

        Returns:
            annotated_image (np.ndarray): The image annotated with detection results.
            detections (list): A list of detection results, each containing bounding boxes,
                                confidence scores, classes, and labels.
        """

        # Check if the input image is valid
        if image is None:
            raise ValueError("Input image is empty or None.")
        
        angle_name = self.GET_ANGLE_NAME(info)
        
        Threshold   = 200
        
        _img = np.copy(image)
        _img = cv2.cvtColor(_img,    cv2.COLOR_BGR2GRAY)
        
        if angle_name=="Bottom-pin_auto_1": 
            crop   = [600, 400, 0, 900] ## [top, bottom, left, right]
            offset = [-32, 0, 1223, 7]  ## [x1, y1, x2, y2]
        
        elif angle_name=="Bottom-pin_auto_2":
            crop   = [600, 400, 400, 0]  ## [top, bottom, left, right]
            offset = [-32, 0, 1198, 7]   ## [x1, y1, x2, y2]
        
        # Pre-process the input image
        _image = self.pre_process(_img, crop_coord=crop)

        # Run inference
        tic = time.time()
        # Post-process the detection results
        detection_results, inference_time = self.infer(_image) #Infer 함수는 템플릿 매칭을 수행합니다.
        xyxy_coord                        = self.get_Template_results(detection_results) ## get_Template_results 함수는 ROI 추출을 위한 크롭 좌표를 반환합니다.
        extracted_roi_img, roi_xyxy       = self.get_ROI(_image, offset_list=offset, xyxy_coordinates=xyxy_coord) ## get_ROI 함수는 템플릿 매칭 결과에 따라 원본 이미지를 크롭합니다.
        
        if binarize: _, extracted_roi_img = cv2.threshold(extracted_roi_img, 160, 255, cv2.THRESH_BINARY)
        
        if patch_mode=="Row_wise":      patches, _, _ = self.CLEAVE_v5(extracted_roi_img, patch_size=(extracted_roi_img.shape[0], 115)) ## 이 함수는 이미지를 행 단위로 잘라서 분할합니다.
        elif patch_mode=="Column_wise" :patches, _, _ = self.CLEAVE_v5(extracted_roi_img, patch_size=(122, extracted_roi_img.shape[1])) ## 이 함수는 이미지를 열 단위로 잘라서 분할합니다.
        else: raise "Please Check your Patching Mode"

        toc = time.time()
        print(f"Time Take: {round((toc - tic),4)} s")
        return extracted_roi_img, patches



# Example usage:
if __name__ == "__main__":
        
    def GET_FILES_LIST(Path, dot_type, limit = None):
        ''' 
        This function returns a list of files with specific extension. 
        Input:   Path, type of file to find, and number of files to find
        Returns: List of files
        
        '''
        filelist = []
        break_outer_loop = False
        
        # Getting the list of all the annotations in the given folder

        for path, dirs, files in os.walk(Path):
            for file in files:
                if file.endswith(dot_type):
                    if limit is not None:
                        limit-=1
                        if limit>=0:filelist.append(os.path.join(path, file))
                        else:
                            break_outer_loop = True
                            break
                    else:filelist.append(os.path.join(path, file))
            if break_outer_loop: break       
        return filelist
        
    source_directory = r"/home/ohjihoon/바탕화면/dino/03_03_Anomaly_dataset/Bottom-pin_auto_2"
    assert os.path.exists(source_directory), "Check your Source Directory"
     
    model  = MODEL(template_path=r"template_4.png")
    images = GET_FILES_LIST(Path=source_directory, dot_type=".png")

    save_patches = True
    save_dir = "/home/ohjihoon/바탕화면/dino/03_03_Anomaly_dataset/patch/Bottom-pin_auto_2"
    os.makedirs(save_dir, exist_ok=True)
    
    #patching_mode = "Row_wise"
    patching_mode = "Column_wise"
    
    ## 
    global_time = 0
     
    for idx, image in enumerate(images):
        tic = time.time()
        img_name = pathlib.PurePath(image.split('.png')[0]).parts[-1]
        print(img_name) 
        image = cv2.imread(image)

    # Run inference
        roi, patches = model.main(image, info = img_name, patch_mode=patching_mode, binarize=True)
        toc = time.time()
        global_time+=(toc-tic)
        
        if save_patches:
            for idx, patch in enumerate(patches):
                
                ####### First Method to Save Patched Images #######
                #patches_path = os.path.join(save_dir, img_name)
                #if not os.path.exists(patches_path):
                #    os.makedirs(patches_path)
                #cv2.imwrite(os.path.join(patches_path, f"patch_{idx}.png"),patch)

                ####### Another method to save patched images #######
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(os.path.join(save_dir, f"{img_name}_patch_{idx}.png"),patch)

        else: 
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                
            cv2.imwrite(os.path.join(save_dir, f"{img_name}.png"),roi)
   

    print(f"Average time taken: {round((global_time/len(images)),4)}s")