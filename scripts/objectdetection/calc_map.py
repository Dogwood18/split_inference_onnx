'''
 * License: 5G-MAG Public License (v1.0)
 * Author: Imed Bouazizi, Liangping Ma
 * Copyright (c) 2023 Qualcomm Inc.
 * Licensed under the License terms and conditions for use, reproduction,
 * and distribution of 5GMAG software (the “License”).
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at https://www.5g-mag.com/license .
 * Unless required by applicable law or agreed to in writing, software distributed under the License is
 * distributed on an “AS IS” BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
'''

import os
import numpy as np
import sys
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
import argparse
import glob

def load_imagenet_coco(filename):
    mappings = {}
    with open(filename, "rt") as f:
        l = f.readline()
        for s in f:
            tokens = s.split(',')
            mappings[tokens[0]] = tokens[3].rstrip()
    
    return mappings

def load_coco_labels(filename):
    labels = {}
    with open(filename, "rt") as f:
        for s in f:
            tokens = s.split(',')
            labels[int(tokens[0])] = tokens[1].rstrip()
    
    return labels

class Evaluator:
    """
    Class for calculating the mean average precision (mAP) image by image.

    Args:
        groundtruth_folder: The folder containing the groundtruth annotations.
        prediction_folder: The folder containing the prediction results.
        mappings: optional parameter if conversion of format from ImageNet to COCO is required
        labels: indices and names of the COCO labels
    """

    def __init__(self, groundtruth_folder, prediction_folder, threshold, mappings=None, labels=None):
        self.groundtruth_folder = groundtruth_folder
        self.prediction_folder = prediction_folder
        self.mappings = mappings
        self.labels = labels
        self.global_aps = {}
        self.global_gt_counter = {}
        self.global_tp_counter = {}
        self.results = None
        self.threshold = threshold

    def evaluate(self, is_imagenet=False):
        mAPs = {}
        gt_file_list = os.listdir(self.groundtruth_folder)
        gt_file_count  = len(gt_file_list)

        #for frame_num in trange(gt_file_count):
        for frame_num in range(gt_file_count):
            image_file = gt_file_list[frame_num]
            groundtruth_file = os.path.join(self.groundtruth_folder, image_file)
            prediction_file = os.path.join(self.prediction_folder, image_file)

            groundtruth_boxes = self.load_groundtruth(groundtruth_file)
            prediction_boxes = self.load_prediction(prediction_file, convert_to_coco=is_imagenet)

            self.calculate_map(groundtruth_boxes, prediction_boxes)
            #print("Result for {}  is mAP: {}".format(image_file,map_score))            
            
            #exit()
             
        for label, aps in self.global_aps.items():
            #mAPs.append((label,np.mean(aps)))
            mAPs[label] = np.mean(aps)
            
        self.results = mAPs
        return self.results

    def load_groundtruth(self, groundtruth_file, scaling=True):
        w,h = tuple(groundtruth_file.split('_')[1].split('x'))
        w = float(w)
        h = float(h)
        #print('Width: {}, Height: {}'.format(w,h))
        groundtruth_boxes = []
        with open(groundtruth_file, "r") as f:
            for line in f:
                box = line.strip().split(" ")
                box =  [box[0],*list(map(float, box[1:]))]
                if scaling:
                    box[1] = box[1] / w
                    box[2] = box[2] / h
                    box[3] = box[3] / w
                    box[4] = box[4] / h

                groundtruth_boxes.append(box)

        return groundtruth_boxes

    def load_prediction(self, prediction_file, convert_to_coco=False):        
        prediction_boxes = []
        with open(prediction_file, "r") as f:
            for line in f:
                box = line.strip().split(" ")
                box =  [box[0],*list(map(float, box[1:]))]

                #if convert_to_coco:
                #    coco_label = self.mappings[box[0]]
                #    if coco_label == 'None':
                #        box[0] = str(box[0])
                #    else:                        
                #        box[0] = coco_label
                #else:   # convert index to COCO label                    
                #    box[0] = self.labels[int(box[0])]
                if box[5] > self.threshold:
                    prediction_boxes.append(box)
        
        prediction_boxes = sorted(prediction_boxes, key=lambda x: x[5], reverse=True)

        return prediction_boxes
    
    def compute_ap(self, precisions, recalls):
        mrec = np.concatenate(([0.], recalls, [1.]))
        mpre = np.concatenate(([0.], precisions, [0.]))

        # calculate the max for every bar
        for i in range(mpre.size - 1, 0, -1):
            mpre[i-1] = np.maximum(mpre[i-1], mpre[i])

        # find the indices where the recall changes
        i = np.where(mrec[1:] != mrec[:-1])[0]
        
        # calculate the area under the curve
        ap = np.sum((mrec[i+1]-mrec[i]) * mpre[i+1])
        return ap

    def calculate_map(self, groundtruth_boxes, prediction_boxes):
        global_aps = []
        iou_thresholds = [0.5+x/20 for x in range(0,10)]

        # calculate the true positives per class
        gt_class_counter = {}
        for gt_box in groundtruth_boxes:
            if gt_box[0] in gt_class_counter:
                gt_class_counter[gt_box[0]] += 1
            else:
                gt_class_counter[gt_box[0]] = 1                    
        
        labels = gt_class_counter.keys()        

        for label in labels:
            gt_class_boxes = [box for box in groundtruth_boxes if box[0]==label]
            gt_count = len(gt_class_boxes)

            # update global GT counter per class
            if label in self.global_gt_counter:
                self.global_gt_counter[label] += gt_count
            else:
                self.global_gt_counter[label] = gt_count

            #print(gt_box)            
            aps = []
            precisions = []
            recalls = []

            # order predications by score in descending order
            pred_boxes = [box for box in prediction_boxes if box[0]==label]
            pred_boxes = sorted(pred_boxes, key=lambda x: x[5], reverse=True)
            #print(pred_boxes)
            pred_count = len(pred_boxes)

            # calculate the AP at different IOU thresholds
            for iou_threshold in iou_thresholds:                
                #true_positives = []
                tp_mask = [0] * pred_count
                associated_mask = [0] * gt_count
                idx = 0

                # iterate over prediction boxes for the class
                for pred_box in pred_boxes:
                    # check if this was not already associated with another ground truth box for this class
                    gtidx = 0
                    # search for an associated ground truth box
                    for gt_box in gt_class_boxes:
                        # check if this ground_truth box was previously associated with a prediction box
                        if not associated_mask[gtidx]:
                            # if not, then calculate the IoU                            
                            iou = self.compute_iou(gt_box, pred_box)
                            #breakpoint()
                            if iou >= iou_threshold:      # True positive
                                associated_mask[gtidx] = 1
                                tp_mask[idx] = 1
                                #true_positives.append((gt_box,pred_box))
                        
                        gtidx += 1
                    
                    idx+=1
                
                # calculate AP for this threshold and class                
                tp_count = np.sum(tp_mask)
                fp_count = pred_count - tp_count
                #print(f'Class: {label}, Threshold: {iou_threshold}, TP: {tp_count}, FP: {fp_count}')                
                
                if pred_count > 0:
                    precision = float(tp_count) / pred_count
                    recall = float(tp_count) / gt_count
                    precisions.append(precision)
                    recalls.append(recall)

            # compute AP for this class
            ap = self.compute_ap(precisions, recalls)            
            
            if label in self.global_aps:
                self.global_aps[label].append(ap)
            else:
                self.global_aps[label] = list([ap])
                

    def compute_iou(self, groundtruth_box, prediction_box):
        x1, y1, bx1, by1 = groundtruth_box[1:]
        w1 = bx1-x1
        h1 = by1-y1
        x2, y2, bx2, by2 = prediction_box[1:5]
        w2 = bx2-x2
        h2 = by2-y2

        #print(groundtruth_box[:5], " === ", prediction_box[:5])
        inter_area = max(0, min(bx1, bx2) - max(x1, x2)) * max(0, min(by1, by2) - max(y1, y2))
        union_area = w1 * h1 + w2 * h2 - inter_area

        #print(inter_area, " - ", union_area)
        iou = inter_area / union_area        

        return iou
    
    def plot_results(self):
        for label, mAP in self.results.items():
            print(f'{label} mAP {mAP*100}%')

        x_axis = list(self.results.keys())
        y_axis = [y*100 for y in list(self.results.values())]

        plt.bar(x_axis, y_axis)
        plt.xlabel("Class")
        plt.ylabel("mAP (%)")
        plt.suptitle(f'Overal mAP {np.mean(y_axis):.2f}%')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='calc_map.py',
        description='Calculate the mAP for the object detection prediction'        
    )

    #parser.add_argument('video_name', help='The name of the video sequence, e.g. Kimono.')
    parser.add_argument('--ds', dest='dataset_name', required=False, default='SFU-HW-Objects', help='Name of the dataset. Defaults to SFU-HW-Objects.')
    parser.add_argument('--threshold', dest='threshold', action='store', default=0.5, type=float, help='The threshold for the prediction confidence to consider the prediction.')
    args = parser.parse_args() 

    mappings = load_imagenet_coco('scripts/objectdetection/imagenet_coco.csv') 
    labels = load_coco_labels('scripts/objectdetection/coco_labels.csv')
    
    directory = f"datasets/{args.dataset_name}/"
    videos = glob.glob(f"{directory}videos/*.mp4")

    for v in videos:
       fname = os.path.basename(v) 
       video_name = v.split("/")[-1].split("_")[0]
       args.video_name = video_name 
       print("video:", args.video_name)
       evaluator_onePass = Evaluator(f'datasets/{args.dataset_name}/ground-truth/{args.video_name}', f'datasets/{args.dataset_name}/predictionsOnepass/{args.video_name}', args.threshold, labels=labels)
       evaluator_split = Evaluator(f'datasets/{args.dataset_name}/ground-truth/{args.video_name}', f'datasets/{args.dataset_name}/predictionsSplit/{args.video_name}', args.threshold, labels=labels)
       mAPs_onePass = evaluator_onePass.evaluate(is_imagenet=False)
       mAPs_split = evaluator_split.evaluate(is_imagenet=False)
       print("** OnePass inference")
       evaluator_onePass.plot_results()
       print("** Split inference")
       evaluator_split.plot_results()
       print("\n")
