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

import torch
import torchvision
from torchvision.ops import masks_to_boxes
import onnx
import onnxruntime
import pickle
import numpy as np
from PIL import Image
import cv2
import sys
import glob
import os
from tqdm import tqdm, trange
from google.protobuf.json_format import MessageToDict
import argparse

SCORE_THRESHOLD = 0.5
DEBUG = 1 # 1 means printing values of some variables
SAVE_FEATURES_IN_FILEs = 1
ERR_THRESHOLD = 0.00001 # error threshold for comparison

def load_coco_labels(filename):
    labels = {}
    with open(filename, "rt") as f:
        for s in f:
            tokens = s.split(',')
            labels[int(tokens[0])] = tokens[1].rstrip()
        
    return labels

class Inference:
    def __init__(self, model_path, model_path_1, model_path_2, dataset_name, bMask = False):        
        self.device = ['CUDAExecutionProvider', "CPUExecutionProvider"]  #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
        self.model_name = model_path
        self.model_name_1 = model_path_1
        self.model_name_2 = model_path_2
        self.dataset_name = dataset_name

        self.bMask = bMask

        # Preprocess the input
        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])        

        # Set the model to evaluation mode        
        self.model = onnx.load(model_path)
        self.model_1 = onnx.load(model_path_1)
        self.model_2 = onnx.load(model_path_2)
        onnx.checker.check_model(self.model)
        onnx.checker.check_model(self.model_1)
        onnx.checker.check_model(self.model_2)

        # print inputs dimension info
        for _input in self.model.graph.input:
            m_dict = MessageToDict(_input)
            dim_info = m_dict.get("type").get("tensorType").get("shape").get("dim")
            #print(dim_info)

        for _input in self.model_1.graph.input:
            m_dict = MessageToDict(_input)
            dim_info = m_dict.get("type").get("tensorType").get("shape").get("dim")
            #print(dim_info)

        for _input in self.model_2.graph.input:
            m_dict = MessageToDict(_input)
            dim_info = m_dict.get("type").get("tensorType").get("shape").get("dim")
            #print(dim_info)

        self.session = onnxruntime.InferenceSession(model_path, providers=self.device) # entire model (no split)
        self.session_1 = onnxruntime.InferenceSession(model_path_1, providers=self.device) # for part 1
        self.session_2 = onnxruntime.InferenceSession(model_path_2, providers=self.device) # for part 2

    def run_inference(self, classlabels=None):
        directory = f"datasets/{self.dataset_name}/"
        videos = glob.glob(f"{directory}videos/*.mp4")

        for v in videos: 
        #print("directory=", directory)
        #print("videos=", videos)
        #for v in [videos[0]]: # test   
            name = os.path.basename(v).split('_')[0]
            fname = os.path.basename(v)
            self.infer(name, directory, f"{directory}videos/{fname}", classlabels=classlabels)

    def infer(self, name, path, video, classlabels=None):
        cap = cv2.VideoCapture(video)
        out_dir = f'{path}predictionsOnepass/{name}/' 
        os.makedirs(out_dir, exist_ok=True)
        out_dir_2 = f'{path}predictionsSplit/{name}/'
        os.makedirs(out_dir_2, exist_ok=True)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        success = True
        frame_number = 0

        print(f'starting inference for {name} with {frame_count} frames')

        for frame_number in range(frame_count):
        #for frame_number in trange(2): # test
            success, image = cap.read()    
            if not success:
                break    
            
            height, width = image.shape[:2]

            preprocess = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()                
            ])

            image = preprocess(image).detach().cpu().numpy()

            input_tensor = np.expand_dims(image, axis=0)
            
            # one-pass inference (i.e., without split)
            ortinputs = {self.session.get_inputs()[0].name: input_tensor}
            output = self.session.run(None, ortinputs)

            # inputs for part 1
            ortinputs_1 = {self.session_1.get_inputs()[0].name: input_tensor}
            
            # inference for part 1
            #   fpn_0, fpn_1, fpn_2, fpn_6 in the output below are the feature layers, corresponding to the layers in fpn_layers_out next
            shape0, cast4, fpn_0, fpn_1, fpn_2, fpn_6, cast3, shape1 = self.session_1.run(None, ortinputs_1)
            feature_fn = f'{out_dir_2}/{os.path.basename(video).replace(".mp4","")}_seq_{frame_number:03d}.npz'
            np.savez(feature_fn, fpn_0=fpn_0, fpn_1=fpn_1, fpn_2=fpn_2, fpn_6=fpn_6) # save features, to be used as input to part 2

            # print the dimensions of feature maps
            if DEBUG:
              print("fpn_0.shape=", fpn_0.shape)
              print("fpn_1.shape=", fpn_1.shape)
              print("fpn_2.shape=", fpn_2.shape)
              print("fpn_6.shape=", fpn_6.shape)
            # feature layers to split
            fpn_layers_out = ['/backbone/fpn/layer_blocks.0/layer_blocks.0.0/Conv_output_0', '/backbone/fpn/layer_blocks.1/layer_blocks.1.0/Conv_output_0', \
                              '/backbone/fpn/layer_blocks.2/layer_blocks.2.0/Conv_output_0', '/backbone/fpn/extra_blocks/p6/Conv_output_0']
           
            # inputs for part 2
            #   note: input_tensor is part of th input, to be used by part 2 to calculate shape
            #   pass a dummy tensor with the same shape as input_tensor
            #input_tensor_dummy = np.empty_like(input_tensor)
            #input_tensor_dummy[:] = 0
            #print("input_tensor.shape=", *input_tensor.shape)
            input_tensor_dummy = np.random.randn(*input_tensor.shape).astype(np.float32)
            features_saved = np.load(feature_fn)              
            fpn_0_saved = features_saved['fpn_0']
            fpn_1_saved = features_saved['fpn_1']
            fpn_2_saved = features_saved['fpn_2']
            fpn_6_saved = features_saved['fpn_6']
            ortinputs_2 = {self.session_1.get_inputs()[0].name: input_tensor_dummy, fpn_layers_out[0]: fpn_0_saved, fpn_layers_out[1]: fpn_1_saved, \
              fpn_layers_out[2]: fpn_2_saved, fpn_layers_out[3]: fpn_6_saved} 
            # inference for part 2
            output_2 = self.session_2.run(None, ortinputs_2)
            
            # For the SFU-HW dataset, the features for each video frame is about 22MB. The dataset has more than 7000 video frames. If 0, delete files after part 2 inference.
            if SAVE_FEATURES_IN_FILEs == 0: 
               if os.path.isfile(feature_fn):
                  os.remove(feature_fn) 

            # check if output and output_2 are identical
            if DEBUG:
                #print("output[0]\n", output[0])
                #print("output[1]\n", output[1])
                #print("output[2]\n", output[2])
                rel_err_0 = np.sum(np.square(output[0] - output_2[0]))/np.sum(np.square(output[0]))
                rel_err_1 = np.sum(np.square(output[1] - output_2[1]))/np.sum(np.square(output[1]))
                rel_err_2 = np.sum(np.square(output[2] - output_2[2]))/np.sum(np.square(output[2]))
                
                if rel_err_0 + rel_err_1 + rel_err_2 ==0:
                   print("\n", f'{video} {frame_number:03d}:', "split inference and onepass inference have identical results\n")
                elif (rel_err_0 > ERR_THRESHOLD) or (rel_err_1 > ERR_THRESHOLD) or (rel_err_2 > ERR_THRESHOLD):
                   print("\n", f'{video} {frame_number:03d}:', "split inference and onepass inference have different results\n")
                   print("\n", "rel_err_0=", rel_err_0, "\trel_err_1=", rel_err_1, "\trel_err_2=", rel_err_2)
                   print("\n", "denominators 0, 1, 2 =", np.sum(np.square(output[0])), np.sum(np.square(output[1])), np.sum(np.square(output[2])))
                   
                else:
                   print("\n", f'{video} {frame_number:03d}:', "split inference and onepass inference have similar results\n")
                   print("Normalized MSE for output[0] =", rel_err_0)
                   print("Normalized MSE for output[1] =", rel_err_1)
                   print("Normalized MSE for output[2] =", rel_err_2)
                   
            #print("***** non-split inference: output = ******\n", output) 
            #print("***** split inference: output_2 = ******\n", output_2) 
            if self.bMask:
                labels, boxes, scores = self.convert_mask(torch.Tensor(output_2[0].squeeze()))
            else:
                # Convert the output to numpy arrays
                boxes = output[0]
                labels = output[2]
                scores = output[1]            
                boxes_2 = output_2[0]
                labels_2 = output_2[2]
                scores_2 = output_2[1]

            output_fn = f'{out_dir}/{os.path.basename(video).replace(".mp4","")}_seq_{frame_number:03d}.txt'
            output_fn_2 = f'{out_dir_2}/{os.path.basename(video).replace(".mp4","")}_seq_{frame_number:03d}.txt'

            with open(output_fn, 'w+') as fn:        
                # Print the detected objects
                for box, label, score in zip(boxes, labels, scores):
                    if score > SCORE_THRESHOLD:
                        if classlabels is not None: 
                            fn.write(f'{classlabels[label]} {float(box[0])/width} {float(box[1])/height} {float(box[2])/width} {float(box[3])/height} {score}\n') 
                        else:
                            fn.write(f'{label} {int(box[0])} {float(box[0])/width} {float(box[1])/height} {float(box[2])/width} {float(box[3])/height} {score}\n')

            with open(output_fn_2, 'w+') as fn:
                # Print the detected objects
                for box, label, score in zip(boxes_2, labels_2, scores_2):
                    if score > SCORE_THRESHOLD:
                        if classlabels is not None:
                            fn.write(f'{classlabels[label]} {float(box[0])/width} {float(box[1])/height} {float(box[2])/width} {float(box[3])/height} {score}\n')
                        else:
                            fn.write(f'{label} {int(box[0])} {float(box[0])/width} {float(box[1])/height} {float(box[2])/width} {float(box[3])/height} {score}\n')


    def convert_mask(self, mask):
        bboxes = []
        scores = []        
        labels = []
        obj_ids = torch.unique(mask)        
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        bboxes = masks_to_boxes(masks)        

        return labels, bboxes, scores

# Example: python3 scripts/objectdetection/infer_split.py SFU-HW-Objects models/retinanet.onnx models/retinanet_part1.onnx models/retinanet_part2.onnx
if __name__ == "__main__":    
    parser = argparse.ArgumentParser(
            prog='infer_split.py',

        description='Run onePass inference and split inference using ONNX models'      
    )

    parser.add_argument('dataset_name', help='Dataset name')
    parser.add_argument('model_location', help='Path to ONNX entire Model')
    parser.add_argument('model_1_location', help='Path to ONNX Model part 1')
    parser.add_argument('model_2_location', help='Path to ONNX Model part 2')
    parser.add_argument('--mask', dest='mask', action='store_true', help='Indicates if output of model is a Mask and needs to be converted')
    args = parser.parse_args()    

    labels = load_coco_labels('scripts/objectdetection/coco_labels.csv')

    infer = Inference(args.model_location, args.model_1_location, args.model_2_location, args.dataset_name, bMask=args.mask)    
    infer.run_inference(labels)





