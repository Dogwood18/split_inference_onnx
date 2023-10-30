'''
 * License: 5G-MAG Public License (v1.0)
 * Author: Liangping Ma 
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
import onnx
import onnxruntime
from onnx.utils import extract_model

model = onnx.load("./models/retinanet.onnx")
onnx.checker.check_model(model)

VERBOSE = 0

if VERBOSE:
  # model is an onnx model
  graph = model.graph
  # graph inputs
  print("---graph input----")
  for input_name in graph.input:
     print(input_name)
  # graph parameters
  print("---graph init----")
  for init in graph.initializer:
     print(init.name)
  # graph outputs
  print("---graph output----")
  for output_name in graph.output:
     print(output_name)
  # iterate over nodes --> even more verbose, enable if desired
  '''
  for node in graph.node:
    # node inputs
    for idx, node_input_name in enumerate(node.input):
        print(idx, node_input_name)
    # node outputs
    for idx, node_output_name in enumerate(node.output):
        print(idx, node_output_name)
  '''

device = ['CUDAExecutionProvider', "CPUExecutionProvider"]

# information obtained by examining the retinanet graph using Netron
''' split points (node first, output next)
/backbone/fpn/extra_blocks/p6/Conv
name: /backbone/fpn/extra_blocks/p6/Conv_output_0


/backbone/fpn/layer_blocks.1/layer_blocks.1.0/Conv
name: /backbone/fpn/layer_blocks.1/layer_blocks.1.0/Conv_output_0


/backbone/fpn/layer_blocks.0/layer_blocks.0.0/Conv
name: /backbone/fpn/layer_blocks.0/layer_blocks.0.0/Conv_output_0

name: /Cast_3_output_0

name: /Shape_1_output_0

/backbone/fpn/layer_blocks.2/layer_blocks.2.0/Conv
name: /backbone/fpn/layer_blocks.2/layer_blocks.2.0/Conv_output_0

name: /Cast_4_output_0

name: /Shape_output_0
'''

fpn_layers_out = ['/backbone/fpn/layer_blocks.0/layer_blocks.0.0/Conv_output_0', '/backbone/fpn/layer_blocks.1/layer_blocks.1.0/Conv_output_0', '/backbone/fpn/layer_blocks.2/layer_blocks.2.0/Conv_output_0', '/backbone/fpn/extra_blocks/p6/Conv_output_0']

split_points_part1 = ['/Shape_output_0','/Cast_4_output_0'] + fpn_layers_out + ['/Cast_3_output_0', '/Shape_1_output_0']
print(split_points_part1)

extract_model('./models/retinanet.onnx', './models/retinanet_part1.onnx', input_names=['input_images'], output_names=split_points_part1)

input_names_part2 = ['input_images'] + fpn_layers_out
end_point = ['/transform/Unsqueeze_12_output_0'] # can be thrown away
output_names_part2 = ['2734', '2712',  '2713'] + end_point
print(output_names_part2)
extract_model('./models/retinanet.onnx', './models/retinanet_part2.onnx', input_names=input_names_part2, output_names=output_names_part2)
