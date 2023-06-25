[![NVIDIA Source Code License](https://img.shields.io/badge/license-NSCL-blue.svg)](https://github.com/NVlabs/SegFormer/blob/master/LICENSE)
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)

# SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers


## Installation

环境，为了适配onnx推理，安装的库版本与原始segformer不同

onnx                      1.14.0  
onnxruntime-gpu           1.15.1  
torch                     1.11.0+cu113  
torchvision               0.12.0+cu113  
mmcv-full                 1.7.1  
timm                      0.9.2  


## Evaluation

```
# Single-gpu testing
python tools/test.py

```
## onnx模型精度
per class results:

+---------------------+-------+-------+  
| Class               | IoU   | Acc   |  
+---------------------+-------+-------+  
| wall                | 72.07 | 86.81 |  
| building            | 79.45 | 92.07 |  
| sky                 | 92.76 | 96.6  |  
| floor               | 76.12 | 88.06 |  
| tree                | 70.99 | 86.75 |  
| ceiling             | 79.91 | 89.21 |  
| road                | 77.52 | 86.01 |  
| bed                 | 81.18 | 91.3  |  
| windowpane          | 55.16 | 71.81 |  
| grass               | 65.03 | 84.52 |  
| cabinet             | 53.38 | 69.29 |  
| sidewalk            | 57.18 | 75.61 |  
| person              | 72.28 | 89.42 |  
| earth               | 30.21 | 41.78 |  
| door                | 37.87 | 53.51 |  
| table               | 47.85 | 62.15 |  
| mountain            | 48.16 | 64.42 |  
| plant               | 48.83 | 59.17 |  
| curtain             | 66.15 | 78.84 |  
| chair               | 47.22 | 64.41 |  
| car                 | 79.56 | 89.83 |  
| water               | 50.72 | 71.74 |  
| painting            | 66.68 | 81.57 |  
| sofa                | 54.99 | 72.48 |  
| shelf               | 38.87 | 58.62 |  
| house               | 37.59 | 47.37 |  
| sea                 | 47.58 | 63.24 |  
| mirror              | 52.58 | 64.17 |  
| rug                 | 52.69 | 57.86 |  
| field               | 26.11 | 39.14 |  
| armchair            | 32.8  | 52.19 |  
| seat                | 55.14 | 72.22 |  
| fence               | 38.32 | 53.61 |  
| desk                | 40.22 | 58.99 |  
| rock                | 32.56 | 50.67 |  
| wardrobe            | 42.13 | 54.64 |  
| lamp                | 50.07 | 62.97 |  
| bathtub             | 61.89 | 69.67 |  
| railing             | 29.51 | 41.04 |  
| cushion             | 44.78 | 58.94 |  
| base                | 18.2  | 26.86 |  
| box                 | 15.07 | 20.82 |  
| column              | 33.82 | 45.21 |  
| signboard           | 31.26 | 40.93 |  
| chest of drawers    | 38.42 | 57.21 |  
| counter             | 35.7  | 47.54 |  
| sand                | 27.96 | 43.13 |  
| sink                | 59.83 | 71.07 |  
| skyscraper          | 56.83 | 67.81 |  
| fireplace           | 62.81 | 77.54 |  
| refrigerator        | 56.87 | 73.96 |  
| grandstand          | 37.48 | 65.08 |  
| path                | 20.05 | 29.06 |  
| stairs              | 24.91 | 32.93 |  
| runway              | 65.95 | 86.25 |  
| case                | 36.91 | 47.43 |  
| pool table          | 88.63 | 94.28 |  
| pillow              | 40.52 | 50.5  |  
| screen door         | 37.3  | 46.98 |  
| stairway            | 26.1  | 31.77 |  
| river               | 12.92 | 20.06 |  
| bridge              | 27.13 | 36.15 |  
| bookcase            | 35.73 | 56.06 |  
| blind               | 35.74 | 39.57 |  
| coffee table        | 49.53 | 68.19 |  
| toilet              | 75.72 | 86.55 |  
| flower              | 31.68 | 44.45 |  
| book                | 42.11 | 58.14 |  
| hill                | 9.86  | 15.34 |  
| bench               | 36.05 | 47.18 |  
| countertop          | 48.29 | 66.62 |  
| stove               | 59.33 | 67.26 |  
| palm                | 45.32 | 60.22 |  
| kitchen island      | 26.92 | 47.4  |  
| computer            | 53.54 | 65.17 |  
| swivel chair        | 32.0  | 39.26 |  
| boat                | 45.08 | 55.37 |  
| bar                 | 29.56 | 32.85 |  
| arcade machine      | 39.55 | 42.99 |  
| hovel               | 30.28 | 33.72 |  
| bus                 | 75.33 | 81.22 |  
| towel               | 48.7  | 63.43 |  
| light               | 43.69 | 48.25 |  
| truck               | 25.69 | 36.92 |  
| tower               | 7.66  | 9.6   |  
| chandelier          | 55.55 | 66.96 |  
| awning              | 19.13 | 23.36 |  
| streetlight         | 15.76 | 19.13 |  
| booth               | 39.0  | 42.1  |  
| television receiver | 58.2  | 70.01 |  
| airplane            | 54.32 | 62.43 |  
| dirt track          | 17.8  | 26.74 |  
| apparel             | 12.04 | 22.65 |  
| pole                | 15.58 | 20.24 |  
| land                | 1.73  | 2.2   |  
| bannister           | 3.17  | 4.5   |  
| escalator           | 23.78 | 29.32 |  
| ottoman             | 18.24 | 22.72 |  
| bottle              | 22.06 | 32.99 |  
| buffet              | 32.33 | 35.22 |  
| poster              | 16.44 | 26.17 |  
| stage               | 12.36 | 16.95 |  
| van                 | 35.33 | 48.07 |  
| ship                | 63.51 | 83.74 |  
| fountain            | 10.07 | 10.31 |  
| conveyer belt       | 52.77 | 62.52 |  
| canopy              | 12.88 | 14.0  |  
| washer              | 63.29 | 68.8  |  
| plaything           | 17.55 | 25.68 |  
| swimming pool       | 37.83 | 60.44 |  
| stool               | 27.12 | 35.8  |  
| barrel              | 35.52 | 63.85 |  
| basket              | 23.48 | 31.59 |  
| waterfall           | 53.84 | 65.12 |  
| tent                | 91.59 | 96.71 |  
| bag                 | 11.13 | 16.43 |  
| minibike            | 52.51 | 63.79 |  
| cradle              | 76.1  | 89.07 |  
| oven                | 26.26 | 54.54 |  
| ball                | 43.36 | 53.27 |  
| food                | 36.94 | 41.13 |  
| step                | 7.29  | 8.64  |  
| tank                | 34.84 | 37.02 |  
| trade name          | 16.69 | 17.98 |  
| microwave           | 45.05 | 49.95 |  
| pot                 | 24.55 | 28.09 |  
| animal              | 53.1  | 56.91 |  
| bicycle             | 41.4  | 65.86 |  
| lake                | 34.88 | 40.95 |  
| dishwasher          | 48.42 | 58.53 |  
| screen              | 52.12 | 70.4  |  
| blanket             | 14.41 | 17.85 |  
| sculpture           | 39.25 | 43.32 |  
| hood                | 41.0  | 45.04 |  
| sconce              | 33.23 | 42.16 |  
| vase                | 23.0  | 35.06 |  
| traffic light       | 16.8  | 24.43 |  
| tray                | 2.15  | 2.85  |  
| ashcan              | 29.24 | 38.38 |  
| fan                 | 47.27 | 54.92 |  
| pier                | 46.14 | 60.71 |  
| crt screen          | 0.02  | 0.06  |  
| plate               | 46.28 | 56.84 |  
| monitor             | 6.94  | 7.63  |  
| bulletin board      | 31.01 | 42.26 |  
| shower              | 0.56  | 0.75  |  
| radiator            | 45.61 | 54.08 |  
| glass               | 5.22  | 5.63  |  
| clock               | 23.98 | 27.8  |  
| flag                | 27.45 | 29.09 |  
+---------------------+-------+-------+  
Summary:  

+--------+-------+-------+-------+  
| Scope  | mIoU  | mAcc  | aAcc  |  
+--------+-------+-------+-------+  
| global | 40.02 | 50.42 | 78.99 |  
+--------+-------+-------+-------+  
