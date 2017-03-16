# Image Captioning
### Group Number : 06
### Group Members:
1. Kriti Joshi (13358)
2. Pramod Chunduri (13221)
### References :
1. Show and Tell : A Neural Image Caption Generator
2. Deep Visual-Semantic Alignments for Generating Image Descriptions (for further work)
### Platform/Frameworks:
1. Python/IPython (preprocessing work)
2. lua with torch (training - testing of neural network)
### Dataset/Model:
1. CNN : Deep convolutional neural network trained on imagenet
We tried to work with VGG16 but landed into memory errors on using GPU so we had to shift to `nin_imagenet_conv.caffemodel` which only contains 30 layers unlike the 38 in VGG16 net.
2. Dataset : `MSCOCO Val2014` 
### Directories:
* `coco`: Contains images for validation, testing and training. Right now it contains `coco_raw.json` file with information regarding how to split the first 6000 images in MSCOCO dataset into val, test and train.
* `misc`: Miscelaneous helpful functions for data loading, language model creation etc. are dtored in this folder.
* `checkpoints`: After training, a .t7 file and .json file gets created which are used for performance evaluation on test set. 
* `my_train`: Preprocessing of coco/coco_raw.json creates .h5 file and .json file in this folder.
* `vis`: The predicted captions are stored in this file. 
### Downloads:
* `coco-caption` directoty from [Ref] to calculate language-scores.
* [Val2014] dataset in `coco` folder
* `SCG-COCO-val2014-proposals`: A collection of proposals for val2014 MSCOCO dataset [Download].
### Execution Instructions:
Run `proposals.m` after downloading `SCG-COCO-val2014-proposals` followed by running `coco_preprocess.ipynb` to obtain **coco_raw.json** in coco folder. 
```sh
$ python prepro.py
```
Preprocesses the `coco/coco_raw.json` file to form **data.json** and **data.h5** file in `my_train` folder. These files are used to retrieve correct images during testing and training phase. This splits val2014 images into val, test and train.
```sh
$ th train.lua
```
Trains the LSTM used and creates **model_id.t7** and **model_id.json** files in `checkpoints` folder. 
```sh
$ th eval.lua
```
Generates captions for the test images and stores them in `vis.json` in `vis` folder along with the corresponding images that are copied to `vis/imgs`.

[Val2014]:<http://mscoco.org/dataset/#download>
[Ref]:<https://github.com/karpathy/neuraltalk2>
[Download]:<https://data.vision.ee.ethz.ch/jpont/mcg/SCG-COCO-val2014-proposals.tgz>

Code was referred from [Ref].