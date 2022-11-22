# SDFA-Net_pytorch
This is the official repo for our work 'Self-distilled Feature Aggregation for Self-supervised Monocular Depth Estimation' (ECCV 2022).

## Setup
We built and ran the repo with CUDA 10.2, Python 3.7.11, and Pytorch 1.7.0. For using this repo, we recommend creating a virtual environment by [Anaconda](https://www.anaconda.com/products/individual). Please open a terminal in the root of the repo folder for running the following commands and scripts.
```
conda env create -f environment.yml
conda activate pytorch170cu10
```

## Pre-trained models
|Model Name|Dataset(s)|Abs Rel.|Sq Rel.|RMSE|RMSElog|A1|
|----------|----------|--------|-------|----|-------|--|
|SDFA-Net-SwinTM_stage1_384 [Baidu](https://pan.baidu.com/s/1E7YXzMClwiBLn4T5CNzn3A)/[Google](https://drive.google.com/file/d/1wpOj39KGgKGHpG_Z_MRGAUgyxF__3sVz/view?usp=sharing)|K|0.100|0.631|4.090|0.183|0.890|
|SDFA-Net-SwinTM_384 [Baidu](https://pan.baidu.com/s/1sqPV3WXqzoT3VzO1ZkNGSg)/[Google](https://drive.google.com/file/d/1RxCJ6lz6MpeHIPLNFmm1hJikeDUOBXu8/view?usp=sharing)|K|0.090|0.538|3.896|0.169|0.906|
|SDFA-Net-SwinTM_CS+K_384 [Baidu](https://pan.baidu.com/s/1m2ybgWahi6EmcOoU1cYDWg)/[Google](https://drive.google.com/file/d/11QJJ1WEQ8Z80JUz7zCmq9t9LBMVvzqYD/view?usp=sharing)|CS+K|0.085|0.531|3.888|0.167|0.911|

* **code for all the download links of Baidu is `sdfa`**
## Prediction
To predict depth maps for your images, please firstly download the pretrained model from the column named `Model Name` in the above table. After unzipping the downloaded model, you could predict the depth maps for your images by
```
python predict.py\
 --image_path <path to your image or folder name for your images>\
 --exp_opts options/_base/networks/sdfa_net.yaml\
 --model_path <path to the downloaded or trained model (.pth)>
```
You also could set `--input_size` to decide the size that the images are reshaped before they are input to the model. If you want to predict on CPU, please set `--cpu`. The depth results `<image name>_pred.npy` and the visualization results `<image name>_visual.png` will be saved in the same folder as the input images.  

## Data preparation
#### Set Data Path
We give an example `path_example.py` for setting the path in the repository.
Please create a python file named `path_my.py` and copy the code in `path_example.py` to the `path_my.py`. Then you can replace the used paths to your folder in the `path_my.py`.
the folder for each dataset should be organized like:
```
<root of kitti>
|---2011_09_26
|   |---2011_09_26_drive_0001_sync
|   |   |---image_02
|   |   |---image_03
|   |   |---velodyne_points
|   |   |---...
|   |---2011_09_26_drive_0002_sync
|   |   |---image_02
|   |   |---image_03
|   |   |---velodyne_points
|   |   |---...
|   '''
|---2011_09_28
|   |--- ...
|---gt_depths_raw.npz (for raw Eigen test set)
|---gt_depths_improved.npz (for improved Eigen test set)
```
```
<root of cityscapes>
|---leftImg8bit
|   |---train
|   |   |---aachen
|   |   |   |---aachen_000000_000019_leftImg8bit.png
|   |   |   |---aachen_000001_000019_leftImg8bit.png
|   |   |   |---...
|   |   |---bochum
|   |   |---...
|   |---train_extra
|   |   |---augsburg
|   |   |---...
|   |---test
|   |   |---...
|   |---val
|   |   |---...
|---rightImg8bit
|   |--- ...
|---camera
|   |--- ...
|---disparity
|   |--- ...
|---gt_depths (for evaluation)
|   |---000_depth.npy
|   |---001_depth.npy
|   |--- ...
```
#### KITTI
For training the methods on the KITTI dataset (the Eigen split), you should download the entire KITTI dataset (about 175GB) by:
```
wget -i ./datasets/kitti_archives_to_download.txt -P <save path>
```
And you could unzip them with:
```
cd <save path>
unzip "*.zip"
```

For evaluating the methods on the KITTI (Eigen raw test set), you should further generate the ground-truth depth file by (as done in the [Monodepth2](https://github.com/nianticlabs/monodepth2)):

```
python datasets/utils/export_kitti_gt_depth.py --data_path <root of KITTI> --split raw
```
If you want to evaluate the method on the KITTI improved test set, you should download the `annotated depth maps` (about 15GB) at [Here](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) and unzip it in the root of the KITTI dataset. Then you could generate the imporved ground-truth depth file by:
```
python datasets/utils/export_kitti_gt_depth.py --data_path <root of KITTI> --split improved
```
As an alternative, we provide the Eigen test subset and the generated `gt_depth` files [Here](https://pan.baidu.com/s/1NejtxajjJt6pQ-VIRJDcUg) (about 2GB) for the people who just want to do the evaluation.
##### Cityscapes (Optional)
Cityscapes could be used to jointly train the model with KITTI, which is helpful to improve the performance of the model. If you want to use the Cityscapes, please download the following parts of the dataset at [Here](https://www.cityscapes-dataset.com/downloads/) and unzip them to your `<root of cityscapes>` (Note: For some files, you should apply for download permission by email.):
```
leftImg8bit_trainvaltest.zip (11GB)
leftImg8bit_trainextra.zip (44GB)
rightImg8bit_trainvaltest.zip (11GB)
rightImg8bit_trainextra.zip (44GB)
camera_trainvaltest.zip (2MB)
camera_trainextra.zip (8MB)
```
Then, please generate the camera parameter matrices by:
```
python datasets/utils/export_cityscapes_matrix.py
```
## Evaluation
To evaluate the methods on the prepared dataset, you could simply use 
```
python evaluate.py\
 --exp_opts <path to the method EVALUATION option>\
 --model_path <path to the downloaded or trained model>
```
We provide the EVALUATION option files in `options/SDFA-Net/eval/*`. Here we introduce some important arguments.
|Argument|Information|
|--------|-----------|
|`--visual_list`|The samples which you want to save the output (path to a `.txt` file)|
|`--save_pred`|Save the predicted depths of the samples which are in `--visual_list`|
|`--save_visual`|Save the visualization results of the samples which are in `--visual_list`|
|`-gpp`|Adopt different post-processing steps.|

The output files are saved in `eval_res\` by default. Please check `evaluate.py` for more information about arguments.

## Training
Plese firstly download the pretrained Swin-trainsformer (Tiny Size) in their [official repo](https://github.com/microsoft/Swin-Transformer) and don't forget to set the path in `path_my.py` to the downloaded model. Then, you could train SDFA-Net simply use the commands provided in `options/SDFA-Net/train/train_scripts.sh`.
For example, you could use the following commands for training the SDFA-Net on KITTI. As mentioned in our paper, we disabled the self-distilled forward propagation and corresponding losses in early training stage. The training is devideded into two stages:
```
# train SDFA-Net at stage1
CUDA_VISIBLE_DEVICES=0 python\
 train_dist.py\
 --name SDFA-Net-SwinT-M_192Crop_KITTI_S_St1_B12\
 --exp_opts options/SDFA-Net/train/sdfa_net-swint-m_192crop_kitti_stereo_stage1.yaml\
 --batch_size 12\
 --epoch 25\
 --visual_freq 2000\
 --save_freq 5
```
After finishing the stage1, please copy the path to the last saved model `<exp_log>/model/last_model.pth` into `<path to .pth>`, and use the following command to continue the training:
```
# train SDFA-Net at stage2
CUDA_VISIBLE_DEVICES=0 python\
 train_dist.py\
 --name SDFA-Net-SwinT-M_192Crop_KITTI_S_St2_B12\
 --exp_opts options/SDFA-Net/train/sdfa_net-swint-m_192crop_kitti_stereo_stage2.yaml\
 --batch_size 12\
 --visual_freq 2000\
 --save_freq 5\
 --pretrained_path <path to .pth>
 ```

## Acknowledgment
Some of this repo come from [Monodepth2](https://github.com/nianticlabs/monodepth2), [AlignSeg](https://github.com/speedinghzl/AlignSeg) and [Swin-transformer](https://github.com/microsoft/Swin-Transformer).