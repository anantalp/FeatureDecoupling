## *Self-Supervised Representation Learning by Rotation Feature Decoupling*

### Abstract:
We introduce a self-supervised learning method that focuses on beneficial properties of representation and their abilities in generalizing to real-world tasks. The method incorporates rotation invariance into the feature learning framework, one of many good and well-studied properties of visual representation, which is rarely appreciated or exploited by previous deep convolutional neural network based self-supervised representation learning methods. Specifically, our model learns a split representation that contains both rotation related and unrelated parts. We train neural networks by jointly predicting image rotations and discriminating individual instances. In particular, our model decouples the rotation discrimination from instance discrimination, which allows us to improve the rotation prediction by mitigating the influence of rotation label noise, as well as discriminate instances without regard to image rotations. The resulting feature has a better generalization ability for more various tasks. Experimental results show that our model outperforms current state-of-the-art methods on standard self-supervised feature learning benchmarks.

### Illustration

<img src="https://raw.githubusercontent.com/philiptheother/FeatureDecoupling/master/_imgs/figure.png">


### Experiments

* In order to train a FeatureDecoupling model in an unsupervised way with AlexNet-like architecture on the ImageNet training images and then evaluate linear classifiers (for ImageNet and Places205) as well as non-linear classifiers (for ImageNet) on top of the learned features please visit the [pytorch_feature_decoupling](https://github.com/philiptheother/FeatureDecoupling/tree/master/pytorch_feature_decoupling) folder.

* For PASCAL VOC 2007 detection task please visit the [caffe_voc_detection](https://github.com/philiptheother/FeatureDecoupling/tree/master/caffe_voc_detection) folder.

### Download the already trained FeatureDecoupling model

* In order to download the FeatureDecoupling model (with AlexNet architecture) trained on the ImageNet training set, go to: [ImageNet_Decoupling_AlexNet](https://mega.nz/#!Wmh3WIDZ!e2TgkXEsMMpZNZvb1Tp8HsdBfeZOA3WKn5g0AkXEwAA). Note that:   
  1. The model is saved in pytorch format.
  2. It expects RGB images that their pixel values are normalized with the following mean RGB values `mean_rgb = [0.485, 0.456, 0.406]` and std RGB values `std_rgb = [0.229, 0.224, 0.225]`. Prior to normalization the range of the image values must be [0.0, 1.0].

 * In order to download the FeatureDecoupling model (with AlexNet architecture) trained on the ImageNet training set and convered in caffe format for PASCAL VOC 2007 classification and detection, go to: [ImageNet_Decoupling_AlexNet_caffe_cls](https://mega.nz/#!e65D3CLZ!jUvWfBt3NBcjZSI90X5mKKe-OHSswN9nWo_aPo1YCOQ). Note that:   
   1. The model is saved in caffe format.
   2. It expects RGB images that their pixel values are normalized with the following mean RGB values `mean_rgb = [0.485, 0.456, 0.406]` and std RGB values `std_rgb = [0.229, 0.224, 0.225]`. Prior to normalization the range of the image values must be [0.0, 1.0].
   3. The weights of the model are rescaled with the approach of [Kr&auml;henb&uuml;hl et al, ICLR 2016](https://github.com/philkr/magic_init).

### Citation

```
@InProceedings{Feng_2019_CVPR,
author = {Feng, Zeyu and Xu, Chang and Tao, Dacheng},
title = {Self-Supervised Representation Learning by Rotation Feature Decoupling},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```
