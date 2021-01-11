# Self-Supervised Autogenous Learning

This repository contains an implementation of the ICPR 2020 paper ["Contextual Classification using Self-Supervised Auxiliary Models for Image Classification"](https://spalaciob.github.io/ssal.html) for reproducing key experiments presented there. 

### Running the Code

Depending on whether the Pytorch or Keras code is used, the corresponding `requirements.txt` must be fulfilled. Results from ImageNet have been implemented in Pytorch and those from CIFAR100 have been implemented in Keras. 

#### Keras (CIFAR100)

By running the `run_all.sh` script, the key experiments for CIFAR100 are reproduced. The Resnet50, WRN28-10 and Densenet190-40 networks are trained with and without SSAL from scratch. Similarly to the script, a single experiment of those can be picked out as well. If Keras does not find CIFAR100, it downloads it. With the additional flag `--pretrained`, pretrained models can be loaded from the `models` folder. If the corresponding models are not found, they are downloaded.

##### Results (from the paper)

- Classification accuracy for multiple high-performance architectures on CIFAR100. The following table contains the original reported accuracy (org), the accuracy of the original model as reimplemented by us (ours), training with a SSAL objective but issuing predictions without the auxiliary output (+TR) and by using the joint prediction (+JP) i.e., both training and predicting with the SSAL branch. Adding the SSAL objective (+JP) consistently yields higher performance.

|                 | Val. Accuracy |      |      |               | Params (M) |      |
|-----------------|:-------------:|:----:|------|---------------|:----------:|------|
|                 | org           | ours | +TR  | +JP           | org        | SSAL |
| Resnet50        | -             | 78.9 | 79.7 | 80.6          | 23.8       | 28.9 |
| SE-WRN~16-8     | 80.9          | 79.0 | 79.0 | 80.2          | 11.1       | 14.9 |
| WRN~28-10       | 80.8          | 80.1 | 80.6 | __81.0__      | 36.6       | 38.2 |
| DenseNet~190-40 | 82.8          | 81.1 | 81.8 | __83.2__      | 26.1       | 38.3 |

#### Pytorch (ImageNet)

_Note: The reported values for ImageNet have been obtained training with internal libraries for more efficient data loading and augmentation. We are currently recomputing results without those libraries. Once we have them, we will update the values here in the repository. The offered pretrained models have been trained on our original code._

The `run_all.sh` has to be run with the Folder of the ImageNet dataset specified in the `--datadir` option, e.g. `bash run_all.sh --datadir "../data/ImageNet"`. The folder needs to have the structure expected by the `ImageNet` loader of Pytorch. The script then trains a Resnet50 on ImageNet - first without and then with SSAL. Similarly to the Keras version, a single experiment can be executed by giving just one of the experiment identifiers. With the additional flag `--pretrained`, pretrained models can be loaded from from the `models` folder. If the corresponding models are not found, they are downloaded.

##### Results (from the paper)
- Accuracy for SSAL Resnet50 on Imagenet. Experiments are run 3 times. Standard deviation is 0.1p.p. (except Top-1 on +LC which was 0.2p.p.). +TR means that the model has been trained using a SSAL objective but the evaluation only uses the original classifier. +JP means that, in addition to training with a SSAL branch, joint predictions are issued combining the output of the original classifier and the SSAL branch. 

|       | baseline   | +TR  | +JP           |
|-------|------------|------|---------------|
| Top-1 | 75.5       | 76.4 | __76.9__      |
| Top-5 | 92.7       | 93.3 | __93.7__      |
