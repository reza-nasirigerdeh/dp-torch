# dp-torch
**dp-torch** is simulation framework based on PyTorch and Opacus to compare the performance of popular convolutional neural networks such as **ResNets** and 
**DenseNets** using different normalization layers including **LayerNorm**, **GroupNorm**, and **NoNorm** as baseline.

**The code of KernelNorm, our proposed normalization layer, is coming soon.**

# Requirements
- Python +3.8
- PyTorch +1.11
- Opacus +1.1

# Installation
Clone the dp-torch repository:
```
git clone https://github.com/reza-nasirigerdeh/dp-torch
```
or 
```
git@github.com:reza-nasirigerdeh/dp-torch.git
```

Install the dependencies:
```
pip3 install -r requirements.txt  -f https://download.pytorch.org/whl/torch_stable.html
```

# Options
### Dataset
| Dataset              | option              |
|:---------------------|:--------------------|
| CIFAR-10             | --dataset cifar10   |
| Imagenette 160-pixel | --dataset imagenette|





### Model

#### ResNet-8
| Model                                               | option              |
|:----------------------------------------------------|:--------------------|
| ResNet8 with no normalization                       | --model resnet8_nn  |
| ResNet8 with layer normalization                    | --model resnet8_ln  |
| ResNet8 with group normalization (group size of 32) | --model resnet8_gn  |

#### PreactResNet18
| Model                                                      | option                        |
|:-----------------------------------------------------------|:------------------------------|
| PreactResNet18 with no normalization                       | --model preact_resnet18_nn    |
| PreactResNet18 with layer normalization                    | --model preact_resnet18_ln    |
| PreactResNet18 with group normalization (group size of 32) | --model preact_resnet18_gn    |        

### Differential privacy parameters
| Description               | option            |
|:--------------------------|:------------------|
| epsilon value (e.g. 8.0)  | --epsilon 8.0     |
| delta value (e.g. 1e-5)   | --delta 1e-5      |
| clipping value (e.g. 1.5) | --clipping 1.5    |

### Other
| Description                                                     | option            |
|:----------------------------------------------------------------|:------------------|
| batch size (e.g. 512)                                           | --batch-size 512  |
| number of train epochs (e.g. 100)                               | --epochs 100      |
| run number to have a separate result file for each run (e.g. 1) | --run 1           |
| activation function (e.g. Mish)                                 | --activation mish |

# Run
**Example1**: Train batch normalized version of VGG-6 on CIFAR-10 with cross-entropy loss function, SGD optimizer with learning rate of 0.025 and momentum of 0.9, and batch size of 32 for 100 epochs:

```
python3 simulate.py --dataset cifar10 --model vgg6_bn --optimizer sgd --momentum 0.9 \
                    --loss cross_entropy --learning-rate 0.025 --batch-size 32 --epochs 100
```

**Example2**: Train layer normalized version of ResNet-34 on imagenette-160-pixel with SGD optimizer, learning rate of 0.005 and momentum of 0.9, and batch size of 16 for 100 epochs. For preprocessing, apply random-resized-crop of shape 128x128 and random horizontal flipping to the train images, and resize test images to 160x160 first, and then, center crop them to 128x128, finally normalize them with the mean and std of imagenet:

```
python3 simulate.py --dataset imagenette_160px --random-hflip \
                    --random-resized-crop 128x128 --resize-test 160x160 --center-crop 128x128  \
                    --norm-mean 0.485,0.456,0.406  --norm-std 0.229,0.224,0.225 \
                    --model resnet34_ln --optimizer sgd --momentum 0.9 \
                    --learning-rate 0.005 --batch-size 16 --epochs 100
```

**Example3**: Train group normalized version of PreactResNet-18 with group size of 32 on CIFAR-100 with SGD optimizer, learning rate of 0.01, momentum of 0.9, and batch size of 16 for 200 epochs. For preprocessing, apply random horizontal flipping and cropping with padding 4x4 to the images. Decay the learning rate by factor of 10 at epochs 100 and 150.

```
python3 simulate.py --dataset cifar100 --random-crop 32x32-4x4 \
                    --random-hflip --model preact_resnet18_gn --group-size 32  \
                    --optimizer sgd --momentum 0.9  --learning-rate 0.01 \
                    --lr-scheduler multi_step --decay-epochs 100,150 --decay-multiplier 0.1 \
                    --batch-size 16 --epochs 200
```

## Citation
If you use **dp-torch** in your study, please cite the following paper: <br />
   ```
@inproceedings{
nasirigerdeh2023knconvnets-ppml,
title={Kernel Normalized Convolutional Networks for Privacy-Preserving Machine Learning},
author={Reza Nasirigerdeh and Javad Torkzadehmahani and Daniel Rueckert and Georgios Kaissis},
booktitle={First IEEE Conference on Secure and Trustworthy Machine Learning},
year={2023},
url={https://openreview.net/forum?id=pyfGjjDmrC}
}
   ```