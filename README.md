# DPTorch
**DPTorch** is simulation framework based on PyTorch and Opacus to compare the performance of different normalization layers including **LayerNorm**, **GroupNorm**, our proposed **KernelNorm** and **NoNorm** as baseline 
using  popular convolutional neural networks such as **ResNets** and **DenseNets** in **differentially private training**.


# Requirements
- Python +3.8
- PyTorch +1.11
- Opacus +1.1

# Installation
Clone the dp-torch repository:
```
git clone git@github.com:reza-nasirigerdeh/dp-torch.git
```

Install the dependencies:
```
pip3 install -r requirements.txt  -f https://download.pytorch.org/whl/torch_stable.html
```

# Options
### Dataset
| Dataset              | option                 |
|:---------------------|:-----------------------|
| CIFAR-10             | --dataset cifar10      |
| Imagenette 160-pixel | --dataset imagenette   |
| ImageNet32x32        | --dataset imagenet32x32|






### Model

#### ResNets
| Model                                                       | option              |
|:------------------------------------------------------------|:--------------------|
| ResNet-8 with no normalization                              | --model resnet8_nn  |
| ResNet-8 with layer normalization                           | --model resnet8_ln  |
| ResNet-8 with group normalization (group size of 32)        | --model resnet8_gn  |
| ResNet-8 with kernel normalization                          | --model resnet8_kn  |
| ResNet-18 with layer normalization                          | --model resnet18_ln |
| ResNet-18 with group normalization (number of groups of 32) | --model resnet18_gn |
| KNResNet-18  (kernel normalization)                         | --model knresnet18  |
| ResNet-13 with kernel normalization                         | --model resnet13_kn |

#### PreactResNet18
| Model                                                       | option                     |
|:------------------------------------------------------------|:---------------------------|
| PreactResNet-18 with no normalization                       | --model preact_resnet18_nn |
| PreactResNet-18 with layer normalization                    | --model preact_resnet18_ln |
| PreactResNet-18 with group normalization (group size of 32) | --model preact_resnet18_gn |        
| PreactResNet-18 with kernel normalization                   | --model preact_resnet18_kn |

#### DenseNet20x16
| Model                                                      | option                   |
|:-----------------------------------------------------------|:-------------------------|
| DenseNet-20x16 with no normalization                       | --model densenet20x16_nn |
| DenseNet-20x16 with layer normalization                    | --model densenet20x16_ln |
| DenseNet-20x16 with group normalization (group size of 32) | --model densenet20x16_gn |
| DenseNet-20x16 with kernel normalization                   | --model densenet20x16_kn |


### Differential privacy parameters
| Description                     | option           |
|:--------------------------------|:-----------------|
| epsilon value (e.g. 8.0)        | --epsilon 8.0    |
| delta value (e.g. 1e-5)         | --delta 1e-5     |
| clipping value (e.g. 1.5)       | --clipping 1.5   |
| privacy accountant (gdp or rdp) | --accountant rdp |


### Other
| Description                                                     | option            |
|:----------------------------------------------------------------|:------------------|
| batch size (e.g. 512)                                           | --batch-size 512  |
| number of train epochs (e.g. 100)                               | --epochs 100      |
| run number to have a separate result file for each run (e.g. 1) | --run 1           |
| activation function (e.g. Mish)                                 | --activation mish |

# Run
**Example1**: Train non-normalized version of ResNet-8 with Mish activation on CIFAR-10 with cross-entropy loss function, 
SGD optimizer with learning rate of 1.0 (divide at epoch 70), batch size of 512, epsilon of 8.0, delta value of 1e-5, and clipping value of 1.0 for 100 epochs (GDP privacy accountant):

```
python3 simulate.py --dataset cifar10 --model resnet8_nn --activation mish \
                    --epsilon 8.0 --delta 1e-5 --learning-rate 1.0 --batch-size 512 --clipping 1.0 \
                    --epochs 100 --decay-epochs 70 --accountant gdp --run 1
```

**Example2**: Train layer normalized version of DenseNet-20x16 with Mish activation on CIFAR-10 with cross-entropy loss function, 
SGD optimizer with learning rate of 1.5 (divide it by 2 at epochs 60 and 80), batch size of 1024, epsilon of 6.0, delta value of 1e-5, and clipping value of 1.5 for 80 epochs (RDP privacy accountant):

```
python3 simulate.py --dataset cifar10 --model densenet20x16_ln --activation mish \
                    --epsilon 6.0 --delta 1e-5 --learning-rate 1.5 --batch-size 1024 --clipping 1.5 \
                    --epochs 80 --decay-epochs 60,80 --accountant rdp --run 1
```

**Example3**: Train group normalized version of DenseNet-20x16 with ReLU activation on Imagenette with cross-entropy loss function, 
SGD optimizer with learning rate of 1.5 (divide it by 2 at epochs 50 and 70), batch size of 512, epsilon of 8.0, delta value of 1e-5, and clipping value of 1.0 for 100 epochs (GDP privacy accountant):

```
python3 simulate.py --dataset imagenette --model preact_resnet18_gn  --activation relu \
                    --epsilon 8.0 --delta 1e-5 --learning-rate 1.5 --batch-size 512 --clipping 1.0 \
                    --epochs 100 --decay-epochs 50,70 --accountant gdp --run 1
```

**Example4**: Train kernel normalized ResNet-13 with Mish activation on CIFAR-10 with cross-entropy loss function, SGD optimizer with learning rate of 2.0 
(divide it by 2 at epochs 50 and 75), batch size of 4096, epsilon of 6.0, delta value of 1e-5, and clipping value of 1.5 for 100 epochs (RDP privacy accountant):

```
python3 simulate.py --dataset cifar10 --model resnet13_kn  --activation mish \
                    --epsilon 6.0 --delta 1e-5 --learning-rate 2.0 --batch-size 4096 --clipping 1.5 \
                    --epochs 100 --decay-epochs 50,75 --accountant rdp --run 1
```

**Example5**: Train KNResNet-18 with Mish activation on ImageNet32x32 with cross-entropy loss function, SGD optimizer with learning rate of 4.0 
(divide it by 2 at epochs 70 and 90), batch size of 8192, epsilon of 8.0, delta value of 8e-7, and clipping value of 2.0 for 100 epochs (RDP privacy accountant):

```
python3 simulate.py --dataset imagenet32x32 --model knresnet18  --activation mish \
                    --epsilon 8.0 --delta 8e-7 --learning-rate 4.0 --batch-size 8192 --clipping 2.0 \
                    --epochs 100 --decay-epochs 70,90 --accountant rdp --run 1
```

## Citation
If you use **dp-torch** in your study, please cite the following papers: <br />
   ```
@inproceedings{
nasirigerdeh2023knconvnets-ppml,
title={Kernel Normalized Convolutional Networks for Privacy-Preserving Machine Learning},
author={Reza Nasirigerdeh and Javad Torkzadehmahani and Daniel Rueckert and Georgios Kaissis},
booktitle={First IEEE Conference on Secure and Trustworthy Machine Learning},
year={2023},
url={https://openreview.net/forum?id=pyfGjjDmrC}
}

@article{
    nasirigerdeh2024kernelnorm,
    title={Kernel Normalized Convolutional Networks},
    author={Reza Nasirigerdeh and Reihaneh Torkzadehmahani and Daniel Rueckert and Georgios Kaissis},
    journal={Transactions on Machine Learning Research},
    year={2024},
    url={https://openreview.net/forum?id=Uv3XVAEgG6},
}
   ```