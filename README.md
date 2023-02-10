# dp-torch
**dp-torch** is simulation framework based on PyTorch and Opacus to compare the performance of popular convolutional neural networks such as **ResNets** and 
**DenseNets** using different normalization layers including **LayerNorm**, **GroupNorm**, and **NoNorm** as baseline in **differentially private training**.

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
git clone git@github.com:reza-nasirigerdeh/dp-torch.git
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

#### ResNet-8
| Model                                                     | option                   |
|:----------------------------------------------------------|:-------------------------|
| DenseNet20x16 with no normalization                       | --model densenet20x16_nn |
| DenseNet20x16 with layer normalization                    | --model densenet20x16_ln |
| DenseNet20x16 with group normalization (group size of 32) | --model densenet20x16_gn |

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
**Example1**: Train non-normalized version of ResNet-8 with Mish activation on CIFAR-10 with cross-entropy loss function, SGD optimizer with learning rate of 1.0, 
batch size of 512, epsilon of 8.0, delta value of 1e-5, and clipping value of 1.0 for 100 epochs:

```
python3 simulate.py --dataset cifar10 --model resnet8_nn --activation mish \
                    --epsilon 8.0 --delta 1e-5 --learning-rate 1.0 --batch-size 512 --clipping 1.0 \
                    --epochs 100 --run 1
```

**Example2**: Train layer normalized version of DenseNet20x16 with Mish activation on CIFAR-10 with cross-entropy loss function, SGD optimizer with learning rate of 1.0, 
batch size of 1024, epsilon of 6.0, delta value of 1e-5, and clipping value of 1.5 for 80 epochs:

```
python3 simulate.py --dataset cifar10 --model densenet20x16_ln --activation mish \
                    --epsilon 6.0 --delta 1e-5 --learning-rate 1.0 --batch-size 1024 --clipping 1.5 \
                    --epochs 80 --run 1
```

**Example3**: Train group normalized version of DenseNet20x16 with relu activation on Imagenette with cross-entropy loss function, SGD optimizer with learning rate of 1.0, 
batch size of 512, epsilon of 8.0, delta value of 1e-5, and clipping value of 1.0 for 100 epochs:

```
python3 simulate.py --dataset imagenette --model preact_resnet18_gn  --activation relu \
                    --epsilon 8.0 --delta 1e-5 --learning-rate 1.5 --batch-size 512 --clipping 1.0 \
                    --epochs 100 --run 1
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