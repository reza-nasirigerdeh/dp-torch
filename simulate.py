"""
    Copyright 2023 Reza NasiriGerdeh and Georgios Kaissis. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import random
import warnings
import math
import argparse
import torch
import os
import wget
import tarfile


from collections import defaultdict

from statistics import mean

import numpy as np
from functorch import grad, make_functional_with_buffers, vmap
from opacus.accountants.utils import get_noise_multiplier
from opacus.data_loader import DPDataLoader
from opacus.optimizers import DPOptimizer
from opacus.utils.batch_memory_manager import wrap_data_loader
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomCrop, RandomHorizontalFlip, ColorJitter, RandomVerticalFlip
from tqdm import tqdm

from models.resnet8 import resnet8_nn, resnet8_ln, resnet8_gn
from models.densenet import densenet20x16_nn, densenet20x16_ln, densenet20x16_gn
from models.preact_resnet import preact_resnet18_nn, preact_resnet18_ln, preact_resnet18_gn
from utils import ResultFile

warnings.filterwarnings("ignore")


def main():

    # ############ Simulation parameters ##################################
    parser = argparse.ArgumentParser(description="Simulate differentially private deep CNNs",
                                     usage=f"python simulate.py ")

    parser.add_argument("--dataset_name", "--dataset", type=str, help="dataset name", default="cifar10")

    parser.add_argument("--model_name", "--model", type=str, help="Model to be trained", default="resnet8_nn")

    parser.add_argument("--activation", "--activation", type=str, help="activation function", default="relu")

    parser.add_argument("--group_size", "--group-size", type=int, help="group size for GroupNorm", default=32)

    parser.add_argument("--learning_rate", "--learning-rate", "--lr", type=float, help="learning rate", default=1.0)

    parser.add_argument("--lr_scheduler", "--lr-scheduler", type=str, help="learning rate scheduler e.g. multi_step or cosine_annealing", default="multi_step")

    parser.add_argument("--momentum", "--momentum", type=float, help="SGD momentum", default=0.0)

    parser.add_argument("--batch_size", "--batch-size", type=int, help="batch size", default=512)

    parser.add_argument("--clipping", "--clipping", type=float, help="clipping value", default=1.0)

    parser.add_argument("--epsilon", "--epsilon", type=float, help="epsilon value", default=8.0)

    parser.add_argument("--delta", "--delta", type=float, help="delta value", default=1e-5)

    parser.add_argument("--aug-mul", "--aug_mul", type=int, help="augmentation multiplicity", default=1)

    parser.add_argument("--run_number", "--run", type=int,
                        help="the run number to have a separate result file for each run ", default=1)

    parser.add_argument("--epochs", "--epochs", type=int, help="number of epochs ", default=50)

    parser.add_argument("--decay_epochs", "--decay-epochs", type=str, help="comma separated list of decay epochs ", default='1000,1500')

    args = parser.parse_args()

    BS = args.batch_size
    LR = args.learning_rate
    max_physical_bs = 112
    epochs = args.epochs
    device = "cuda:0"
    reduction = "mean"
    deterministic = False
    target_epsilon = args.epsilon
    target_delta = args.delta
    clip_norm = args.clipping

    if deterministic:
        print("Determinism makes things slower.")
        SEED = 1302
        torch.backends.cudnn.deterministic = True
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        cpu_generator = torch.Generator(device="cpu").manual_seed(SEED)
        gpu_generator = torch.Generator(device=device).manual_seed(SEED)
    else:
        cpu_generator = torch.Generator(device="cpu")
        gpu_generator = torch.Generator(device=device)

    # ########## Dataset ######################
    if args.dataset_name == 'cifar10':
        num_classes = 10
        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2471, 0.2435, 0.2616)
        train_trans = Compose([ToTensor(), Normalize(cifar10_mean, cifar10_std)])
        test_trans = Compose([ToTensor(), Normalize(cifar10_mean, cifar10_std)])

        train_ds = CIFAR10(
            "./data",
            train=True,
            download=True,
            transform=train_trans)

        val_ds = CIFAR10(
            "./data",
            train=False,
            transform=test_trans)

    elif args.dataset_name == 'imagenette':
        # if imagenette-160px has not already been downloaded
        if not os.path.exists('./data/imagenette2-160'):
            # download imagenette-160px dataset
            print("Downloading the dataset ...")
            file_path = wget.download(url='https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz',
                                      out='./data')

            # extract tgz file
            print("Extracting the dataset ...")
            tar = tarfile.open(name=file_path, mode="r:gz")
            tar.extractall(path='./data')
            tar.close()

        num_classes = 10
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)
        train_trans = Compose([Resize((128, 128)),
                               ToTensor(),
                               Normalize(imagenet_mean, imagenet_std)
                               ])
        test_trans = Compose([Resize((128, 128)),
                              ToTensor(),
                              Normalize(imagenet_mean, imagenet_std)
                              ])

        train_ds = ImageFolder(root='./data/imagenette2-160/train', transform=train_trans)
        val_ds = ImageFolder(root='./data/imagenette2-160/val', transform=test_trans)

    else:
        print("Dataset name must be cifar10|imagenette!")
        exit()

    train_loader = DataLoader(
        train_ds,
        batch_size=BS,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=10,
        drop_last=True,
    )
    train_loader = DPDataLoader.from_data_loader(train_loader, generator=cpu_generator)

    val_loader = DataLoader(
        val_ds,
        batch_size=max_physical_bs,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=10,
    )

    # #################### Model #######################
    if args.activation == 'relu':
        activation = torch.nn.ReLU
    elif args.activation == 'mish':
        activation = torch.nn.Mish
    else:
        print("Activation function can be relu|mish")
        exit()

    if args.model_name == 'resnet8_nn':
        model = resnet8_nn(num_classes=num_classes, activation=activation)
    elif args.model_name == 'resnet8_ln':
        model = resnet8_ln(num_classes=num_classes, activation=activation)
    elif args.model_name == 'resnet8_gn':
        model = resnet8_gn(num_classes=num_classes, group_size=args.group_size, activation=activation)

    elif args.model_name == 'densenet20x16_nn':
        model = densenet20x16_nn(num_classes=num_classes, activation=activation)
    elif args.model_name == 'densenet20x16_ln':
        model = densenet20x16_ln(num_classes=num_classes, activation=activation)
    elif args.model_name == 'densenet20x16_gn':
        model = densenet20x16_gn(num_classes=num_classes, group_size=args.group_size, activation=activation)

    elif args.model_name == 'preact_resnet18_nn':
        model = preact_resnet18_nn(num_classes=num_classes, activation=activation)
    elif args.model_name == 'preact_resnet18_ln':
        model = preact_resnet18_ln(num_classes=num_classes, activation=activation)
    elif args.model_name == 'preact_resnet18_gn':
        model = preact_resnet18_gn(num_classes=num_classes, group_size=args.group_size, activation=activation)

    else:
        print("Model name can be resnet8_nn|resnet8_ln|resnet8_gn|resnet8_kn or"
              " densenet20x16_nn|densenet20x16_ln|densenet20x16_gn|densenet20x16_kn")

    model = model.to(device)

    fmodel, *_ = make_functional_with_buffers(model)
    criterion = torch.nn.CrossEntropyLoss(reduction=reduction)

    def compute_loss(params, buffers, sample, target, fmodel, loss_fn):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)
        predictions = fmodel(params, buffers, batch)
        loss = loss_fn(predictions, targets)
        return loss

    compute_per_sample_grads = vmap(
        grad(compute_loss), in_dims=(None, None, 0, 0, None, None), randomness="different"
    )

    optim = torch.optim.SGD(model.parameters(), lr=LR, momentum=args.momentum)

    noise = get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_rate=1 / len(train_loader),
        epochs=epochs,
        accountant="gdp",
    )

    print(model)
    print()

    print("############ Simulation params ############")
    print(f"Dataset: {args.dataset_name}")
    print(f"Model: {args.model_name}")
    print(f"Batch size: {BS}")
    print(f"Learning rate: {LR}")
    print(f"Epsilon: {target_epsilon}")
    print(f"Delta: {target_delta}")
    print(f"Clipping: {clip_norm}")
    print(f"Noise multiplier: {noise:.2f}")
    print(f"Decay epochs: {args.decay_epochs}")
    print(f"Epochs: {epochs}")
    print(f"Run: {args.run_number}")
    print("########################\n")

    optim = DPOptimizer(
        optim,
        noise_multiplier=noise,
        max_grad_norm=clip_norm,
        expected_batch_size=BS,
        loss_reduction=reduction,
        generator=gpu_generator,
    )

    decay_epochs = [int(epoch) for epoch in args.decay_epochs.split(',')]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=decay_epochs, gamma=0.5, verbose=True)

    train_loader = wrap_data_loader(
        data_loader=train_loader, max_batch_size=max_physical_bs, optimizer=optim
    )

    total_steps = len(train_loader.dataset) // BS * (math.ceil(BS / max_physical_bs))

    train_loss = defaultdict(list)
    val_loss = defaultdict(list)
    acc_train = defaultdict(list)
    acc_val = defaultdict(list)

    torch.set_grad_enabled(False)

    # ########### Result File ##################
    result_file_name = f"{args.dataset_name}-{args.model_name}-{args.activation}-e{args.epsilon}-lr{args.learning_rate}" \
                       f"-bs{args.batch_size}-c{args.clipping}-m{args.momentum}-{args.lr_scheduler}"
    result_file_name += (f'-gs{args.group_size}' if 'gn' in args.model_name else '')
    result_file_name += f'-run{args.run_number}'
    if args.aug_mul != 1:
        result_file_name += f'-aug{args.aug_mul}'

    result_file_name += '.csv'

    result_file = ResultFile(result_file_name=result_file_name)

    result_file.write_header('epoch,train_loss,test_loss,train_accuracy,test_accuracy')

    for epoch in range(epochs):

        model.train()
        for batch, target in tqdm(
            train_loader, leave=False, desc=f"Epoch {epoch+1}", total=total_steps
        ):
            batch = batch.to(device)
            target = target.to(device)
            y_pred = model(batch)
            loss = criterion(y_pred, target)
            train_loss[epoch].append(loss.item())
            acc = accuracy(y_pred, target).item()
            acc_train[epoch].append(acc)

            # augmentation multiplicity
            if args.aug_mul == 1:
                per_sample_grads = compute_per_sample_grads(
                    tuple(model.parameters()),
                    tuple(model.buffers()),
                    batch,
                    target,
                    fmodel,
                    criterion,
                )
            elif args.aug_mul == 3:
                # per-sample grads on the original batch
                orig_per_sample_grads = compute_per_sample_grads(
                    tuple(model.parameters()),
                    tuple(model.buffers()),
                    batch,
                    target,
                    fmodel,
                    criterion,
                )

                # per-sample grads on the horizontally flipped batch
                hflip_batch = RandomHorizontalFlip(p=1.0)(batch)
                hflip_per_sample_grads = compute_per_sample_grads(
                                tuple(model.parameters()),
                                tuple(model.buffers()),
                                hflip_batch,
                                target,
                                fmodel,
                                criterion,
                            )

                # per-sample grads on the randomly cropped  batch
                crop_batch = RandomCrop(size=(32, 32), padding=(4, 4))(batch)
                crop_per_sample_grads = compute_per_sample_grads(
                    tuple(model.parameters()),
                    tuple(model.buffers()),
                    crop_batch,
                    target,
                    fmodel,
                    criterion,
                )

                # take average over the gradients computed on the original, horizontally flipped, and randomly cropped batches
                per_sample_grads = [(grad1+grad2+grad3)/3 for grad1, grad2, grad3 in
                                     zip(orig_per_sample_grads, hflip_per_sample_grads, crop_per_sample_grads)]
            else:
                print("Augmentation multiplicity can be 1 or 3!")
                exit()

            for param, grad_sample in zip(model.parameters(), per_sample_grads):
                param.grad_sample = grad_sample
                param.grad = (
                    grad_sample.mean(0) if reduction == "mean" else grad_sample.sum(0)
                )
            optim.step()
            optim.zero_grad(True)

        model.eval()
        for batch, target in tqdm(val_loader, leave=False):
            batch = batch.to(device)
            target = target.to(device)
            y_pred = model(batch)
            loss = criterion(y_pred, target)
            val_loss[epoch].append(loss.item())
            acc = accuracy(y_pred, target).item()
            acc_val[epoch].append(acc)

        scheduler.step()

        print(f"Epoch {epoch+1}")
        print(f"\tTrain loss: {mean(train_loss[epoch]):.4f}")
        print(f"\tTest loss: {mean(val_loss[epoch]):.4f}")
        print(f"\tTrain accuracy: {mean(acc_train[epoch]):.4f}")
        print(f"\tTest accuracy: {mean(acc_val[epoch]):.4f}")

        result_file.write_result(epoch=epoch+1,
                                 result_list=[mean(train_loss[epoch]), mean(val_loss[epoch]),
                                              mean(acc_train[epoch]), mean(acc_val[epoch])])
    result_file.close()


if __name__ == "__main__":
    main()
