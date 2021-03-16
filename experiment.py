import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
import torchvision

import pandas as pd
import numpy as np
import os
from skimage import io
from PIL import Image

class MultipleCameraDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None, num_cam = 3):
        """
        Args:
            file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self._frame = np.asarray(pd.read_csv(root_dir + csv_file, delimiter=',', skipinitialspace=True, header = None))
        self.root_dir = root_dir
        self.transform = transform
        self.num_cam = 1

    def __len__(self):
        return len(self._frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        images = []
        for i in range(self.num_cam) :
            img_name = os.path.join(self.root_dir,
                                    self._frame[idx,i])
            image = Image.open(img_name)
            if self.transform:
                image = self.transform(image)
            images.append(image)
        images = torch.cat(images)
        #state = self._frame[idx, self.num_cam:]
        state = 0.0
        state = np.array([state])
        state = state.astype('float')
        return images, state

class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['batch_size']/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = self.params['batch_size']/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        return val_loss

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.sample_images()
        print("Images saved at the end of validation end.")
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels = test_label)
        vutils.save_image(recons.data[:4,:3,:,:],
                          f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                          f"recons_{self.logger.name}_{self.current_epoch}_cam1.png",
                          normalize=True,
                          nrow=12)

        vutils.save_image(test_input.data[:4,:3,:,:],
                           f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                           f"real_img_{self.logger.name}_{self.current_epoch}_cam1.png",
                           normalize=True,
                           nrow=12)
        if  recons.shape[1] >= 6 :
            vutils.save_image(recons.data[:4,3:6,:,:],
                              f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                              f"recons_{self.logger.name}_{self.current_epoch}_cam2.png",
                              normalize=True,
                              nrow=12)

            vutils.save_image(test_input.data[:4,3:6,:,:],
                               f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                               f"real_img_{self.logger.name}_{self.current_epoch}_cam2.png",
                               normalize=True,
                               nrow=12)
        if  recons.shape[1] >= 9 :
            vutils.save_image(recons.data[:4,6:9,:,:],
                              f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                              f"recons_{self.logger.name}_{self.current_epoch}_cam3.png",
                              normalize=True,
                              nrow=12)

            vutils.save_image(test_input.data[:4,6:9,:,:],
                               f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                               f"real_img_{self.logger.name}_{self.current_epoch}_cam3.png",
                               normalize=True,
                               nrow=12)

        try:
            #samples = self.model.sample(144,
            #                            self.curr_device,
            #                            labels = test_label)
            #vutils.save_image(samples.cpu().data[0,:,:,:],
            #                  f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
            #                  f"{self.logger.name}_{self.current_epoch}.png",
            #                  normalize=True,
            #                  nrow=12)
            pass
        except:
            pass


        del test_input, recons #, samples


    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    @data_loader
    def train_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            dataset = CelebA(root = self.params['data_path'],
                             split = "train",
                             transform=transform,
                             download=True)
        elif self.params['dataset'] == 'user':
            dataset = torchvision.datasets.ImageFolder(
                            root=self.params['data_path']+'train/', 
                            transform=transform)
        elif self.params['dataset'] == 'multicam' :
            dataset = MultipleCameraDataset(
                            csv_file = 'images.csv', 
                            root_dir = self.params['data_path'], 
                            transform = transform,
                            num_cam = self.params['num_cam'])
        else:
            raise ValueError('Undefined dataset type')

        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size= self.params['batch_size'],
                          shuffle = True,
                          drop_last=True)

    @data_loader
    def val_dataloader(self):
        print("Inside val dataloader ....")
        transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            dataset = CelebA(root = self.params['data_path'],
                            split = "test",
                            transform=transform,
                            download=True)
        elif self.params['dataset'] == 'user':
            dataset = torchvision.datasets.ImageFolder(
                            root=self.params['data_path']+'train/', 
                            transform=transform)
        elif self.params['dataset'] == 'multicam' :
            dataset = MultipleCameraDataset(
                            csv_file = 'images.csv', 
                            root_dir = self.params['data_path'], 
                            transform = transform,
                            num_cam = self.params['num_cam'])
        else:
            raise ValueError('Undefined dataset type')

        self.num_val_imgs = len(dataset)
        self.sample_dataloader =  DataLoader(dataset,
                            batch_size= self.params['batch_size'],
                            shuffle = True,
                            drop_last=True)
        # self.num_val_imgs = len(self.sample_dataloader)
        return self.sample_dataloader

    def data_transforms(self):

        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))

        if self.params['dataset'] == 'celeba':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])
        elif self.params['dataset'] == 'user' :
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=self.params['img_size']),
                # torchvision.transforms.ColorJitter(brightness=0.5, 
                #                                contrast=0.5,
                #                                saturation=0.5,
                #                                hue=0.05),
                torchvision.transforms.ToTensor()
                ])
        elif self.params['dataset'] == 'multicam' :
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=self.params['img_size']),
                # torchvision.transforms.ColorJitter(brightness=0.5, 
                #                                contrast=0.5,
                #                                saturation=0.5,
                #                                hue=0.05),
                torchvision.transforms.ToTensor()
                ])
        else:
            raise ValueError('Undefined dataset type')
        return transform

# train_dataset = torchvision.datasets.ImageFolder(root=input_folder, transform=transform)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers, drop_last=True)
	
