import datetime
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from src.utils import toOneHot, iou_accuracy

class DataGenerator(Dataset):
    """
    Custom Dataset for DataLoader
    """
    def __init__(self, data, img_size, mask_size):
        """
        Args:
            data (dict) :contains images and masks names
            img_size (integer): image size ``squared image``
            mask_size (integer): mask size ``squared mask``

        """
        self.data = data
        self.img_size = img_size
        self.mask_size = mask_size
        self.imgProcessor = None

        # images processing
        self.transform_image = transforms.Compose([
            transforms.Resize((self.img_size)),
            transforms.ToTensor()
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize((self.mask_size)),
            toOneHot
        ])

    def __len__(self):
        return len(self.data["img"])

    def __getitem__(self, index):
        img = self.transform_image(Image.open(self.data['img'][index]))
        mask = self.transform_mask(Image.open(self.data['mask'][index]))

        return img, mask

class Context(object):
    """
    Set of methods to train networks
    """

    def __init__(self):
        super(Context, self).__init__()
        self.train_data = {"img":[], "mask":[]}
        self.valid_data = {"img":[], "mask":[]}
        self.trainloader = None
        self.validloader = None
        self.img_size = None
        self.mask_size = None
        self.epochs = None
        self.batch_size = None
        self.DEVICE = "cpu"

    def fit(self, img_dir, mask_dir, img_size, mask_size, device, validation_split=.3, shuffle_data=True):
        """
        Prepare data for training
        """
        # set gpu or cpu device
        self.DEVICE=device
        
        # set dimensions
        self.img_size = img_size
        self.mask_size = mask_size

        # get images names and extensions
        img_names = {x[:-4]:x[-4:] for x in os.listdir(img_dir)}
        mask_names = {x[:-4]:x[-4:] for x in os.listdir(mask_dir)}

        # extract similar names in both directories
        names = sorted(list(set(img_names.keys()).intersection(mask_names.keys())))
        
        if shuffle_data:
            np.random.shuffle(names)

        # split train/validation
        split_indice = int(len(names)*(1-validation_split))

        for i, name in enumerate(names):
            if i < split_indice:
                self.train_data["img"].append(img_dir + '/' + name + img_names[name])
                self.train_data["mask"].append(mask_dir + '/' + name + mask_names[name])
            else:
                self.valid_data["img"].append(img_dir + '/' + name + img_names[name])
                self.valid_data["mask"].append(mask_dir + '/' + name + mask_names[name])

        # to numpy
        self.train_data["img"] = np.array(self.train_data["img"])
        self.train_data["mask"] = np.array(self.train_data["mask"])

        self.valid_data["img"] = np.array(self.valid_data["img"])
        self.valid_data["mask"] = np.array(self.valid_data["mask"])

        return self

    def set(self, nb_epochs=5, batch_size=1):
        """
        Set training parameters
        
        Args:
            nb_epochs (int): Number of epochs ``default 5``
            batch_size (int): batch sizeb ``default 1``
        """
        self.batch_size = batch_size
        self.epochs = nb_epochs

        # data loaders
        self.trainloader = DataLoader(DataGenerator(self.train_data, self.img_size, self.mask_size),
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=2,
                                      pin_memory=True)
        
        self.validloader = DataLoader(DataGenerator(self.valid_data, self.img_size, self.mask_size),
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=2,
                                      pin_memory=True)

    def train(self, model):
        """
        Train the model
        
        Args:
            model: pytorch Model
        """
        print("\nTraining...\n")
        
        nb_batches = math.ceil(len(self.trainloader.dataset) / self.batch_size)

        for epoch in range(self.epochs):

            running_loss = []
            iou_score = []
            start = time.time()
            ## print infos
            print(f"Epoch {epoch+1}/{self.epochs}")

            model.train()
            
            for batch, (features, targets) in enumerate(self.trainloader):

                model.optimizer.zero_grad()
                ## loss calculation
                loss = model.criterion(model(features.to(self.DEVICE)), targets.to(self.DEVICE))
                ## backward propagation
                loss.backward()
                ## weights optimization
                model.optimizer.step()

                # running loss
                running_loss.append(loss.item())

                ## IOU accuracy
                iou_score.append(iou_accuracy(model(features.to(self.DEVICE)), targets.to(self.DEVICE)))

                ## print train infos
                print(
                    f"\r Batch: {batch+1}/{nb_batches}",
                    f" - time: {str(datetime.timedelta(seconds = time.time() - start))}",
                    f" - T_loss: {np.mean(running_loss):.3f}",
                    f" - T_IOU: {np.mean(iou_score):.2f}", end = ''
                )
            ## save loss
            model.train_loss.append(np.mean(running_loss))
            ## save accuracy
            model.train_accuracy.append(np.mean(iou_score))

            ## evaluate
            self.evaluate(model, self.validloader, True)

    def evaluate(self, model, validloader, save_loss=False):
        """
        Evaluation part
        
        Args:
            model: pytorch model
            validloader: DatasetLoader for validation data
            save_loss(boolean): if 'True' save the validation loss
        """
        running_loss = []
        iou_score = []

        model.eval()
        for i, (features, targets) in enumerate(validloader):
            ## compute loss for prediction
            loss = model.criterion(model(features.to(self.DEVICE)), targets.to(self.DEVICE))

            ## save loss
            running_loss.append(loss.item())

            ## IOU accuracy
            iou_score.append(iou_accuracy(model(features.to(self.DEVICE)), targets.to(self.DEVICE)))
        # save accuracy
        model.valid_accuracy.append(np.mean(iou_score))

        if save_loss:
            model.valid_loss.append(np.mean(running_loss))

        ## print infos
        print(
            f" - V_loss: {np.mean(running_loss):.3f}",
            f" - V_IOU:{np.mean(model.valid_accuracy):.2f}"
        )
    def predict(self, model, img_path, mask_path=None, plot=False):
        """
        Inference
        
        Args:
            model: Unet model
            img_path (String): image path
            mask_path (String): optional, mask path
            plot: (boolean): optional, if 'True' plot the prediction
        outputs:
            yhat: predicted segmentation
        """
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # open image
        img = Image.open(img_path)
        # transform image
        img = transforms.Resize((256))(img)
        img = transforms.ToTensor()(img)
        img = img.unsqueeze(0)
        # predict
        yhat = model(img.to(DEVICE))
        yhat = yhat[0].permute(1,2,0)

        if plot:
            #plots
            fig, axes = plt.subplots(2,6, figsize=(18,6))
            channel = 0
            for i in range( axes.shape[0]):
                for j in range(1, axes.shape[1]):
                    axes[i,j].imshow((yhat).cpu().float().detach().numpy()[:,:,channel],cmap = 'viridis')
                    channel+=1
            axes[0,0].imshow(img[0].permute(1,2,0))

            # mask
            if mask_path is not None:
                mask = Image.open(mask_path)
                #plot ground truth
                axes[1,0].imshow(mask)

            plt.show()

        return yhat.cpu().float().detach().numpy()
