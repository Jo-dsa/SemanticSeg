
# Face Parsing

Implementation of a one-stage Face Parsing model using Unet architecture.<br/>
The face is divided into 10 classes (background, eyes, nose, lips, ear, hair, teeth, eyebrows, general face, beard).<br/>
This results in regions of interest that are difficult to segment. We address this problem by implementing different loss functions:<br/>

-  Tversky focal loss
-  Weighted Cross entropy
-  α-balanced focal loss

The latter leads to a better accuracy of 0.90 (IOU-train) and 0.72 (IOU-test) for the non fine-tuned model, with a Unet-16 driven on 1 gpu.

## Implementation

Language version : Python 3.7.6 <br />
Operating System : MacOS Catalina 10.15.4 <br />
Framework: Pytorch

## Results

We can observe the probability map for each channel. <br />
It allows us to estimate how much a pixel belongs to a specific part of the face. <br />
<img src="https://dl.dropboxusercontent.com/s/z6loxo2ttt9hnbl/heatmap.png?dl=0" height="300">

Segmentation sample: <br />
<img src="https://dl.dropboxusercontent.com/s/gk0l5dpw4k4txqc/inference.png?dl=0" height="300">
  

### Getting Started
Install dependencies: <br />
```python
# use a virtual-env to keep your packages unchanged
>> pip install -r requirements.txt
```
Train the model: <br />
```python
# Train the model
>> ./main.py
```
Inference: <br />
```python
from src.model import Unet
from src.framework import Context

img = "./img_relative_path.png"

# load model
model = torch.load("my_model.pt", map_location=torch.device('cpu'))
# create context
ctx = Context()
# predict
yhat = ctx.predict(model, img, plot=True)
```

## References

- [Mut1ny](https://www.mut1ny.com/face-headsegmentation-dataset) Face/head dataset
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)
Olaf Ronneberger, Philipp Fischer, and Thomas Brox.
Tech report, arXiv, May 2015.
- [A novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation](https://arxiv.org/pdf/1810.07842.pdf)
Abraham, Nabila and Khan, Naimul Mefraz.
Tech report, arXiv, Oct 2018.
- [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf)
Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár.
Tech report, arXiv, Feb 2018.
