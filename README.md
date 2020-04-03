# Semantic-Segmentation-Keras
I have implemented three different SegNet, UNet, and DeepLabV3 for Semantic Segmentation of nuclei in digital histology images using Keras.

# Semantic Segmentation
"Semantic segmentation, or image segmentation, is the task of clustering parts of an image together which belong to the same object class. It is a form of pixel-level prediction because each pixel in an image is classified according to a category. Some example benchmarks for this task are Cityscapes, PASCAL VOC and ADE20K. Models are usually evaluated with the Mean Intersection-Over-Union (Mean IoU) and Pixel Accuracy metrics." -- [PapersWithCode](https://paperswithcode.com/task/semantic-segmentation).

# Dataset
**Monuseg dataset** (available [here](https://monuseg.grand-challenge.org/Data/)) contains multi organ tissue images with the ground truth segmentation masks for nucli. The dataset can also be downloaded from [this](https://drive.google.com/drive/folders/1hnHjxFb52BdhxkcV_N7MdWLdagzXHzmq?usp=sharing) Google Drive link.

# Requirements
- python version 3.x
- tensorflow version 1.x

# Project Structure
.

├── dir_name/

└── main.py

# 1. SegNet
## Model Diagram
![SegNet Architecture](https://www.researchgate.net/profile/Vijay_Badrinarayanan/publication/283471087/figure/fig1/AS:391733042008065@1470407843299/An-illustration-of-the-SegNet-architecture-There-are-no-fully-connected-layers-and-hence.png)
## Model Summary
None

# 2. UNet
## Model Diagram:
![UNet Architecture](https://vasanashwin.github.io/retrospect/images/unet.png)
## Model Summary
None

# 3. DeepLabV3
## Model Diagram
![UNet Architecture](https://miro.medium.com/max/1590/1*R7tiLxyeHYHMXTGJIanZiA.png)
![UNet Architecture](https://media.arxiv-vanity.com/render-output/2143434/x1.png)
## Model Summary
None

# Performance Measures

## Accuracy
It is defined as <img src="https://render.githubusercontent.com/render/math?math=accuracy = \frac{TP%2BTN}{TP%2BFP%2BTN%2BFN}">.

## F1 Score (or Dice Score)
F1 Score is defined as the harmonic mean of precision and recall as <img src="https://render.githubusercontent.com/render/math?math=F_1=\frac{2}{\frac{1}{precision}%2B\frac{1}{recall}}$ where $precision=\frac{TP}{TP%2BFP}$ and $recall=\frac{TP}{TP%2BFN}">. This is equivalent to Dice score coefficient which is defined as <img src="https://render.githubusercontent.com/render/math?math=DSC = \frac{2\times{TP}}{2\times{TP}%2BFP%2BFN}">.

# Quantitatvie Results
None

# Qualitative Results
None

# Replication Instructions
- how to setup environment
- how to train the models using my code to replicate the results

# Load Pretrained Models
'''
    !git clone https://github.com/zeeshanalipnhwr/Semantic-Segmentation-Keras
    !mv Semantic-Segmentation-Keras Semantic_Segmentation_Keras

    %tensorflow_version 1.x
    %matplotlib inline

    # load the pretrained model weights for future use or deployment
    model.load_weights("drive/My Drive/segnet_basic_71_f1.model")

    # make prediction for a sample on the network
    prediction = model.predict(sample_image)
    prediction = prediction.round(0)

    display.display_masked(sample_image, prediction, "Tissue Image", "Predicted Mask")
'''
