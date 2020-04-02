# Semantic-Segmentation-Keras
I have implemented three different SegNet, UNet, and DeepLabV3 for Semantic Segmentation of nuclei in digital histology images using Keras.

# Semantic Segmentation:
"Semantic segmentation, or image segmentation, is the task of clustering parts of an image together which belong to the same object class. It is a form of pixel-level prediction because each pixel in an image is classified according to a category.

Some example benchmarks for this task are Cityscapes, PASCAL VOC and ADE20K. Models are usually evaluated with the Mean Intersection-Over-Union (Mean IoU) and Pixel Accuracy metrics." -- [source](https://paperswithcode.com/task/semantic-segmentation)

# Dataset:
**Monuseg dataset** (available [here](https://monuseg.grand-challenge.org/Data/)) contains multi organ tissue images with the ground truth segmentation masks for nucli. The dataset can also be downloaded from [this](https://drive.google.com/drive/folders/1hnHjxFb52BdhxkcV_N7MdWLdagzXHzmq?usp=sharing) Google Drive link.

# 1. SegNet:

## Model Diagram:
![SegNet Architecture](https://www.researchgate.net/profile/Vijay_Badrinarayanan/publication/283471087/figure/fig1/AS:391733042008065@1470407843299/An-illustration-of-the-SegNet-architecture-There-are-no-fully-connected-layers-and-hence.png)

# 2. UNet:

## Model Diagram:

# 3. DeepLabV3:

## Model Diagram:

# Performance Measures

## Accuracy
It is defined as $accuracy = \frac{TP+TN}{TP+FP+TN+FN}$.

## F1 Score (or Dice Score)
F1 Score is defined as the harmonic mean of precision and recall as $F_1=\frac{2}{\frac{1}{precision}+\frac{1}{recall}}$ where $precision=\frac{TP}{TP+FP}$ and $recall=\frac{TP}{TP+FN}$. This is equivalent to Dice score coefficient which is defined as $DSC = \frac{2\times{TP}}{2\times{TP}+FP+FN}$.

# Results:
None
