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
F1 Score is defined as the harmonic mean of precision and recall as <img src="https://render.githubusercontent.com/render/math?math=F_1=\frac{2}{\frac{1}{precision}%2B\frac{1}{recall}}"> where <img src="https://render.githubusercontent.com/render/math?math=precision=\frac{TP}{TP%2BFP}"> and <img src="https://render.githubusercontent.com/render/math?math=recall=\frac{TP}{TP%2BFN}">. This is equivalent to Dice score coefficient which is defined as <img src="https://render.githubusercontent.com/render/math?math=DSC = \frac{2\times{TP}}{2\times{TP}%2BFP%2BFN}">.

# Quantitatvie Results
None

# Qualitative Results
None

# Replication Instructions
Follow the colab notebooks for training-and-testing in the Colab Notebooks directory for respective models.

# Load Pretrained Models
Either follow the colab notebooks for predictions using the pretrained models in the Colab Notebooks directory for respective models, or follow the following script using your console.

## 1. Clone this repository to your current directory

    !git clone https://github.com/zeeshanalipnhwr/Semantic-Segmentation-Keras
    !mv Semantic-Segmentation-Keras Semantic_Segmentation_Keras

## 2. Create a model

    from Semantic_Segmentation_Keras.Models import SegNet
    from Semantic_Segmentation_Keras.Configs import SegNet_Configs

    # create the model
    model = SegNet.SegNet(depth=SegNet_Configs.DEPTH).SegNet(input_shape=(SegNet_Configs.RESHAPE[0],
                                                                          SegNet_Configs.RESHAPE[1], 3))
    # following two models are NOT IMPLEMENTED YET
    #model = UNet.SegNet(depth=UNet_Configs.DEPTH).UNet(input_shape=(UNet_Configs.RESHAPE[0],
                                                                     UNet_Configs.RESHAPE[1], 3))
    #model = DeepLabV3.DeepLabV3(depth=DeepLabV3_Configs.DEPTH).DeepLabV3(input_shape=(DeepLabV3_Configs.RESHAPE[0],
                                                                                       DeepLabV3_Configs.RESHAPE[1], 3))
    model.summary()

## 3. Load the pretrained model weights

    model.load_weights("drive/My Drive/segnet_basic_72_f1.model")

    # load a sample image
    image_path = None
    sample_image = data_loading.load_image(image_path)

## 4. Make prediction for a sample on the network

    from Semantic_Segmentation_Keras.Utils import display, load_data
    import numpy as np
    import cv2

    # load a sample image
    image_path = "drive/My Drive/sample_tissue_image.tif"
    sample_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    sample_image = cv2.resize(sample_image, SegNet_Configs.RESHAPE)
    sample_image = np.array(sample_image, dtype="float") / 255.0
    sample_image = np.expand_dims(sample_image, axis=0)

    # make prediction for a sample on the network
    prediction = model.predict(sample_image)
    prediction = prediction.round(0)

    # display the sample image along with its predicted mask
    display.display_masked(sample_image[0], prediction[0], "Tissue Image", "Predicted Mask")
