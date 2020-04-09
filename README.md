# Semantic Segmentation of Nuclei in Digital Histology Images
This project includes SegNet, UNet, and DeepLabV3 for Semantic Segmentation of nuclei in digital histology images using Keras.

# Semantic Segmentation
"Semantic segmentation, or image segmentation, is the task of clustering parts of an image together which belong to the same object class. It is a form of pixel-level prediction because each pixel in an image is classified according to a category. Some example benchmarks for this task are Cityscapes, PASCAL VOC and ADE20K. Models are usually evaluated with the Mean Intersection-Over-Union (Mean IoU) and Pixel Accuracy metrics." -- [PapersWithCode](https://paperswithcode.com/task/semantic-segmentation).

# Dataset
**MoNuSeg dataset** (available [here](https://monuseg.grand-challenge.org/Data/)) contains multi organ tissue images with the ground truth segmentation masks for nucli. The dataset can also be downloaded from [this](https://drive.google.com/drive/folders/1hnHjxFb52BdhxkcV_N7MdWLdagzXHzmq?usp=sharing) Google Drive link.

Following is a sample training Tissue image with its correcsponding ground truth segmentation mask.
![Train0](https://github.com/zeeshanalipnhwr/Semantic-Segmentation-Keras/blob/master/Images/Train0.JPG)

The dataset contains *30* training, *14* testing samples.

**Tissue images shapes**: *1000x1000x3*

**Ground truth segmentation masks shapes**: *1000x1000x3*

# Data Preprocessing for training
1. The training data is split into training and validation sets with *75:25* ratio.
2. Ground truth segmentation masks are reshaped such that only one channal of each is kept.
3. Both the training and validation images and their correcsponding ground truth segmentation masks are split into *256x256* subimages using sliding-window approch with around *20%* overlap ratio using a custom data spliter.

# Data Preprocessing for testing
1. Training images and their correcsponding ground truth segmentation masks are padded with zeroes such that we get *1024x1024* size images. This is done to avoid model crash or to avoid getting output shape lesser than that of input.
2. After getting predictions on these padded test images, the paddings are removed from predictions and test data to get performance measures.

# Data Augmentation
Following three augmentations are applied on the training and validation images and their correcsponding ground truth segmentation masks using a custom data augmenter:
1. Rotations of angles *90*, *180*, *270* degrees.
2. Horizontal flips
3. Vertical flips

These augmentations were applied on *10%* of the training and *20%* of the validation data.

# Requirements
- python version 3.x
- tensorflow version 1.x

# Project Structure

    .
    ├── Colab Notebooks   # interactive notebooks containing sets of steps followed for training, testing, and predictions
    ├── Configs           # configuration files for respective models
    ├── Images            # screenshots or images needed for better presentation of README.md file
    ├── Models            # complete implementations of models of the project
        ├── DeepLabV3.py  # a complete implementation of a DeepLabV3 standard model
        ├── SegNet.py     # a complete implementation of a SegNet standard model
        └── UNet.py       # a complete implementation of a UNet standard model
    ├── Training Plots    # training and validation performance graphs for loss, accuracy, and f1 scores
    ├── Utils             # files that include custom functionalities needed for this project
    ├── README.md         # a complete overview of this directory
    └── train.py          # functions to train a model with simple or augmented data


# 1. SegNet
## Model Diagram
![SegNet Architecture](https://www.researchgate.net/profile/Vijay_Badrinarayanan/publication/283471087/figure/fig1/AS:391733042008065@1470407843299/An-illustration-of-the-SegNet-architecture-There-are-no-fully-connected-layers-and-hence.png)
## Model Summary
Go to respective colab notebook to view the detailed model summary.

# 2. UNet
## Model Diagram:
![UNet Architecture](https://vasanashwin.github.io/retrospect/images/unet.png)
## Model Summary
Go to respective colab notebook to view the detailed model summary.

# 3. DeepLabV3
## Model Diagram
![DeepLabV3 Architecture](https://miro.medium.com/max/1590/1*R7tiLxyeHYHMXTGJIanZiA.png)
![DeepLabV3 Architecture](https://media.arxiv-vanity.com/render-output/2143434/x1.png)
## Model Summary
Go to respective colab notebook to view the detailed model summary.

# Performance Measures

## Accuracy
It is defined as <img src="https://render.githubusercontent.com/render/math?math=accuracy = \frac{TP%2BTN}{TP%2BFP%2BTN%2BFN}">.

## F1 Score (or Dice Score)
F1 Score is defined as the harmonic mean of precision and recall as <img src="https://render.githubusercontent.com/render/math?math=F_1=\frac{2}{\frac{1}{precision}%2B\frac{1}{recall}}"> where <img src="https://render.githubusercontent.com/render/math?math=precision=\frac{TP}{TP%2BFP}"> and <img src="https://render.githubusercontent.com/render/math?math=recall=\frac{TP}{TP%2BFN}">. This is equivalent to Dice score coefficient which is defined as <img src="https://render.githubusercontent.com/render/math?math=DSC = \frac{2\times{TP}}{2\times{TP}%2BFP%2BFN}">.

# Quantitatvie Results
| Model | Accuracy | Precision | Recall | F1 Score (Dice Score) |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| SegNet | 0.9138 | **0.79** | 0.77 | 0.7799 |
| UNet | 0.9048 | 0.72 | **0.86** | 0.7802 |
| DeepLabV3 | 0.8606 | 0.61 | 0.80 | 0.6943 |
| SegNet_ResNet | 0.9042 | 0.74 | 0.81 | 0.7686 |
| UNet_ResNet | **0.9154** | 0.76 | 0.83 | **0.7942** |
| DeepLabV3_ResNet | - | - | - | - |

# Qualitative Results
## 1. SegNet
![SegNet](https://github.com/zeeshanalipnhwr/Semantic-Segmentation-Keras/blob/master/Images/SegNet_Qualitative_Results.JPG)

## 2. UNet
![UNet](https://github.com/zeeshanalipnhwr/Semantic-Segmentation-Keras/blob/master/Images/UNet_Qualitative_Results.JPG)

## 3. DeepLabV3
![DeepLabV3](https://github.com/zeeshanalipnhwr/Semantic-Segmentation-Keras/blob/master/Images/DeepLabV3_Qualitative_Results.JPG)

# Replication Instructions
Use the colab notebooks in the Colab Notebooks directory for training, testing, and predictions on different models.

# Pretrained Models
- SegNet_basic.model: https://drive.google.com/file/d/1-_FIvsHR_7hz0qnQe4lbU9YNHMNZ0GEF/view?usp=sharing
- UNet_basic.model: https://drive.google.com/file/d/13QR42aOatLQIN0G6bZ8z24TaW6LasEMw/view?usp=sharing
- DeepLabV3_basic.model: https://drive.google.com/file/d/1m1G-3huYC775H9R39WCC60WKaDvW0Ntt/view?usp=sharing

# Instructions to load a pretrained model
Either use the colab notebooks in the Colab Notebooks directory for predictions on respective models, or follow the following steps using your console.
## 1. Clone this repository to your current directory

    !git clone https://github.com/zeeshanalipnhwr/Semantic-Segmentation-Keras
    !mv Semantic-Segmentation-Keras Semantic_Segmentation_Keras

## 2. Create a model

```python
# import all the models and their respective configuration files
from Semantic_Segmentation_Keras.Models import SegNet, UNet, DeepLabV3
from Semantic_Segmentation_Keras.Configs import SegNet_Configs, UNet_Configs, DeepLabV3_Configs
```

```python
# create a model of your choice among the above availabe models
model = SegNet.SegNet(depth=SegNet_Configs.DEPTH).SegNet(input_shape=(None,None, 3))
#model = UNet.SegNet(depth=UNet_Configs.DEPTH).UNet(input_shape=(None, None, 3))
#model = DeepLabV3.DeepLabV3(depth=DeepLabV3_Configs.DEPTH).DeepLabV3(input_shape=(None, None, 3))
```

```python
# optionally view the created model summary
model.summary()
```

## 3. Load the respective pretrained-model weights

```python
model.load_weights("Model/segnet_basic.model")
#model.load_weights("Model/unet_basic.model")
#model.load_weights("Model/deeplabv3_basic.model")
```

## 4. Make prediction for a sample on the network

```python
from Semantic_Segmentation_Keras.Utils import display
import numpy as np
import cv2

print_statements = False # do you need to see the print results blow?

# load a sample image
image_path = "drive/My Drive/sample_tissue_image.tif"
sample_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
sample_image = np.array(sample_image, dtype="float") / 255.0
sample_image = np.expand_dims(sample_image, axis=0)
if print_statements: print ("sample_image shape:", sample_image.shape)

# in order to avoid a crash of model, make sure the image spatial dimentions are a multiple of 16
# the multiple factor 16 represented the ratio to which the actual image is reduced to by a model
padx, pady = 0, 0 # number zeros to add in x and y respectively
origional_sample_image_shape = sample_image.shape
if print_statements: print ("origional_sample_image_shape", origional_sample_image_shape)

if sample_image.shape[1]//16 != sample_image.shape[1]/16:
    padx = int(2**round(np.log2(sample_image.shape[1]))-sample_image.shape[1])//2
if sample_image.shape[2]//16 != sample_image.shape[2]/16:
    pady = int(2**round(np.log2(sample_image.shape[2]))-sample_image.shape[2])//2
if print_statements: print ("padx={}, pady={}".format(padx, pady))

sample_image_padded = np.zeros((1, sample_image.shape[1]+2*padx,
                                  sample_image.shape[2]+2*pady, 3))
if print_statements: print ("sample_image_padded shape", sample_image_padded.shape)

sample_image_padded[:, padx:padx+sample_image.shape[1],
                      pady:pady+sample_image.shape[2], :] = sample_image
sample_image = sample_image_padded
if print_statements: print ("sample_image shape:", sample_image.shape)

# make prediction for a sample on the network
prediction = model.predict(sample_image)
prediction = prediction.round(0)
if print_statements: print ("prediction shape:", prediction.shape)

# discard the predictions for the padded portion of sample_image
prediction = prediction[:, padx:padx+origional_sample_image_shape[1],
                          pady:pady+origional_sample_image_shape[2], :]
if print_statements: print ("prediction shape:", prediction.shape)

# remove the padded zeros from sample_image
sample_image = sample_image[:, padx:padx+origional_sample_image_shape[1],
                                pady:pady+origional_sample_image_shape[2], :]
if print_statements: print ("sample_image shape:", sample_image.shape)

# display the sample image along with its predicted mask
display.display_masked(sample_image[0], prediction[0], "Tissue Image", "Predicted Mask")
```

## License
This project is licensed under the terms of the [MIT License](https://github.com/zeeshanalipnhwr/Semantic-Segmentation-Keras/blob/master/LICENSE).

## Acknowledgements
The [./Utils/custom_layers.py](https://github.com/zeeshanalipnhwr/Semantic-Segmentation-Keras/blob/master/Utils/custom_layers.py) contains updated classes from [ykamikawa/tf-keras-SegNet/layers.py](https://github.com/ykamikawa/tf-keras-SegNet/blob/master/layers.py) file.

This project structure followed guidlines from [DongjunLee/hb-base](https://github.com/DongjunLee/hb-base) repository.

The [./.github/CONTRIBUTING.md](https://github.com/zeeshanalipnhwr/Semantic-Segmentation-Keras/blob/master/.github/CONTRIBUTING.md) was adapted from a basic template for [contributing guidelines](https://gist.github.com/briandk/3d2e8b3ec8daf5a27a62).

The [./.github/PULL_REQUEST_TEMPLATE.md](https://github.com/zeeshanalipnhwr/Semantic-Segmentation-Keras/blob/master/.github/PULL_REQUEST_TEMPLATE.md) is taken from [TalAter/open-source-templates](https://github.com/zeeshanalipnhwr/Semantic-Segmentation-Keras/blob/master/.github/PULL_REQUEST_TEMPLATE.md).
## Author
`Maintainer` [Zeeshan Ali](https://github.com/zeeshanalipnhwr) (zapt1860@gmail.com)
