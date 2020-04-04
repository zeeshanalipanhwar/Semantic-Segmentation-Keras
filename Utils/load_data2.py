from glob import glob
import numpy as np
import cv2

from sklearn.model_selection import train_test_split

def load_image(image_path, load_as="rgb"): #Load an image from a file path
    if load_as.lower() == "gray": color_scheme = cv2.COLOR_BGR2GRAY
    elif load_as.lower() == "hls": color_scheme = cv2.COLOR_BGR2HLS
    elif load_as.lower() == "hsv": color_scheme = cv2.COLOR_BGR2HSV
    elif load_as.lower() == "lab": color_scheme = cv2.COLOR_BGR2LAB
    elif load_as.lower() == "luv": color_scheme = cv2.COLOR_BGR2LUV
    elif load_as.lower() == "xyz": color_scheme = cv2.COLOR_BGR2XYZ
    elif load_as.lower() == "yuv": color_scheme = cv2.COLOR_BGR2YUV
    else: color_scheme = cv2.COLOR_BGR2RGB
    return cv2.cvtColor(cv2.imread(image_path), color_scheme)

def split_image_into_subimages(image, sub_image_size=(256, 256), overlap_ratio=.5):
    W_overlap = round(sub_image_size[0] * overlap_ratio)
    H_overlap = round(sub_image_size[1] * overlap_ratio)
    subimages = []
    for i in range(0, image.shape[0], W_overlap):
        x1 = i
        if x1+sub_image_size[0] <= image.shape[0]: x2 = x1+sub_image_size[0]
        else: x1, x2 = image.shape[0]-sub_image_size[0], image.shape[0]
        for j in range(0, image.shape[1], H_overlap):
            y1 = j
            if y1+sub_image_size[1] <= image.shape[1]: y2 = y1+sub_image_size[1]
            else: y1, y2 = image.shape[1]-sub_image_size[1], image.shape[1]
            subimages.append(image[x1:x2, y1:y2, :].copy())
    subimages = np.array(subimages)
    return subimages

def merge_subimages_into_image(subimages, image_size=(1000, 1000), overlap_ratio=.5):
    W_overlap = round(subimages.shape[0] * overlap_ratio)
    H_overlap = round(subimages.shape[1] * overlap_ratio)

    count = 0
    image = np.zeros((image_size[0], image_size[1], subimages[0].shape[-1]))
    for i in range(0, image_size[0], W_overlap):
        if x1+subimages.shape[0] <= image_size[0]: x2 = x1+subimages.shape[0]
        else: x1, x2 = image_size[0]-subimages.shape[0], image_size[0]
        for j in range(0, image_size[1], H_overlap):
            if y1+subimages.shape[1] <= image_size[1]: y2 = y1+subimages.shape[1]
            else: y1, y2 = image_size[1]-subimages.shape[1], image_size[1]
            image[x1:x2, y1:y2, :] = subimages[count].copy()
            count += 1
    return image

def load_training_data(files_path, load_as="rgb", sub_images_size=(256, 256)):
    X_train, Y_train = [], []

    for image_path in glob(files_path+'/Training/TissueImages/*'):
        image_name = image_path.split('/')[-1]
        tissue_image = load_image(files_path+'/Training/TissueImages/'+image_name, load_as)
        ground_truth = load_image(files_path+'/Training/GroundTruth/'+image_name.split('.')[0]+'_bin_mask.png', load_as)
        tissue_subimages = split_image_into_subimages(tissue_image, sub_image_size=(256, 256), overlap_ratio=.5)
        ground_truth_subimages = split_image_into_subimages(ground_truth, sub_image_size=(256, 256), overlap_ratio=.5)
        X_train.append(tissue_subimages)
        Y_train.append(ground_truth_subimages)

    X_train = np.array(X_train, dtype="float") / 255.0
    Y_train = np.array(Y_train, dtype="float") / 255.0

    X_train = X_train.reshape((X_train.shape[0]*X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4]))
    Y_train = Y_train.reshape((Y_train.shape[0]*Y_train.shape[1], Y_train.shape[2], Y_train.shape[3], Y_train.shape[4]))

    Y_train = Y_train[:, :, :, 0]
    Y_train = Y_train.reshape((Y_train.shape[0], Y_train.shape[1], Y_train.shape[2], 1))

    # round the float values in Y_train added to it by cv2.resize
    Y_train = Y_train.round(0)

    return X_train, Y_train

def load_testing_data(files_path, load_as="rgb", sub_images_size=(256, 256)):
    X_test, Y_test = [], []

    for image_path in glob(files_path+'/Test/TissueImages/*'):
        image_name = image_path.split('/')[-1]
        tissue_image = load_image(files_path+'/Test/TissueImages/'+image_name, load_as)
        ground_truth = load_image(files_path+'/Test/GroundTruth/'+image_name.split('.')[0]+'_bin_mask.png', load_as)
        tissue_subimages = split_image_into_subimages(tissue_image, sub_image_size=(256, 256), overlap_ratio=.5)
        ground_truth_subimages = split_image_into_subimages(ground_truth, sub_image_size=(256, 256), overlap_ratio=.5)
        X_test.append(tissue_subimages)
        Y_test.append(ground_truth_subimages)

    X_test = np.array(X_test, dtype="float") / 255.0
    Y_test = np.array(Y_test, dtype="float") / 255.0

    X_test = X_test.reshape((X_test.shape[0]*X_test.shape[1], X_test.shape[2], X_test.shape[3], X_test.shape[4]))
    Y_test = Y_test.reshape((Y_test.shape[0]*Y_test.shape[1], Y_test.shape[2], Y_test.shape[3], Y_test.shape[4]))

    Y_test = Y_test[:, :, :, 0]
    Y_test = Y_test.reshape((Y_test.shape[0], Y_test.shape[1], Y_test.shape[2], 1))

    # round the float values in Y_test added to it by cv2.resize
    Y_test = Y_test.round(0)

    return X_test, Y_test

def load_data(files_path, load_as="rgb", sub_images_size=(256, 256), validation_size=0.2):
    '''
    Aurguments:
        files_path: path to the directories that contains the expected Test and Training data directories
        load_as: ("rgb" by default)
        sub_images_size: ((256, 256) by default)
        validation_size: (0.2 by default) it is the ratio of validation data versus train data
    Returns:
    '''
    # loading the train data
    X_train, Y_train = load_training_data(files_path, load_as="rgb", sub_images_size=(256, 256))

    # split the training set to training and validation sets with the ratio 80:20 (by default) or valid_size
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=validation_size, random_state=42)

    X_test, Y_test = load_testing_data(files_path, load_as="rgb", sub_images_size=(256, 256))

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test
