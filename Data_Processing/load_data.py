from glob import glob
import numpy as np
import cv2

def load(image_path, load_as="rgb"): #Load an image from a file path
    if load_as.lower() = "gray": color_scheme = cv2.COLOR_BGR2GRAY
    elif load_as.lower() = "hls": color_scheme = cv2.COLOR_BGR2HLS
    elif load_as.lower() = "hsv": color_scheme = cv2.COLOR_BGR2HSV
    elif load_as.lower() = "lab": color_scheme = cv2.COLOR_BGR2LAB
    elif load_as.lower() = "luv": color_scheme = cv2.COLOR_BGR2LUV
    elif load_as.lower() = "xyz": color_scheme = cv2.COLOR_BGR2XYZ
    elif load_as.lower() = "yuv": color_scheme = cv2.COLOR_BGR2YUV
    else: color_scheme = cv2.COLOR_BGR2RGB
    return cv2.cvtColor(cv2.imread(image_path), color_scheme)

def load_training_data(files_path, load_as="rgb", resize_as=(1024, 1024)):
    X_train, Y_train = [], []

    for image_path in glob(files_path+'/Training/TissueImages/*'):
        image_name = image_path.split('/')[-1]
        tissue_image = load(files_path+'/Training/TissueImages/'+image_name, load_as)
        ground_truth = load(files_path+'/Training/GroundTruth/'+image_name.split('.')[0]+'_bin_mask.png', load_as)
        X_train.append(cv2.resize(tissue_image, resize_as))
        Y_train.append(cv2.resize(ground_truth, resize_as))

    X_train = np.array(X_train, dtype="float") / 255.0
    Y_train = np.array(Y_train, dtype="float") / 255.0

    Y_train = Y_train[:, :, :, 0]
    Y_train = Y_train.reshape((Y_train.shape[0], Y_train.shape[1], Y_train.shape[2], 1))

    # round the float values in Y_train added to it by cv2.resize
    Y_train = Y_train.round(0)
    
    return X_train, Y_train

def load_testing_data(files_path, load_as="rgb", resize_as=(1024, 1024)):
    X_test, Y_test = [], []

    for image_path in glob(files_path+'/Test/TissueImages/*'):
        image_name = image_path.split('/')[-1]
        tissue_image = load(files_path+'/Test/TissueImages/'+image_name, load_as)
        ground_truth = load(files_path+'/Test/GroundTruth/'+image_name.split('.')[0]+'_bin_mask.png', load_as)
        X_test.append(cv2.resize(tissue_image, resize_as))
        Y_test.append(cv2.resize(ground_truth, resize_as))

    X_test = np.array(X_test, dtype="float") / 255.0
    Y_test = np.array(Y_test, dtype="float") / 255.0

    Y_test = Y_test[:, :, :, 0]
    Y_test = Y_test.reshape((Y_test.shape[0], Y_test.shape[1], Y_test.shape[2], 1))

    # round the float values in Y_test added to it by cv2.resize
    Y_test = Y_test.round(0)

    return X_test, Y_test
