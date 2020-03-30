import numpy as np
import cv2

def load(image_path): #Load an image from a file path
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

def load_training_data(files_path):
    X_train, Y_train = [], []

    for image_path in glob(files_path+'/Training/TissueImages/*'):
        image_name = image_path.split('/')[-1]
        X_train.append(cv2.resize(load(files_path+'/Training/TissueImages/'+image_name), (1024, 1024)))
        Y_train.append(cv2.resize(load(files_path+'/Training/GroundTruth/'+image_name.split('.')[0]+'_bin_mask.png'), (1024, 1024)))

    X_train = np.array(X_train, dtype="float") / 255.0
    Y_train = np.array(Y_train, dtype="float") / 255.0

    Y_train = Y_train[:,:,:,0]
    Y_train = Y_train.reshape((Y_train.shape[0], Y_train.shape[1], Y_train.shape[2], 1))

    # round the float values in Y_train added to it by cv2.resize
    Y_train = Y_train.round(0)
    
    return X_train, Y_train

def load_testing_data(files_path):
    X_test, Y_test = [], []

    for image_path in glob(files_path+'/Test/TissueImages/*'):
        image_name = image_path.split('/')[-1]
        X_test.append(cv2.resize(load(files_path+'/Test/TissueImages/'+image_name), (1024, 1024)))
        Y_test.append(cv2.resize(load(files_path+'/Test/GroundTruth/'+image_name.split('.')[0]+'_bin_mask.png'), (1024, 1024)))

    X_test = np.array(X_test, dtype="float") / 255.0
    Y_test = np.array(Y_test, dtype="float") / 255.0

    Y_test = Y_test[:,:,:,0]
    Y_test = Y_test.reshape((Y_test.shape[0], Y_test.shape[1], Y_test.shape[2], 1))

    # round the float values in Y_test added to it by cv2.resize
    Y_test = Y_test.round(0)

    return X_test, Y_test
