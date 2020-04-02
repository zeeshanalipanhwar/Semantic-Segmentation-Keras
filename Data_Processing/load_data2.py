from sklearn.model_selection import train_test_split

def load_data(files_path, load_as="rgb", resize_as=(1024, 1024), valid_size=0.2)
    '''
    Aurguments:
        files_path: path to the directories that contains the expected Test and Training data directories
        load_as="rgb"
        resize_as=(1024, 1024)
        valid_size=0.2
    Returns:
    '''
    print ("loading the data..")
    X_train, Y_train = load_training_data(files_path, load_as="rgb", resize_as=(1024, 1024))
    print ("Train data of size: {} loaded!".format(X_train.shape))

    #spliting the training set to training and validation sets with the ratio 80:20 (by default) or valid_size
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=valid_size, random_state=42)
    print ("Validation data of size: {} loaded!".format(X_test.shape))

    X_test, Y_test = load_testing_data(files_path, load_as="rgb", resize_as=(1024, 1024))
    print ("Test data of size: {} loaded!".format(X_test.shape))

    return X_train, X_valid, Y_train, Y_valid, X_test, Y_test
