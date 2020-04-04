from Semantic_Segmentation_Keras.Configs import SegNet_Configs
from Semantic_Segmentation_Keras.Utils import custom_matrics

def train_model(model, X_train, Y_train, validation_data, optimizer, batch_size, epochs,
                save=False, save_to=None, save_as=None):
    if save:
        if not save_to or not save_as:
            raise ValueError("'save_to' and 'save_as' must not be 'None' when 'save' is 'True'!")
    # compile the model
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy", custom_matrics.f1_score])

    # train the network without using augmented data
    H = model.fit(X_train, Y_train, validation_data=validation_data, batch_size=batch_size, epochs=epochs)

    if save: # save the model for future use or deployment
        model.save("{}{}.model".format(save_to, save_as))

    return H # training and validation history

def train_model_on_augmented_data(model, train_data, validation_data, optimizer, batch_size, epochs,
                                  save=False, save_to=None, save_as=None):
    if save:
        if not save_to or not save_as:
            raise ValueError("'save_to' and 'save_as' must not be 'None' when 'save' is 'True'!")
    # compile the model
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy", custom_matrics.f1_score])

    # train the network using augmented data
    H = model.fit_generator(train_data, validation_data=validation_data, epochs=epochs)

    if save: # save the model for future use or deployment
        model.save("{}{}.model".format(save_to, save_as))

    return H # training and validation history
