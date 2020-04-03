from Semantic_Segmentation_Keras.Configs import SegNet_Configs
from Semantic_Segmentation_Keras.Utils import custom_matrics

def train_model(model, X_train, Y_train, validation_data, save=False, save_as=None):
    # compile the model
    model.compile(loss="binary_crossentropy", optimizer=SegNet_Configs.OPTIMIZER,
                  metrics=["accuracy", custom_matrics.f1_score])
    
    # train the network without using augmented data
    H = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid),
                  batch_size=SegNet_Configs.BATCHSIZE,
                  epochs=SegNet_Configs.EPOCHS)
    
    if save: # save the model for future use or deployment
        if not save_as: raise ValueError("'save_as' must not be 'None' when 'save' is 'True'!")
        else: model.save("drive/My Drive/segnet_basic.model")
    return H # training and validation history
