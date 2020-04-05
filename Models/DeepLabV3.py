from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers import Input, concatenate, UpSampling2D
from keras import backend as K

class DeepLabV3:
    def __init__(self):
        pass

    def atrous_spatial_pyramid_pooling(self, input_layer, depth, dropout):
        conv11_layer = Conv2D(depth, (1, 1), activation='relu', padding="same")(input_layer)
        conv11_layer = Dropout(dropout)(conv11_layer)
        
        atrous_conv1 = Conv2D(depth, (3, 3), dilation_rate=(6 ,6 ), activation='relu', padding="same")(input_layer)
        atrous_conv1 = Dropout(dropout)(atrous_conv1)
        atrous_conv2 = Conv2D(depth, (3, 3), dilation_rate=(12,12), activation='relu', padding="same")(input_layer)
        atrous_conv2 = Dropout(dropout)(atrous_conv2)
        atrous_conv3 = Conv2D(depth, (3, 3), dilation_rate=(18,18), activation='relu', padding="same")(input_layer)
        atrous_conv3 = Dropout(dropout)(atrous_conv3)

        return conv11_layer, atrous_conv1, atrous_conv2, atrous_conv3

    def DeepLabV3(self, input_shape, depth):
        input_layer = Input(shape=input_shape)

        # Block one that reduces the input shape by 4
        output_layer = Conv2D(depth, (3, 3), activation='relu', padding="same")(input_layer)
        output_layer = Conv2D(depth, (3, 3), activation='relu', padding="same")(output_layer)
        output_layer = MaxPooling2D(pool_size=(2, 2))(output_layer)
        output_layer = Dropout(dropout)(output_layer)
        
        # Block two that reduces the input shape by 8
        output_layer = Conv2D(depth*2, (3, 3), activation='relu', padding="same")(output_layer)
        output_layer = Conv2D(depth*2, (3, 3), activation='relu', padding="same")(output_layer)
        output_layer = MaxPooling2D(pool_size=(2, 2))(output_layer)
        output_layer = Dropout(dropout)(output_layer)
        
        # Block three that reduces the input shape by 16
        output_layer = Conv2D(depth*4, (3, 3), activation='relu', padding="same")(output_layer)
        output_layer = Conv2D(depth*4, (3, 3), activation='relu', padding="same")(output_layer)
        output_layer = MaxPooling2D(pool_size=(2, 2))(output_layer)
        output_layer = Dropout(dropout)(output_layer)
        
        # Block four that retains 16 using Atrous Convolution
        output_layer = Conv2D(depth*8, (3, 3), dilation_rate=(2,2), activation='relu', padding="same")(output_layer)
        output_layer = Conv2D(depth*8, (3, 3), dilation_rate=(2,2), activation='relu', padding="same")(output_layer)
        output_layer = MaxPooling2D(pool_size=(2, 2))(output_layer)
        output_layer = Dropout(dropout)(output_layer)
        
        # Block five of atrous spatial pyramid pooling and image max pooling
        conv11_layer, atrous_conv1, atrous_conv2, atrous_conv3 = self.atrous_spatial_pyramid_pooling(output_layer, depth*8, dropout)
        maxpooled_in = MaxPooling2D(pool_size=(16, 16))(input_layer)

        # Block six of concatination and then a 1x1 conv on the concatenated output
        concatenated = concatenate([conv11_layer, atrous_conv1, atrous_conv2, atrous_conv3, maxpooled_in])
        output_layer = Conv2D(depth, (1, 1), activation='relu', padding="same")(concatenated)
        
        # Block of 16 times upsampling of the output using Bilinear interpolation
        output_layer = UpSampling2D(size=(16, 16), data_format=None, interpolation='bilinear')(output_layer)
        
        # Dense Layer for classification
        output_layer = Dense(1, activation="softmax")(output_layer)
        
        # Create the model using the input layer and the final output layer
        model = Model(input_layer, output_layer)
        return model
