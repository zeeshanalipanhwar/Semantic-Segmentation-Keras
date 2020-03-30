from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers import Input
from keras import backend as K

class DeepLabV3:
    def __init__(self):
        pass

    def atrous_spatial_pyramid_pooling(self, input_layer, depth, dropout):
        conv11_layer = Conv2D(depth, (1, 1), activation='relu', padding="same")(input_layer)
        conv11_layer = Dropout(dropout)(conv11_layer)
        
        atrous_conv1 = AtrousConvolution2D(depth, (3, 3), atrous_rate=(06,06), activation='relu', padding="same")(input_layer)
        atrous_conv1 = Dropout(dropout)(atrous_conv1)
        atrous_conv2 = AtrousConvolution2D(depth, (3, 3), atrous_rate=(12,12), activation='relu', padding="same")(input_layer)
        atrous_conv2 = Dropout(dropout)(atrous_conv2)
        atrous_conv3 = AtrousConvolution2D(depth, (3, 3), atrous_rate=(18,18), activation='relu', padding="same")(input_layer)
        atrous_conv3 = Dropout(dropout)(atrous_conv3)

        return conv11_layer, atrous_conv1, atrous_conv2, atrous_conv3

    def encoder(self, input_layer, depth, dropout = 0.25):
        # Block one that reduces the input shape by 4
        output_layer = Conv2D(depth, (3, 3), activation='relu', padding="same")(input_layer)
        output_layer = Conv2D(depth, (3, 3), activation='relu', padding="same")(output_layer)
        output_layer = MaxPooling2D(pool_size=(2, 2))(output_layer)
        output_layer = Dropout(dropout)(output_layer)
        output_layer = Conv2D(depth*2, (3, 3), activation='relu', padding="same")(output_layer)
        output_layer = Conv2D(depth*2, (3, 3), activation='relu', padding="same")(output_layer)
        output_layer = MaxPooling2D(pool_size=(2, 2))(output_layer)
        output_layer = Dropout(dropout)(output_layer)
        
        # Block two that reduces the input shape by 8
        output_layer = Conv2D(depth*4, (3, 3), activation='relu', padding="same")(output_layer)
        output_layer = Conv2D(depth*4, (3, 3), activation='relu', padding="same")(output_layer)
        output_layer = MaxPooling2D(pool_size=(2, 2))(output_layer)
        output_layer = Dropout(dropout)(output_layer)
        
        # Block three that reduces the input shape by 16
        output_layer = Conv2D(depth*8, (3, 3), activation='relu', padding="same")(output_layer)
        output_layer = Conv2D(depth*8, (3, 3), activation='relu', padding="same")(output_layer)
        output_layer = MaxPooling2D(pool_size=(2, 2))(output_layer)
        output_layer = Dropout(dropout)(output_layer)
        
        # Block four that retains 16 using Atrous Convolution
        output_layer = AtrousConvolution2D(depth*4, (3, 3), atrous_rate=(2,2), activation='relu', padding="same")(output_layer)
        output_layer = AtrousConvolution2D(depth*4, (3, 3), atrous_rate=(2,2), activation='relu', padding="same")(output_layer)
        output_layer = MaxPooling2D(pool_size=(2, 2))(output_layer)
        output_layer = Dropout(dropout)(output_layer)
        
        # Block five of atrous spatial pyramid pooling and image max pooling
        conv11_layer, atrous_conv1, atrous_conv2, atrous_conv3 = self.atrous_spatial_pyramid_pooling(output_layer, depth*2, dropout)
        maxpooled_in = MaxPooling2D(pool_size=(32, 32))(input_layer)

        # Block six of concatination and 1x1 conv on the concatenated output
        concatenated = concatenate([conv11_layer, atrous_conv1, atrous_conv2, atrous_conv3, maxpooled_in])
        encoded_out = Conv2D(depth, (3, 3), activation='relu', padding="same")(concatenated)

        return encoded_out
        
    def decoder(self, encoded_out, output_shape):
        #decoded_out = UpSampling2D(size=(32, 32))(encoded_out)
        decoded_out = Conv2DTranspose(filters=output_shape[-1], kernel_size=(528, 528), strides=(16, 16))(encoded_out)
        return decoded_out

    def DeepLabV3(self, input_shape):
        depth = 32
        input_layer = Input(shape=input_shape)
        encoded_out = self.encoder(input_layer, depth)
        decoded_out = self.decoder(encoded_out, output_shape=input_shape)
        decoded_out = Dense(1, activation="softmax")(decoded_out)
        model = Model(input_layer, decoded_out)
        return model
