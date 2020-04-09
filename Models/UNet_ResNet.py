from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers import Input, concatenate, Add
from keras import backend as K

class UNet_ResNet:
    def __init__(self, depth = 16):
        self.depth = depth

    def CompositeConv2D(self, input_layer, num_convs, filters):
        output = input_layer
        for i in range(num_convs):
            output = Conv2D(filters, kernel_size=(3, 3), padding='same', activation="relu")(output)
        return output
    
    def resnet_block(self, input_layer, num_comp_convs, filters):
        output = input_layer
        for i in range(num_comp_convs):
            output = self.CompositeConv2D(output, 2, filters)
        output = Add()([output, input_layer])
        return output
    
    def encoder_block(self, input_layer, depth, dropout, num_comp_convs):
        output_layer = self.resnet_block(input_layer, num_comp_convs, depth)
        output_layer = Add()([output_layer, input_layer])
        output_layer = MaxPooling2D(pool_size=(2, 2))(output_layer)
        output_layer = Dropout(dropout)(output_layer)
        return output_layer

    def encoder(self, input_layer, depth):
        block1 = Conv2D(depth, kernel_size=(7, 7), padding='same', activation="relu")(input_layer)
        block2 = self.encoder_block(block1, depth*2, dropout=0.25, num_comp_convs=3)
        block3 = self.encoder_block(block2, depth*4, dropout=0.25, num_comp_convs=3)
        block4 = self.encoder_block(block3, depth*8, dropout=0.25, num_comp_convs=5)
        return block1, block2, block3, block4

    def decoder_block(self, input_layer, depth, dropout):
        output_layer = self.resnet_block(input_layer, num_comp_convs, depth)
        output_layer = Dropout(dropout)(output_layer)
        output_layer = Add()([output_layer, input_layer])
        return output_layer

    def decoder(self, block1, block2, block3, block4, block5, depth):
        upconvolved = Conv2DTranspose(depth, (3, 3), strides = (2, 2), padding = 'same')(block5)
        concatenated = concatenate([block4, upconvolved])
        output_layer = self.decoder_block(concatenated, depth, dropout=0.25)

        upconvolved = Conv2DTranspose(depth, (3, 3), strides = (2, 2), padding = 'same')(output_layer)
        concatenated = concatenate([block3, upconvolved])
        output_layer = self.decoder_block(concatenated, depth//2, dropout=0.25)

        upconvolved = Conv2DTranspose(depth, (3, 3), strides = (2, 2), padding = 'same')(output_layer)
        concatenated = concatenate([block2, upconvolved])
        output_layer = self.decoder_block(concatenated, depth//4, dropout=0.25)

        upconvolved = Conv2DTranspose(depth, (3, 3), strides = (2, 2), padding = 'same')(output_layer)
        concatenated = concatenate([block1, upconvolved])
        output_layer = self.decoder_block(concatenated, depth//8, dropout=0.25)

        return output_layer

    def UNet_ResNet(self, input_shape):
        input_layer = Input(shape=input_shape)

        block1, block2, block3, block4 = self.encoder(input_layer, self.depth)

        block5 = MaxPooling2D(pool_size=(2, 2))(block4)
        block5 = self.resnet_block(block5, 2, self.depth*16)

        decoded = self.decoder(block1, block2, block3, block4, block5, self.depth*8)

        upconvolved = Conv2DTranspose(self.depth, (3, 3), strides = (2, 2), padding = 'same')(decoded)

        output_layer = Conv2D(1, (1, 1), activation='sigmoid', padding="same")(upconvolved)

        model = Model(input_layer, output_layer)
        return model
