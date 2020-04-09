from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers import Input
from keras.models import Model

from ..Utils.custom_layers import MaxPooling2DWithIndices, MaxUnpooling2DWithIndices

class SegNet_ResNet:
    def __init__(self, depth = 64):
        self.depth = depth

    def CompositeConv2D(self, input_layer, num_convs, filters):
        output = input_layer
        for i in range(num_convs):
            output = Conv2D(filters, kernel_size=(3, 3), padding='same', activation="relu")(output)
            output = BatchNormalization(axis=3)(output)
        return output
        
    def resnet_block(self, input_layer, num_comp_convs, filters):
        output = input_layer
        for i in range(num_comp_convs):
            output = CompositeConv2D(output, 2, filters)
        return output+input_layer

    def encoder_resnet(self, input_layer):
        encoded_out = Conv2D(self.depth, kernel_size=(7, 7), padding='same', activation="relu")(input_layer)
        encoded_out, indices1 = MaxPooling2DWithIndices(pool_size=2,strides=2)(encoded_out)

        encoded_out = self.resnet_block(encoded_out, 3, self.depth)
        encoded_out, indices2 = MaxPooling2DWithIndices(pool_size=2, strides=2)(encoded_out)

        encoded_out = self.resnet_block(encoded_out, 3, self.depth)
        encoded_out, indices3 = MaxPooling2DWithIndices(pool_size=2, strides=2)(encoded_out)

        encoded_out = self.resnet_block(encoded_out, 5, self.depth)
        encoded_out, indices4 = MaxPooling2DWithIndices(pool_size=2, strides=2)(encoded_out)

        encoded_out = self.resnet_block(encoded_out, 2, self.depth)
        encoded_out, indices5 = MaxPooling2DWithIndices(pool_size=2, strides=2)(encoded_out)

        return [encoded_out, indices1, indices2, indices3, indices4, indices5]

    def decoder_resnet(self, encoded_out, indices1, indices2, indices3, indices4, indices5):
        decoded_out = MaxUnpooling2DWithIndices()([encoded_out, indices5])
        decoded_out = self.resnet_block(decoded_out, 2, self.depth)

        decoded_out = MaxUnpooling2DWithIndices()([decoded_out, indices4])
        decoded_out = self.resnet_block(decoded_out, 5, self.depth)

        decoded_out = MaxUnpooling2DWithIndices()([decoded_out, indices3])
        decoded_out = self.resnet_block(decoded_out, 3, self.depth)

        decoded_out = MaxUnpooling2DWithIndices()([decoded_out, indices2])
        decoded_out = self.resnet_block(decoded_out, 3, self.depth)

        decoded_out = MaxUnpooling2DWithIndices()([decoded_out, indices1])
        decoded_out = Conv2D(self.depth, kernel_size=(7, 7), padding='same', activation="relu")(decoded_out)
        
        return decoded_out

    def SegNet_ResNet(self, input_shape):
        input_layer = Input(shape=input_shape)
        encoded_out, indices1, indices2, indices3, indices4, indices5 = self.encoder_resnet(input_layer)
        decoded_out = self.decoder_resnet(encoded_out, indices1, indices2, indices3, indices4, indices5)
        decoded_out = Conv2D(1, kernel_size=(3, 3), padding="same", activation="sigmoid")(decoded_out)
        model = Model(input_layer, decoded_out)
        return model
