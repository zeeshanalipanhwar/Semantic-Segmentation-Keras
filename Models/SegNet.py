from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers import Input
from keras.models import Model

class SegNet:
    def __init__(self, depth = 64):
        self.depth = depth

    def CompositeConv2D(self, input_layer, num_convs, filters):
        output = input_layer
        for i in range(num_convs):
            output = Conv2D(filters, kernel_size=(3, 3), padding='same', activation="relu")(output)
            output = BatchNormalization(axis=3)(output)
        return output
        
    def encoder(self, input_layer):
        encoded_out = self.CompositeConv2D(input_layer, 2, self.depth)
        encoded_out, indices1 = MaxPoolingWithIndices(pool_size=2,strides=2)(encoded_out)

        encoded_out = self.CompositeConv2D(encoded_out, 2, self.depth*2)
        encoded_out, indices2 = MaxPoolingWithIndices(pool_size=2,strides=2)(encoded_out)

        encoded_out = self.CompositeConv2D(encoded_out, 3, self.depth*4)
        encoded_out, indices3 = MaxPoolingWithIndices(pool_size=2,strides=2)(encoded_out)

        encoded_out = self.CompositeConv2D(encoded_out, 3, self.depth*4)
        encoded_out, indices4 = MaxPoolingWithIndices(pool_size=2,strides=2)(encoded_out)

        encoded_out = self.CompositeConv2D(encoded_out, 3, self.depth*4)
        encoded_out, indices5 = MaxPoolingWithIndices(pool_size=2,strides=2)(encoded_out)

        return [encoded_out, indices1, indices2, indices3, indices4, indices5]

    def decoder(self, encoded_out, indices1, indices2, indices3, indices4, indices5):
        decoded_out = UpSamplingWithIndices()([encoded_out, indices5])
        decoded_out = self.CompositeConv2D(decoded_out, 3, self.depth*4)

        decoded_out = UpSamplingWithIndices()([decoded_out, indices5])
        decoded_out = self.CompositeConv2D(decoded_out, 3, self.depth*4)

        decoded_out = UpSamplingWithIndices()([decoded_out, indices5])
        decoded_out = self.CompositeConv2D(decoded_out, 3, self.depth*4)

        decoded_out = UpSamplingWithIndices()([decoded_out, indices5])
        decoded_out = self.CompositeConv2D(decoded_out, 2, self.depth*2)

        decoded_out = UpSamplingWithIndices()([decoded_out, indices5])
        decoded_out = self.CompositeConv2D(decoded_out, 2, self.depth)
        
        return decoded_out

    def SegNet(self, input_shape):
        input_layer = Input(shape=input_shape)
        encoded_out, indices1, indices2, indices3, indices4, indices5 = self.encoder(input_layer)
        decoded_out = self.decoder(encoded_out, indices1, indices2, indices3, indices4, indices5)
        decoded_out = Conv2D(1, kernel_size=(1, 1), activation="sigmoid")(decoded_out)
        model = Model(input_layer, decoded_out)
        return model
