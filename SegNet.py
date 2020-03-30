class SegNet:
    def __init__(self):
        pass

    def encoder(self, input_layer, depth=2):
        encoded_out = Conv2D(depth, (3, 3), activation='relu', padding="same")(input_layer)
        encoded_out = BatchNormalization(axis=-1)(encoded_out)
        encoded_out = Conv2D(depth, (3, 3), activation='relu', padding="same")(encoded_out)
        encoded_out = BatchNormalization(axis=-1)(encoded_out)
        encoded_out = MaxPooling2D(pool_size=(2, 2))(encoded_out)

        encoded_out = Conv2D(2*depth, (3, 3), activation='relu', padding="same")(encoded_out)
        encoded_out = BatchNormalization(axis=-1)(encoded_out)
        encoded_out = Conv2D(2*depth, (3, 3), activation='relu', padding="same")(encoded_out)
        encoded_out = BatchNormalization(axis=-1)(encoded_out)
        encoded_out = MaxPooling2D(pool_size=(2, 2))(encoded_out)

        encoded_out = Conv2D(4*depth, (3, 3), activation='relu', padding="same")(encoded_out)
        encoded_out = BatchNormalization(axis=-1)(encoded_out)
        encoded_out = Conv2D(4*depth, (3, 3), activation='relu', padding="same")(encoded_out)
        encoded_out = BatchNormalization(axis=-1)(encoded_out)
        encoded_out = Conv2D(4*depth, (3, 3), activation='relu', padding="same")(encoded_out)
        encoded_out = BatchNormalization(axis=-1)(encoded_out)
        encoded_out = MaxPooling2D(pool_size=(2, 2))(encoded_out)

        encoded_out = Conv2D(8*depth, (3, 3), activation='relu', padding="same")(encoded_out)
        encoded_out = BatchNormalization(axis=-1)(encoded_out)
        encoded_out = Conv2D(8*depth, (3, 3), activation='relu', padding="same")(encoded_out)
        encoded_out = BatchNormalization(axis=-1)(encoded_out)
        encoded_out = Conv2D(8*depth, (3, 3), activation='relu', padding="same")(encoded_out)
        encoded_out = BatchNormalization(axis=-1)(encoded_out)
        encoded_out = MaxPooling2D(pool_size=(2, 2))(encoded_out)

        encoded_out = Conv2D(8*depth, (3, 3), activation='relu', padding="same")(encoded_out)
        encoded_out = BatchNormalization(axis=-1)(encoded_out)
        encoded_out = Conv2D(8*depth, (3, 3), activation='relu', padding="same")(encoded_out)
        encoded_out = BatchNormalization(axis=-1)(encoded_out)
        encoded_out = Conv2D(8*depth, (3, 3), activation='relu', padding="same")(encoded_out)
        encoded_out = BatchNormalization(axis=-1)(encoded_out)
        encoded_out = MaxPooling2D(pool_size=(2, 2))(encoded_out)

        return encoded_out

    def decoder(self, encoded_out, depth=2):
        decoded_out = UpSampling2D(size=(2, 2))(encoded_out)
        decoded_out = Conv2D(8*depth, (3, 3), activation='relu', padding="same")(decoded_out)
        decoded_out = BatchNormalization(axis=-1)(decoded_out)
        decoded_out = Conv2D(8*depth, (3, 3), activation='relu', padding="same")(decoded_out)
        decoded_out = BatchNormalization(axis=-1)(decoded_out)
        decoded_out = Conv2D(8*depth, (3, 3), activation='relu', padding="same")(decoded_out)
        decoded_out = BatchNormalization(axis=-1)(decoded_out)

        decoded_out = UpSampling2D(size=(2, 2))(decoded_out)
        decoded_out = Conv2D(8*depth, (3, 3), activation='relu', padding="same")(decoded_out)
        decoded_out = BatchNormalization(axis=-1)(decoded_out)
        decoded_out = Conv2D(8*depth, (3, 3), activation='relu', padding="same")(decoded_out)
        decoded_out = BatchNormalization(axis=-1)(decoded_out)
        decoded_out = Conv2D(8*depth, (3, 3), activation='relu', padding="same")(decoded_out)
        decoded_out = BatchNormalization(axis=-1)(decoded_out)

        decoded_out = UpSampling2D(size=(2, 2))(decoded_out)
        decoded_out = Conv2D(4*depth, (3, 3), activation='relu', padding="same")(decoded_out)
        decoded_out = BatchNormalization(axis=-1)(decoded_out)
        decoded_out = Conv2D(4*depth, (3, 3), activation='relu', padding="same")(decoded_out)
        decoded_out = BatchNormalization(axis=-1)(decoded_out)
        decoded_out = Conv2D(4*depth, (3, 3), activation='relu', padding="same")(decoded_out)
        decoded_out = BatchNormalization(axis=-1)(decoded_out)

        decoded_out = UpSampling2D(size=(2, 2))(decoded_out)
        decoded_out = Conv2D(2*depth, (3, 3), activation='relu', padding="same")(decoded_out)
        decoded_out = BatchNormalization(axis=-1)(decoded_out)
        decoded_out = Conv2D(2*depth, (3, 3), activation='relu', padding="same")(decoded_out)
        decoded_out = BatchNormalization(axis=-1)(decoded_out)

        decoded_out = UpSampling2D(size=(2, 2))(decoded_out)
        decoded_out = Conv2D(depth, (3, 3), activation='relu', padding="same")(decoded_out)
        decoded_out = BatchNormalization(axis=-1)(decoded_out)
        decoded_out = Conv2D(depth, (3, 3), activation='relu', padding="same")(decoded_out)
        decoded_out = BatchNormalization(axis=-1)(decoded_out)

        return decoded_out

    def SegNet(self, input_shape):
        depth = 16
        input_layer = Input(shape=input_shape)
        encoded_out = self.encoder(input_layer, depth)
        decoded_out = self.decoder(encoded_out, depth)
        decoded_out = Dense(1, activation="sigmoid")(decoded_out)
        model = Model(input_layer, decoded_out)
        return model
