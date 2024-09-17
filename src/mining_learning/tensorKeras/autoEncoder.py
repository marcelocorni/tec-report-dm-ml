import numpy as np
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam

class Autoencoder:
    def __init__(self, input_dim, latent_space_dim, learning_rate=1e-3):
        self.input_dim = input_dim
        self.latent_space_dim = latent_space_dim
        self.learning_rate = learning_rate
        self.model = self._build()

    def _build(self):
        # Definir o encoder
        encoder_input = layers.Input(shape=(self.input_dim,), name="encoder_input")
        x = layers.Dense(32, activation='elu')(encoder_input)
        x = layers.Dense(16, activation='elu')(x)
        latent_space = layers.Dense(self.latent_space_dim, activation='elu', name='latent_space')(x)

        # Definir o decoder
        decoder_input = layers.Input(shape=(self.latent_space_dim,), name="decoder_input")
        x = layers.Dense(16, activation='elu')(decoder_input)
        x = layers.Dense(32, activation='elu')(x)
        decoder_output = layers.Dense(self.input_dim, activation='linear')(x)

        # Definir o modelo Autoencoder completo
        encoder_model = models.Model(encoder_input, latent_space, name="encoder")
        decoder_model = models.Model(decoder_input, decoder_output, name="decoder")

        autoencoder_input = encoder_input
        autoencoder_output = decoder_model(encoder_model(autoencoder_input))
        autoencoder_model = models.Model(autoencoder_input, autoencoder_output, name="autoencoder")

        # Compilar o modelo
        autoencoder_model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')

        return autoencoder_model

    def train(self, X_train, batch_size, epochs, validation_data=None):
        # Treinar o modelo Autoencoder
        history = self.model.fit(X_train, X_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=(validation_data, validation_data) if validation_data is not None else None)
        return history

    def encode(self, X):
        # Obter a representação latente dos dados
        encoder = self.model.get_layer('encoder')
        return encoder.predict(X)

    def decode(self, X):
        # Reconstruir os dados a partir da representação latente
        decoder = self.model.get_layer('decoder')
        return decoder.predict(X)
