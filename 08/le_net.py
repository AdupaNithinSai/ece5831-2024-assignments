import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

class LeNet:
    def __init__(self, batch_size=32, epochs=20):
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self._create_lenet()
        self._compile()
    
    def _create_lenet(self):
        self.model = Sequential([
            Input(shape=(28, 28, 1)),
            Conv2D(filters=6, kernel_size=(5, 5), activation='sigmoid', padding='same'),
            AveragePooling2D(pool_size=(2, 2), strides=2),
            Conv2D(filters=16, kernel_size=(5, 5), activation='sigmoid', padding='same'),
            AveragePooling2D(pool_size=(2, 2), strides=2),
            Flatten(),
            Dense(120, activation='sigmoid'),
            Dense(84, activation='sigmoid'),
            Dense(10, activation='softmax')
        ])

    def _compile(self):
        if self.model is None:
            print('Error: Create a model first.')
        
        self.model.compile(optimizer='Adam',loss='categorical_crossentropy', metrics=['accuracy'])
    
    def _preprocess(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train / 255.0
        x_test = x_test / 255.0

        self.x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        self.x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

        self.y_train = to_categorical(y_train, 10)
        self.y_test = to_categorical(y_test, 10)

    def train(self):
        self._preprocess()
        self.model.fit(self.x_train, self.y_train, 
                       batch_size=self.batch_size, 
                       epochs=self.epochs)

    def save(self, model_path_name):
        self.model.save(model_path_name + '.keras', include_optimizer=False)

    def load(self, model_path_name):
        # Load the model
        self.model = load_model(model_path_name + '.keras', compile=False)
        self._compile()  
        
    def predict(self, images):
        preprocessed_images = np.array(images) / 255.0
        if len(preprocessed_images.shape) == 3:
            preprocessed_images = preprocessed_images.reshape(preprocessed_images.shape[0], 28, 28, 1)
        else:
            preprocessed_images = preprocessed_images.reshape(1, 28, 28, 1)

        predictions = np.argmax(self.model.predict(preprocessed_images), axis=1)
        return predictions