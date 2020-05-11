from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt


class GAN:
    def __init__(self):
        self.imgShape = (28, 28, 11)
        self.noiseDim = 128

        self.discriminator = self.buildDiscriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=Adam(lr=1e-5, beta_1=0.1),
                                   metrics=['accuracy'])
        self.discriminator.trainable = False

        self.generator = self.buildGenerator()
        self.combined = self.buildCombined()
        self.combined.compile(loss='binary_crossentropy',
                              optimizer=Adam(lr=8e-4, beta_1=0.5))


    def buildGenerator(self):
        model = Sequential()
        model.add(Dense(input_dim=(self.noiseDim + 10), units=1024)) # z=100, y=10
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        model.add(Dense(128*7*7))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
        model.add(UpSampling2D())
        model.add(Conv2D(64, 5, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        model.add(UpSampling2D())
        model.add(Conv2D(1, 5, padding='same'))
        model.add(Activation('tanh'))
        return model

    def buildDiscriminator(self):
        model = Sequential()
        model.add(Conv2D(64, kernel_size=5, strides=2, input_shape=self.imgShape, padding="same"))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=5, strides=2, padding="same"))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        img = Input(shape=self.imgShape)
        validity = model(img)
        return Model(img, validity)

    def buildCombined(self):
        noise = Input(shape=(self.noiseDim + 10,))
        label = Input(shape=(28, 28, 10,))

        fakeImage = self.generator(noise)
        fakeImage = Concatenate(axis=3)([fakeImage, label])

        self.discriminator.trainable = False
        valid = self.discriminator(fakeImage)

        model = Model(input=[noise, label], output=valid)
        return model


    def labelToImage(self, label):
        channel = np.zeros((28, 28, 10))
        channel[:, :, label] += 1
        return channel

    def conbiOnehot(self, noise, label):
        oneHot = np.eye(10)[label]
        return np.concatenate((noise, oneHot), axis=1)


    def train(self, epochs, batchSize):
        (X, Y), (_, _) = mnist.load_data()
        X = (X.astype(np.float32) - 127.5) / 127.5
        X = X.reshape([-1, 28, 28, 1])

        discriminator = self.buildDiscriminator()
        d_opt = Adam(lr=1e-5, beta_1=0.1)
        discriminator.compile(loss='binary_crossentropy', optimizer=d_opt, metrics=['accuracy'])

        g_opt = Adam(lr=.8e-4, beta_1=0.5)
        self.combined.compile(loss='binary_crossentropy', optimizer=g_opt)

        halfBatch = int(batchSize / 2)
        for epoch in range(epochs + 1):
            noise = np.random.normal(0, 1, (halfBatch, self.noiseDim))
            labelNoise = np.random.randint(0, 10, halfBatch)
            noise = self.conbiOnehot(noise, labelNoise)
            fakeImage = self.generator.predict(noise)
            labelImage = np.array([self.labelToImage(i) for i in labelNoise])
            fakeImage = np.concatenate((fakeImage, labelImage), axis=3)

            index = np.random.randint(0, X.shape[0], halfBatch)
            realImage, realLabel = X[index], Y[index]
            labelImage = np.array([self.labelToImage(i) for i in realLabel])
            realImage = np.concatenate((realImage, labelImage), axis=3)

            fakeLoss = self.discriminator.train_on_batch(fakeImage, np.zeros(halfBatch))
            realLoss = self.discriminator.train_on_batch(realImage, np.ones(halfBatch))
            disLoss = 0.5 * np.add(realLoss, fakeLoss)

            noise = np.random.normal(0, 1, (batchSize, self.noiseDim))
            randomLabel = np.random.randint(0, 10, batchSize)
            noise = self.conbiOnehot(noise, randomLabel)
            randomImage = np.array([self.labelToImage(i) for i in randomLabel])
            genLoss = self.combined.train_on_batch([noise, randomImage], np.ones(batchSize))

            print("epoch:%d,  [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, disLoss[0], 100 * disLoss[1], genLoss))
            if epoch % 1000 == 0:
                self.save_imgs(epoch)

        #self.generator.save('mnist_generator.h5')

    def save_imgs(self, epoch):
        r, c = 10, 10
        label = [i for i in range(10)] * 10

        noise = np.random.normal(0, 1, (r * c, self.noiseDim))
        noehot = np.array([np.eye(10)[i] for i in label])
        noise = np.concatenate((noise, noehot), axis=1)
        genImgs = self.generator.predict(noise)

        genImgs = 0.5 * genImgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(genImgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/cGAN_mnist/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=10000, batchSize=64)