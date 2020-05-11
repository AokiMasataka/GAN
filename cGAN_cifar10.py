from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt


class ConditionalGAN:
    def __init__(self):
        self.imageShape = (32, 32, 3 + 10)
        self.noiseDim = 512

        self.discriminator = self.buildDiscriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=Adam(lr=1e-5, beta_1=0.1),
                                   metrics=['accuracy'])

        self.generator = self.buildGenerator()
        self.combined = self.buildCombined()
        self.combined.compile(loss='binary_crossentropy',
                              optimizer=Adam(lr=8e-4, beta_1=0.5))

    def buildGenerator(self):
        model = Sequential()
        model.add(Dense(input_dim=(self.noiseDim + 10), units=2048))  # z=100, y=10
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        model.add(Reshape((4, 4, 128), input_shape=(128 * 4 * 4,)))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(64, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        model.add(UpSampling2D((2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(3, (5, 5), padding='same'))
        model.add(Activation('tanh'))
        return model

    def buildDiscriminator(self):
        model = Sequential()
        model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(self.imageShape)))
        model.add(LeakyReLU(0.2))
        model.add(Conv2D(128, (5, 5), strides=(2, 2)))
        model.add(LeakyReLU(0.2))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        return model

    def buildCombined(self):
        noise = Input(shape=(self.noiseDim + 10,))
        label = Input(shape=(32, 32, 10,))

        fakeImage = self.generator(noise)
        fakeImage = Concatenate(axis=3)([fakeImage, label])

        self.discriminator.trainable = False
        valid = self.discriminator(fakeImage)

        model = Model(input=[noise, label], output=valid)
        return model

    def labelToImage(self, label):
        channel = np.zeros((32, 32, 10))
        channel[:, :, label] += 1
        return channel

    def conbiOnehot(self, noise, label):
        oneHot = np.eye(10)[label]
        return np.concatenate((noise, oneHot), axis=1)

    def train(self, iteration, batchSize):
        (X, Y), (_, _) = cifar10.load_data()
        X = (X.astype(np.float32) - 127.5) / 127.5
        X = X.reshape([-1, 32, 32, 3])
        halfBatch = int(batchSize / 2)

        discriminator = self.buildDiscriminator()
        d_opt = Adam(lr=1e-5, beta_1=0.1)
        discriminator.compile(loss='binary_crossentropy', optimizer=d_opt, metrics=['accuracy'])

        g_opt = Adam(lr=.8e-4, beta_1=0.5)
        self.combined.compile(loss='binary_crossentropy', optimizer=g_opt)

        for i in range(iteration + 1):
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

            print("epoch:%d,  [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (i, disLoss[0], 100 * disLoss[1], genLoss))
            if i % 1000 == 0:
                self.save_imgs(i)


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
                axs[i, j].imshow(genImgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/cGAN_cifar/cifar_c_%d.png" % epoch)
        plt.close()

if __name__ == '__main__':
    gan = ConditionalGAN()
    gan.train(iteration=100000, batchSize=128)