import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


path = os.getcwd() + '\\augmented_data\\'

img_size = (64, 64)
trainX = np.zeros((72_800, img_size[0], img_size[1]))

update_per = 10000

for n, i in enumerate(os.listdir(path)):
    filename = os.fsdecode(i)
    temp_img = cv2.imread(path + filename, 0) / 255
    temp_img = cv2.resize(temp_img, img_size)
    trainX[n] = temp_img
    if n % update_per == 0:
        print(f'Image #: {n}')

trainY = np.zeros(len(trainX))

# generator

random_input = tf.keras.layers.Input(shape = 100)

x = tf.keras.layers.Dense(img_size[0] * 5 * 5)(random_input)
x = tf.keras.layers.Activation('swish')(x)
x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)

x = tf.keras.layers.Reshape((5, 5, img_size[0]))(x)

x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(5,5))(x)
x = tf.keras.layers.Activation('swish')(x)
x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)

x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(19,19))(x)
x = tf.keras.layers.Activation('swish')(x)
x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)

x = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(38,38))(x)
generated_image = tf.keras.layers.Activation('sigmoid')(x)

generator_network = tf.keras.models.Model(inputs=random_input, outputs=generated_image)
generator_network.summary()

# discriminator

image_input = tf.keras.layers.Input(shape=(img_size[0], img_size[1], 1))

x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3))(image_input)
x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

x = tf.keras.layers.Conv2D(filters=128, kernel_size=(5,5))(x)
x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)

x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7,7))(x)
x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)

x = tf.keras.layers.Flatten()(x)

x = tf.keras.layers.Dense(1)(x)
real_vs_fake_output = tf.keras.layers.Activation('sigmoid')(x)

discriminator_network = tf.keras.models.Model(inputs=image_input, outputs=real_vs_fake_output)
discriminator_network.summary()

adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.5)
discriminator_network.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])

# gan combined model

discriminator_network.trainable=False

g_output = generator_network(random_input)
d_output = discriminator_network(g_output)

dcgan_model = tf.keras.models.Model(random_input, d_output)
dcgan_model.summary()

dcgan_model.compile(loss='binary_crossentropy', optimizer=adam_optimizer)

indices = [i for i in range(len(trainX))]

def get_random_noise(batch_size, noise_size):
    random_values = np.random.randn(batch_size*noise_size)
    random_noise_batch = np.reshape(random_values, (batch_size, noise_size))
    return random_noise_batch

def get_fake_samples(generator_network, batch_size, noise_size):
    random_noise_batch = get_random_noise(batch_size, noise_size) 
    fake_samples = generator_network.predict_on_batch(random_noise_batch)
    return fake_samples

def get_real_samples(batch_size):
    random_indices = np.random.choice(indices, size=batch_size)
    real_images = trainX[np.array(random_indices),:]
    return real_images

def show_generator_results(generator_network):
    for k in range(9):
        plt.figure(figsize=(7, 7))
        fake_samples = get_fake_samples(generator_network, 9, noise_size)
        for j in range(9):
            plt.subplot(990 + 1 + j)
            plt.imshow(fake_samples[j,:,:,-1], cmap='gray')
            plt.axis('off')
        plt.show()
    return

epochs = 200
batch_size = 100
steps = 500
noise_size = 100
show_sample = False

for i in range(0, epochs):
    if show_sample and i % 10 == 0:
        fake_samples = get_fake_samples(generator_network, 1, noise_size).reshape(img_size)
        plt.title(f'Epoch: {i}')
        plt.axis('off')
        plt.imshow(fake_samples, cmap='gray')
        plt.show()
    for j in range(steps):
        fake_samples = get_fake_samples(generator_network, batch_size//2, noise_size).reshape((50,img_size[0],img_size[1]))
        real_samples = get_real_samples(batch_size=batch_size//2)

        fake_y = np.zeros((batch_size//2, 1))
        real_y = np.ones((batch_size//2, 1))
        
        input_batch = np.vstack((fake_samples, real_samples))
        output_labels = np.vstack((fake_y, real_y))
        
        discriminator_network.trainable=True
        loss_d = discriminator_network.train_on_batch(input_batch, output_labels)
        
        gan_input = get_random_noise(batch_size, noise_size)
        
        gan_output = np.ones((batch_size))
        
        discriminator_network.trainable=False
        loss_g = dcgan_model.train_on_batch(gan_input, gan_output)
        
        if j % 50 == 0:
            print(f'Epoch: {i}, Step: {j}, D-Loss: {loss_d[0]:.3f}, D-Acc: {loss_d[1] * 100:.3f}, G-Loss: {loss_g:.3f}')