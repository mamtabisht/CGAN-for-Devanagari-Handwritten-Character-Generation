
# working on Devanagari Handwritten Character Dataset
# Code for generating Devanagari handwritten characters and numerals using Conditional Generative Adversarial Network (CGAN)
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
#matplotlib inline
import tensorflow as tf
import keras
import glob
import cv2

# Import the backend
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
import os

print(os.listdir("D:/DevanagariHandwrittenCharacterDataset"))

#Read the train & test Images and preprocessing
train_images = []
train_labels = [] 
for directory_path in glob.glob("D:/DevanagariHandwrittenCharacterDataset/Train/*"):
    label = directory_path.split("_")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28,28))
        train_images.append(img)
        train_labels.append(label)
train_images = np.array(train_images)
train_labels = np.array(train_labels)

label_to_id = {v:i for i,v in enumerate(np.unique(train_labels))}
id_to_label = {v: k for k, v in label_to_id.items()}
train_label_ids = np.array([label_to_id[x] for x in train_labels])

print(train_images.shape),print( train_label_ids.shape), print(train_labels.shape)

from matplotlib import pyplot
# plot raw pixel data
pyplot.imshow(train_images[100], cmap='gray_r')

# plot images from the training dataset
for i in range(9):
	# define subplot
	pyplot.subplot(3, 3, 1 + i)
	# turn off axis
	pyplot.axis('off')
	# plot raw pixel data
	pyplot.imshow(train_images[i], cmap='gray_r')# or cmap='gray_r'
pyplot.show()
################################################

#conditional gan 
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.datasets.fashion_mnist import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Concatenate

# define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1), n_classes=46):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 50)(in_label)
	# scale up to image dimensions with linear activation
	n_nodes = in_shape[0] * in_shape[1]
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((in_shape[0], in_shape[1], 1))(li)
	# image input
	in_image = Input(shape=in_shape)
	# concat label as a channel
	merge = Concatenate()([in_image, li])
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# flatten feature maps
	fe = Flatten()(fe)
	# dropout
	fe = Dropout(0.4)(fe)
	# output
	out_layer = Dense(1, activation='sigmoid')(fe)
	# define model
	model = Model([in_image, in_label], out_layer)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# define the standalone generator model
def define_generator(latent_dim, n_classes=46):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 50)(in_label)
	# linear multiplication
	n_nodes = 7 * 7
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((7, 7, 1))(li)
	# image generator input
	in_lat = Input(shape=(latent_dim,))
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	gen = Dense(n_nodes)(in_lat)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Reshape((7, 7, 128))(gen)
	# merge image gen and label input
	merge = Concatenate()([gen, li])
	# upsample to 14x14
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
	gen = LeakyReLU(alpha=0.2)(gen)
	# upsample to 28x28
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	# output
	out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
	# define model
	model = Model([in_lat, in_label], out_layer)
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# get noise and label inputs from generator model
	gen_noise, gen_label = g_model.input
	# get image output from the generator model
	gen_output = g_model.output
	# connect image output and label input from generator as inputs to discriminator
	gan_output = d_model([gen_output, gen_label])
	# define gan model as taking noise and label and outputting a classification
	model = Model([gen_noise, gen_label], gan_output)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

# load DHCD images
def load_real_samples():
	X = expand_dims(train_images, axis=-1)
	# convert from ints to floats
	X = X.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	return [X, train_label_ids]

# # select real samples
def generate_real_samples(dataset, n_samples):
	# split into images and labels
	images, labels = dataset
	# choose random instances
	ix = randint(0, images.shape[0], n_samples)
	# select images and labels
	X, labels = images[ix], labels[ix]
	# generate class labels
	y = ones((n_samples, 1))
	return [X, labels], y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=46):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	z_input, labels_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	images = generator.predict([z_input, labels_input])
	# create class labels
	y = zeros((n_samples, 1))
	return [images, labels_input], y

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			# get randomly selected 'real' samples
			[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
			# update discriminator model weights
			d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
			# generate 'fake' examples
			[X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# update discriminator model weights
			d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
			# prepare points in latent space as input for the generator
			[z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
			# summarize loss on this batch
			print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
				(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
	# save the generator model
	g_model.save('cgan_generator.h5')

# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
d_model.summary()
# create the generator
g_model = define_generator(latent_dim)
g_model.summary()
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = load_real_samples()
# train model
train(g_model, d_model, gan_model, dataset, latent_dim)

#############################################

# example of loading the generator model and generating images
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.models import load_model
from matplotlib import pyplot

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=46):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]

# create and save a plot of generated images
def save_plot(examples, n):
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i, :, :, 0], cmap='gray')
    filename = 'cgan5.png' 
    pyplot.savefig(filename) 
    #pyplot.savefig(filename, dpi=10, bbox_inches="tight",transparent=True, pad_inches=0)
    pyplot.show()

# load model
model = load_model('cgan_generator.h5')
# generate images
latent_points, labels = generate_latent_points(100,100)
# specify labels
labels = asarray([x for _ in range(10) for x in range(10)])
# generate images
X  = model.predict([latent_points, labels])
# scale from [-1,1] to [0,1]
X = (X + 1) / 2.0
# plot the result
save_plot(X, 1)
######################################
# load model
# example of loading the generator model and generating images of selected label
from math import sqrt
from numpy import asarray
from numpy.random import randn
from keras.models import load_model
from matplotlib import pyplot

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_class):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = asarray([n_class for _ in range(n_samples)])
	return [z_input, labels]

# create and save a plot of generated images
def save_plot(examples, n_examples):
    # plot images
    for i in range(n_examples):
        # define subplot
        pyplot.subplot(sqrt(n_examples), sqrt(n_examples), 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i, :, :, 0], cmap='gray')
    filename = 'cgan8.png' 
    #pyplot.savefig(filename)
    pyplot.savefig(filename, dpi=10, bbox_inches="tight",transparent=True, pad_inches=1)
    pyplot.show()

names = ['ka','kha','ga','gha','kna','cha','chha','ja','jha','yna','taamatar',
         'thaa','daa','dhaa','adna','tabala','tha','da','dha','na','pa','pha','ba','bha','ma',
         'yaw','ra','la','waw','motosaw','petchiryakha','patalosaw','ha','chhya','tra','gya','digit_0',
         'digit_1','digit_2','digit_3','digit_4','digit_5','digit_6','digit_7','digit_8','digit_9']

# load model
model = load_model('cgan_generator.h5')
latent_dim = 100
n_examples = 100 
n_class = 0 
# generate images
latent_points, labels = generate_latent_points(latent_dim, n_examples, n_class)
# generate images
X  = model.predict([latent_points, labels])
# scale from [-1,1] to [0,1]
X = (X + 1) / 2.0
# plot the result
save_plot(X, n_examples)
############################################################


#generate samples in folder labelwise
# make folder for results
from os import makedirs    
makedirs('cgan_generated/7', exist_ok=True)  

from math import sqrt
from numpy import asarray
from numpy.random import randn
from keras.models import load_model
from matplotlib import pyplot

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_class):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = asarray([n_class for _ in range(n_samples)])
	return [z_input, labels]

def save_plot_generated(examples,n_examples):
    image_no=1
    for i in range(n_examples):
        #print(i)
        pyplot.imshow(examples[i, :, :, 0], cmap='gray')
        # save plot to file=
        #filename = 'i.png'
        filename = 'cgan_generated/7/'+str(image_no) + '.png'
        #pyplot.savefig(filename)
        pyplot.savefig(filename, dpi=10, bbox_inches="tight",transparent=True, pad_inches=0)

        image_no += 1
        #pyplot.close()   
        pyplot.show()# load model
        
# load model
model = load_model('cgan_generator.h5')
latent_dim = 100
n_examples = 100 # must be a square
n_class = 7 # sneaker
# generate images
latent_points, labels = generate_latent_points(latent_dim, n_examples, n_class)
# generate images
X  = model.predict([latent_points, labels])
# scale from [-1,1] to [0,1]
X = (X + 1) / 2.0
# plot the result
save_plot_generated(X, n_examples)        

##############################################################
#generate samples in folder labelwise
#edited in loop
# make folder for results
'''
names = ['character_1_ka','character_2_kha','character_3_ga','character_4_gha','character_5_kna',
         'character_6_cha','character_7_chha','character_8_ja','character_9_jha','character_10_yna',
         'character_11_taamatar',
         'character_12_thaa','character_13_daa','character_14_dhaa','character_15_adna','character_16_tabala',
         'character_17_tha','character_18_da','character_19_dha','character_20_na','character_21_pa',
         'character_22_pha','character_23_ba','character_24_bha','character_25_ma',
         'character_26_yaw','character_27_ra','character_28_la','character_29_waw','character_30_motosaw',
         'character_31_petchiryakha','character_32_patalosaw','character_33_ha','character_34_chhya',
         'character_35_tra','character_36_gya','digit_0',
         'digit_1','digit_2','digit_3','digit_4','digit_5','digit_6','digit_7','digit_8','digit_9']

'''
names = ['digit_0','digit_1','digit_2','digit_3','digit_4','digit_5','digit_6','digit_7','digit_8','digit_9', 'character_15_adna','character_23_ba',' character_24_bha',' character_6_cha',' character_7_chha',' character_34_chhya',' character_18_da',' character_13_daa',' character_19_dha',' character_14_dhaa',' character_3_ga',' character_4_gha','character_36_gya',' character_33_ha',' character_8_ja',' character_9_jha',' character_1_ka',' character_2_kha',' character_5_kna',' character_28_la',' character_25_ma',' character_30_motosaw',' character_20_na',' character_21_pa','character_32_patalosaw',' character_31_petchiryakha','character_22_pha','character_27_ra',' character_11_taamatar',' character_16_tabala','character_17_tha',' character_12_thaa', 'character_35_tra',' character_29_waw',' character_26_yaw',' character_10_yna']
from os import makedirs 
from math import sqrt
from numpy import asarray
from numpy.random import randn
from keras.models import load_model
from matplotlib import pyplot


image_folder_no=0
for dir in range(46):
    print(dir)
    makedirs('2cgan_generated/'+names[image_folder_no], exist_ok=True)
   
    # generate points in latent space as input for the generator
    def generate_latent_points(latent_dim, n_samples, n_class):
        # generate points in the latent space
        x_input = randn(latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        z_input = x_input.reshape(n_samples, latent_dim)
        # generate labels
        labels = asarray([n_class for _ in range(n_samples)])
        return [z_input, labels]

    def save_plot_generated(examples,n_examples):
        image_no=201
        for i in range(n_examples):
            pyplot.imshow(examples[i, :, :, 0], cmap='gray')
            # save plot to file=
            #filename = 'i.png'
            pyplot.axis('off')
            filename = '2cgan_generated/'+names[image_folder_no]+'/'+str(image_no) + '.png'
            #pyplot.savefig(filename)
            pyplot.savefig(filename, dpi=10, bbox_inches="tight",transparent=True, pad_inches=0)
            image_no += 1
            #pyplot.close()   
            pyplot.show()# load model
        
    # load model
    model = load_model('cgan_generator.h5')
    latent_dim = 100
    n_examples = 100 
    n_class = image_folder_no  
    # generate images
    latent_points, labels = generate_latent_points(latent_dim, n_examples, n_class)
    # generate images
    X  = model.predict([latent_points, labels])
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # plot the result
    save_plot_generated(X, n_examples) 
    image_folder_no += 1       



#####################################################################3
# example of generating an image for a specific point in the latent space
from keras.models import load_model
from numpy import asarray
from matplotlib import pyplot
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.models import load_model
from matplotlib import pyplot
# load model
model = load_model('cgan_generator.h5')

latent_dim = 100
n_examples = 1 
n_class = 46 

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=46):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]

# generate images
latent_points, labels = generate_latent_points(latent_dim, n_examples, n_class)
# generate images
X  = model.predict([latent_points, labels])

# scale from [-1,1] to [0,1]
X = (X + 1) / 2.0
# plot the result
pyplot.imshow(X[0, :, :,0],  cmap='gray')
#pyplot.show()
filename = 'cgan5.png' 

#pyplot.savefig(filename)
#pyplot.show()
pyplot.axis('off')
pyplot.savefig(filename, dpi=10, bbox_inches="tight",transparent=True, pad_inches=0)
pyplot.show()
##################################################

