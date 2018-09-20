import numpy as np
import multiprocessing
import threading
import queue
import imageio
import os,time
import torch
import skimage.transform
import torchvision.transforms
import util
import network_layers

from functools import partial

def build_recognition_system(vgg16,num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,K)
	* labels: numpy.ndarray of shape (N)
	'''


	train_data = np.load("../data/train_data.npz")
	# ----- TODO -----
	training_sample_num = train_data['image_names'].shape[0]

	# with multiprocessing.Pool(num_workers) as p:
	# 	args = zip(list(range(training_sample_num)), train_data['image_names'])
	# 	p.map(get_image_feature, args)

	with multiprocessing.Pool(num_workers) as p:
		p.map(partial(get_image_feature_pytorch, vgg16=vgg16), train_data['image_names'])
	# Test
	# for i in range(training_sample_num):
	# 	args = i, train_data['image_names'][i]
	# 	get_image_feature(args)

	for i in range(training_sample_num):
		get_image_feature_pytorch(train_data['image_names'][i], vgg16)

	# for i in range(train_data['image_names'].shape[0]):
	# 	temp = np.load("../temp/"+"training_image_"+str(i)+".npz")
	# 	features = np.append(features, np.reshape(temp['feature'], (1,-1)), axis=0)
	# 	labels = np.append(labels, temp['label'])
	#print(weights)
	#print(vgg16.features)
	pass


def evaluate_recognition_system(vgg16,num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''


	test_data = np.load("../data/test_data.npz")
	# ----- TODO -----
	pass


def preprocess_image(image):
	'''
	Preprocesses the image to load into the prebuilt network.

	[input]
	* image: numpy.ndarray of shape (H,W,3)

	[output]
	* image_processed: torch.Tensor of shape (3,H,W)
	'''
	image = image.astype('float') / 255
	image = skimage.transform.resize(image, (224, 224, 3))
	#print("Preprocessed image shape: ", image.shape)

	return image

def preprocess_image_pytorch(image):
	image = image.astype('float') / 255
	image = skimage.transform.resize(image, (224, 224, 3))
	image = np.swapaxes(image, axis1=1, axis2=2)
	image = np.swapaxes(image, axis1=0, axis2=1)
	image = image[np.newaxis, :]
	#print("Preprocessed image shape: ", image.shape)
	return torch.from_numpy(image)

def get_image_feature(args):
	'''
	Extracts deep features from the prebuilt VGG-16 network.
	This is a function run by a subprocess.
 	[input]
	* i: index of training image
	* image_path: path of image file
	* vgg16: prebuilt VGG-16 network.
	* time_start: time stamp of start time
 	[saved]
	* feat: evaluated deep feature
	'''

	#i,image_path,vgg16 = args
	i, image_path = args

	img_path = "../data/"+image_path[0]
	image = imageio.imread(img_path)
	img = preprocess_image(image)

	weights = util.get_VGG16_weights()
	feature = network_layers.extract_deep_feature(img, weights)

	#np.save("../temp/"+"image_feature_"+str(i)+".npy", feat)

	pass

def get_image_feature_pytorch(image_path, vgg16):

	img_path = "../data/"+image_path[0]
	image = imageio.imread(img_path)
	img = preprocess_image_pytorch(image)

	vgg16.forward(img)

def distance_to_set(feature,train_features):
	'''
	Compute distance between a deep feature with all training image deep features.

	[input]
	* feature: numpy.ndarray of shape (K)
	* train_features: numpy.ndarray of shape (N,K)

	[output]
	* dist: numpy.ndarray of shape (N)
	'''

	# ----- TODO -----

	pass
