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

	training_sample_num = train_data['image_names'].shape[0]

	### Use functions in network_layers.py to train the system (very slow)
	# with multiprocessing.Pool(num_workers) as p:
	# 	args = zip(list(range(training_sample_num)), train_data['image_names'])
	# 	p.map(get_image_feature, args)

	os.makedirs("../temp/", exist_ok=True)
	### Use Pytorch VGG16 to train the system
	if len(vgg16.classifier) == 7:
		vgg16.classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-2])

	with multiprocessing.Pool(num_workers) as p:
		args = zip(list(range(training_sample_num)), train_data['image_names'])
		p.map(partial(get_image_feature_pytorch, vgg16=vgg16), args)

	### Test the feature extraction using functions in network_layers.py
	# for i in range(training_sample_num):
	# 	args = i, train_data['image_names'][i]
	# 	get_image_feature(args)

	# ## Test the feature extraction using pytorch
	# for i in range(training_sample_num):
	# 	args = i, train_data['image_names'][i]
	# 	get_image_feature_pytorch(args, vgg16)

	features = np.empty((0, 4096))

	for i in range(training_sample_num):
		temp = np.load("../temp/"+"image_feature_"+str(i)+".npy")
		features = np.append(features, temp, axis=0)

	labels = train_data['labels']

	np.savez("trained_system_deep.npz", features=features, labels=labels)

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

	i, image_path = args

	img_path = "../data/"+image_path[0]
	image = imageio.imread(img_path)
	img = preprocess_image(image)

	weights = util.get_VGG16_weights()
	feature = network_layers.extract_deep_feature(img, weights)

	np.save("../temp/"+"image_feature_"+str(i)+".npy", feature)

def evaluate_one_image(args):

	train_features = np.load("trained_system_deep.npz")['features']
	labels = np.load("trained_system_deep.npz")['labels']

	i, image_path = args
	img_path = "../data/"+image_path[0]
	image = imageio.imread(img_path)
	img = preprocess_image(image)

	weights = util.get_VGG16_weights()
	feature = network_layers.extract_deep_feature(img, weights)
	label = labels[np.argmax(distance_to_set(feature, train_features))]
	print("Evaluation done for image ", i)
	np.save("../temp/"+"predicted_label_deep_"+str(i)+".npy", label)

def get_image_feature_pytorch(args, vgg16):

	i, image_path = args
	img_path = "../data/"+image_path[0]
	image = imageio.imread(img_path)
	img = preprocess_image_pytorch(image)

	feature = vgg16.forward(img)

	np.save("../temp/"+"image_feature_"+str(i)+".npy", feature.detach().numpy())

def evaluate_one_image_pytorch(args, vgg16):
	### This is a function called in sub-process

	train_features = np.load("trained_system_deep.npz")['features']
	labels = np.load("trained_system_deep.npz")['labels']

	i, image_path = args
	img_path = "../data/"+image_path[0]
	image = imageio.imread(img_path)
	img = preprocess_image_pytorch(image)

	feature = vgg16.forward(img).detach().numpy()
	label = labels[np.argmax(distance_to_set(feature, train_features))]

	np.save("../temp/"+"predicted_label_deep_"+str(i)+".npy", label)


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

	test_sample_num = test_data['image_names'].shape[0]

	if len(vgg16.classifier) == 7:
		vgg16.classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-2])

	with multiprocessing.Pool(num_workers) as p:
		args = zip(list(range(test_sample_num)), test_data['image_names'])
		#p.map(partial(evaluate_one_image_pytorch, vgg16=vgg16), args)
		p.map(evaluate_one_image, args)

	conf = np.zeros((8,8))

	for i in range(test_sample_num):
		predicted_label = np.load("../temp/"+"predicted_label_deep_"+str(i)+".npy")
		conf[test_data['labels'][i], int(predicted_label)] += 1

	accuracy = np.trace(conf) / np.sum(conf)
	#print("Accuracy (VGG16): ", accuracy)

	return conf, accuracy

def preprocess_image(image):
	'''
	Preprocesses the image to load into the prebuilt network.

	[input]
	* image: numpy.ndarray of shape (H,W,3)

	[output]
	* image_processed: numpy.ndarray of shape (224, 224, 3)
	'''
	image = image.astype('float') / 255
	image = skimage.transform.resize(image, (224, 224, 3), mode='reflect')

	return image

def preprocess_image_pytorch(image):
	'''
	Preprocesses the image to load into the prebuilt network.

	[input]
	* image: numpy.ndarray of shape (H,W,3)

	[output]
	* image_processed: torch.Tensor of shape (1, 3, 224, 224)
	'''
	image = image.astype('float') / 255
	image = skimage.transform.resize(image, (224, 224, 3), mode='reflect')


	mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
	std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
	image = (image - mean) / std

	image = np.swapaxes(image, axis1=1, axis2=2)
	image = np.swapaxes(image, axis1=0, axis2=1)
	image = image[np.newaxis, :]

	return torch.from_numpy(image)


def distance_to_set(feature,train_features):
	'''
	Compute distance between a deep feature with all training image deep features.

	[input]
	* feature: numpy.ndarray of shape (K)
	* train_features: numpy.ndarray of shape (N,K)

	[output]
	* dist: numpy.ndarray of shape (N)
	'''
	### Negative euclidean distance
	return -np.linalg.norm(feature - train_features, axis=1)
