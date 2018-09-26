import numpy as np
import threading
import queue
import imageio
import os,time
import math
import visual_words

import matplotlib.pyplot as plt
from sklearn import preprocessing
import multiprocessing
import scipy.spatial.distance

def compute_feature_one_image(args):
	i, image_path, label = args
	dictionary = np.load("dictionary.npy")

	SPM_layer_num = 2
	K = dictionary.shape[0]

	feature = np.reshape(get_image_feature("../data/" + image_path[0], dictionary, SPM_layer_num, K), (1, -1))

	np.savez("../temp/"+"training_image_"+str(i)+".npz", feature=feature, label=label)


def build_recognition_system(num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,M)
	* labels: numpy.ndarray of shape (N)
	* dictionary: numpy.ndarray of shape (K,3F)
	* SPM_layer_num: number of spatial pyramid layers
	'''

	train_data = np.load("../data/train_data.npz")
	dictionary = np.load("dictionary.npy")

	SPM_layer_num = 2
	cluster_num = dictionary.shape[0]
	training_sample_num = train_data['image_names'].shape[0]

	features = np.empty((0, int(cluster_num*(pow(4, SPM_layer_num+1) - 1)/3)))
	labels = []

	os.mkdir("../temp/")
	with multiprocessing.Pool(num_workers) as p:
		args = zip(list(range(training_sample_num)), train_data['image_names'], train_data['labels'])
		p.map(compute_feature_one_image, args)
		
	for i in range(train_data['image_names'].shape[0]):
		temp = np.load("../temp/"+"training_image_"+str(i)+".npz")
		features = np.append(features, np.reshape(temp['feature'], (1,-1)), axis=0)
		labels = np.append(labels, temp['label'])

	np.savez("trained_system.npz", features=features, labels=labels, dictionary=dictionary, SPM_layer_num=SPM_layer_num)

def evaluate_one_image(index):

	i = index

	test_data = np.load("../data/test_data.npz")
	trained_system = np.load("trained_system.npz")

	features = trained_system['features']
	labels = trained_system['labels']
	dictionary = trained_system['dictionary']
	SPM_layer_num = int(trained_system['SPM_layer_num'])

	img_path = "../data/"+test_data['image_names'][i][0]
	image = imageio.imread(img_path)
	wordmap = visual_words.get_visual_words(image, dictionary)

	hist = get_feature_from_wordmap_SPM(wordmap, SPM_layer_num, dictionary.shape[0])
	similarity = distance_to_set(hist, features)
	predicted_label = np.argmax(similarity)

	np.save("../temp/"+"predicted_label_"+str(i)+".npy", labels[predicted_label])

def evaluate_recognition_system(num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''

	test_data = np.load("../data/test_data.npz")
	trained_system = np.load("trained_system.npz")

	features = trained_system['features']
	labels = trained_system['labels']
	dictionary = trained_system['dictionary']
	SPM_layer_num = int(trained_system['SPM_layer_num'])

	test_sample_num = test_data['image_names'].shape[0]

	with multiprocessing.Pool(num_workers) as p:
		index = list(range(test_sample_num))
		p.map(evaluate_one_image, index)

	conf = np.zeros((8,8))

	for i in range(test_sample_num):
		predicted_label = np.load("../temp/"+"predicted_label_"+str(i)+".npy")
		if test_data['labels'][i] != predicted_label:
			print(test_data['image_names'][i][0]+" is predicted as: "+str(predicted_label))
		conf[test_data['labels'][i], int(predicted_label)] += 1

	accuracy = np.trace(conf) / np.sum(conf)
	print("Accuracy (Bag of visual words): ", accuracy)

	return conf, accuracy


def get_image_feature(file_path,dictionary,layer_num,K):
	'''
	Extracts the spatial pyramid matching feature.

	[input]
	* file_path: path of image file to read
	* dictionary: numpy.ndarray of shape (K,3F)
	* layer_num: number of spatial pyramid layers
	* K: number of clusters for the word maps

	[output]
	* feature: numpy.ndarray of shape (K)
	'''

	image = imageio.imread(file_path)
	wordmap = visual_words.get_visual_words(image,dictionary)
	return get_feature_from_wordmap_SPM(wordmap, layer_num, K)


def distance_to_set(word_hist,histograms):
	'''
	Compute similarity between a histogram of visual words with all training image histograms.

	[input]
	* word_hist: numpy.ndarray of shape (K)
	* histograms: numpy.ndarray of shape (N,K)

	[output]
	* sim: numpy.ndarray of shape (N)
	'''

	minima = np.minimum(word_hist, histograms)
	similarity = np.sum(minima, axis=1)

	return similarity

def get_feature_from_wordmap(wordmap,dict_size):
	'''
	Compute histogram of visual words.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* dict_size: dictionary size K

	[output]
	* hist: numpy.ndarray of shape (K)
	'''

	return np.histogram(wordmap, bins=dict_size, density=True)


def get_feature_from_wordmap_SPM(wordmap,layer_num,dict_size):
	'''
	Compute histogram of visual words using spatial pyramid matching.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* layer_num: number of spatial pyramid layers
	* dict_size: dictionary size K

	[output]
	* hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
	'''
	hist_all = []
	norm_factor = wordmap.shape[0]*wordmap.shape[1]

	for i in range(layer_num+1):

		if i == 0 or i == 1:
			weight = pow(2, -layer_num)
		else:
			weight = pow(2, layer_num-i-1)

		cell_num = pow(2, i)

		x = np.array_split(wordmap, cell_num, axis=0)
		for rows in x:
			y = np.array_split(rows, cell_num, axis=1)
			for cols in y:
				hist, bin_edges = np.histogram(cols, bins=dict_size)
				hist_all = np.append(hist_all, hist / norm_factor * weight)

	# Visualization of the final concatenated nhistogram
	# bin_edges = np.arange(hist_all.shape[0]+1)
	# plt.bar(bin_edges[:-1], hist_all, width = 1)
	# plt.xlim(min(bin_edges), max(bin_edges))
	# plt.show()
	return hist_all
