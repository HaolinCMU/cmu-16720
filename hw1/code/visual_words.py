import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import matplotlib.pyplot as plt
import util
import random
from math import *

def extract_filter_responses(image):
	'''
	Extracts the filter responses for the given image.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	[output]
	* filter_responses: numpy.ndarray of shape (H,W,3F)
	'''
	image = image.astype('float')/255

	if image.ndim == 2:
		image = np.stack((image,)*3, axis=-1)
	elif image.shape[2] > 3:
		image = image[:,:,0:3]

	image = skimage.color.rgb2lab(image)

	filter_responses = np.empty((image.shape[0], image.shape[1], 0))

	filter_scale = np.array([1, 2, 4, 8, 8*sqrt(2)])

	for i in range(filter_scale.shape[0]):

		## 1. Gaussian
		filter_responses = np.concatenate((filter_responses, scipy.ndimage.filters.gaussian_filter(image, sigma=(filter_scale[i], filter_scale[i], 0))), axis=2)

		## 2. Laplacian of Gaussian
		r = scipy.ndimage.filters.gaussian_laplace(image[:,:,0], filter_scale[i])
		g = scipy.ndimage.filters.gaussian_laplace(image[:,:,1], filter_scale[i])
		b = scipy.ndimage.filters.gaussian_laplace(image[:,:,2], filter_scale[i])
		filter_responses = np.concatenate((filter_responses, np.stack((r,g,b), axis=-1)), axis=2)
		#filter_responses = np.concatenate((filter_responses, scipy.ndimage.filters.gaussian_laplace(image, sigma=(filter_scale[i], filter_scale[i], 0))), axis=2)

		## 3. Derivative of Gaussian in the x direction
		filter_responses = np.concatenate((filter_responses, scipy.ndimage.filters.gaussian_filter(image, sigma=(0, filter_scale[i], 0), order=1)), axis=2)
		## 4. Derivative of Gaussian in the y direction
		filter_responses = np.concatenate((filter_responses, scipy.ndimage.filters.gaussian_filter(image, sigma=(filter_scale[i], 0, 0), order=1)), axis=2)


	return filter_responses



def get_visual_words(image,dictionary):
	'''
	Compute visual words mapping for the given image using the dictionary of visual words.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)

	[output]
	* wordmap: numpy.ndarray of shape (H,W)
	'''

	filter_response = extract_filter_responses(image)
	H = filter_response.shape[0]
	W = filter_response.shape[1]
	filter_response = filter_response.reshape(H*W, filter_response.shape[-1])

	dist = scipy.spatial.distance.cdist(filter_response, dictionary)
	# (H*W) * K
	wordmap = np.argmin(dist, axis=1)
	wordmap = wordmap.reshape(H, W)

	return wordmap


def compute_dictionary_one_image(args):
	'''
	Extracts random samples of the dictionary entries from an image.
	This is a function run by a subprocess.

	[input]
	* i: index of training image
	* alpha: number of random samples
	* image_path: path of image file
	* time_start: time stamp of start time

	[saved]
	* sampled_response: numpy.ndarray of shape (alpha,3F)
	'''

	i,alpha,image_path = args
	# ----- TODO -----
	image = imageio.imread("../data/"+image_path[0])
	response = extract_filter_responses(image)

	x = np.random.choice(response.shape[0], alpha, replace=True)
	y = np.random.choice(response.shape[1], alpha, replace=True)
	random_samples = response[x, y, :]

	np.savez("../temp/"+str(i)+".npz", filter_responses=random_samples)


def compute_dictionary(num_workers=2):
	'''
	Creates the dictionary of visual words by clustering using k-means.

	[input]
	* num_workers: number of workers to process in parallel

	[saved]
	* dictionary: numpy.ndarray of shape (K,3F)
	'''

	train_data = np.load("../data/train_data.npz")

	alpha = 300
	K = 200

	with multiprocessing.Pool(num_workers) as p:
		args = zip(list(range(train_data['image_names'].shape[0])), [alpha]*train_data['image_names'].shape[0], train_data['image_names'])
		p.map(compute_dictionary_one_image, args)

	filter_responses = np.empty((0, 60))
	os.mkdir("../temp/")
	for i in range(train_data['image_names'].shape[0]):
		response = np.load("../temp/"+str(i)+".npz")
		filter_responses = np.append(filter_responses, response['filter_responses'], axis=0)

	kmeans = sklearn.cluster.KMeans(n_clusters=K, n_jobs=num_workers).fit(filter_responses)
	dictionary = kmeans.cluster_centers_

	np.save("dictionary.npy", dictionary)
