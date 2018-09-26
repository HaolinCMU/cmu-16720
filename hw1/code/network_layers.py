import numpy as np
import scipy.ndimage
import os,time
import skimage.measure

def extract_deep_feature(x,vgg16_weights):
	'''
	Extracts deep features from the given VGG-16 weights.

	[input]
	* x: numpy.ndarray of shape (H,W,3)
	* vgg16_weights: numpy.ndarray of shape (L,3)

	[output]
	* feat: numpy.ndarray of shape (K)
	'''
	linear_layer_counter = 0

	for weight in vgg16_weights:
		# Terminate at FC7 layer
		if linear_layer_counter == 2:
			x = relu(x)
			return x

		layer_type = weight[0]

		if layer_type == "conv2d":
			layer_weight = weight[1]
			bias = weight[2]
			x = multichannel_conv2d(x, layer_weight, bias)
			#print("Image shape after "+layer_type+" : ", x.shape)

		elif layer_type == "maxpool2d":
			kernel_size = weight[1]
			x = max_pool2d(x, kernel_size)
			#print("Image shape after "+layer_type+" : ", x.shape)
		elif layer_type == "relu":
			x = relu(x)
			#print("Image shape after "+layer_type+" : ", x.shape)
		elif layer_type == "linear":
			layer_weight = weight[1]
			bias = weight[2]
			x = linear(x, layer_weight, bias)
			linear_layer_counter += 1
			#print("Image shape after "+layer_type+" : ", x.shape)

	return x

def multichannel_conv2d(x,weight,bias):
	'''
	Performs multi-channel 2D convolution.

	[input]
	* x: numpy.ndarray of shape (H,W,input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim,kernel_size,kernel_size)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* feat: numpy.ndarray of shape (H,W,output_dim)
	'''
	feat = np.empty((x.shape[0], x.shape[1], 0))
	input_dim = x.shape[2]
	output_dim = weight.shape[0]

	for i in range(output_dim):
		weight_single = np.swapaxes(weight[i], axis1=0, axis2=2) # input_dim * kernel_size * kernel_size
		#print("weight_single")
		convolved = np.sum(scipy.ndimage.convolve(x, weight_single), axis=2)[:,:,np.newaxis]
		#print("convolution")
		feat = np.append(feat, convolved, axis=2)
		#print("i: ", i)

	return feat + bias

def relu(x):
	'''
	Rectified linear unit.

	[input]
	* x: numpy.ndarray

	[output]
	* y: numpy.ndarray
	'''
	return np.maximum(x, 0)

def max_pool2d(x,size):
	'''
	2D max pooling operation.

	[input]
	* x: numpy.ndarray of shape (H,W,input_dim)
	* size: pooling receptive field

	[output]
	* y: numpy.ndarray of shape (H/size,W/size,input_dim)
	'''
	return skimage.measure.block_reduce(x, (size, size, 1), func=np.max)


def linear(x,W,b):
	'''
	Fully-connected layer.

	[input]
	* x: numpy.ndarray of shape (input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* y: numpy.ndarray of shape (output_dim)
	'''
	return W.dot(x) + b
