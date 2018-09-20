import numpy as np
import torchvision
import util
import matplotlib.pyplot as plt
import visual_words
import visual_recog
import deep_recog
import skimage.io

if __name__ == '__main__':

	num_cores = util.get_num_CPU()

	# path_img = "../data/kitchen/sun_aasmevtpkslccptd.jpg"
	# image = skimage.io.imread(path_img)
	# image = image.astype('float')/255
	# filter_responses = visual_words.extract_filter_responses(image)
	# util.display_filter_responses(filter_responses)

	# print("Start generating the dictionary...")
	# visual_words.compute_dictionary(num_workers=num_cores)
	# print("Dictionary is generated...")

	# path_img = "../data/kitchen/sun_aasmevtpkslccptd.jpg"
	# image = skimage.io.imread(path_img)
	#dictionary = np.load('dictionary.npy')
	# print(dictionary.shape)
	#wordmap = visual_words.get_visual_words(image,dictionary)
	# print(wordmap.shape)
	#
	# cm = plt.get_cmap('gist_rainbow')
	# colored_img = cm(wordmap)
	# orig_img = plt.imshow(image)
	# plt.show()
	# imgplot = plt.imshow(colored_img)
	# plt.show()
	#hist, bin_edges = visual_recog.get_feature_from_wordmap(wordmap, dictionary.shape[0])
	#visual_recog.get_feature_from_wordmap_SPM(wordmap, 2, dictionary.shape[0])
	# print(hist.shape)
	#util.save_wordmap(wordmap, filename)
	# plt.bar(bin_edges[:-1], hist, width = 1)
	# plt.xlim(min(bin_edges), max(bin_edges))
	# plt.show()

	# print("Start building a recognition system...")
	# visual_recog.build_recognition_system(num_workers=num_cores)
	# print("Recognition system is built...")

	#conf, accuracy = visual_recog.evaluate_recognition_system(num_workers=num_cores)
	# print("Start evaluation...")
	# visual_recog.evaluate_recognition_system(num_workers=num_cores)
	# print("Evaluation done!")
	#print(conf)
	#print(np.diag(conf).sum()/conf.sum())

	vgg16 = torchvision.models.vgg16(pretrained=True).double()
	vgg16.eval()

	#deep_recog.build_recognition_system(vgg16,num_workers=num_cores//2)

	conf, accuracy = deep_recog.evaluate_recognition_system(vgg16,num_workers=num_cores//2)
	#print(conf)
	#print(np.diag(conf).sum()/conf.sum())
