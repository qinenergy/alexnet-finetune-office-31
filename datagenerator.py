"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf
import numpy as np
import cv2

from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

# BGR
IMAGENET_MEAN = tf.constant([104.007, 116.669, 122.679], dtype=tf.float32)#tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
IMAGENET_MEAN_FULL = np.load("ilsvrc_2012_mean.npy")
IMAGENET_MEAN_FULL = np.transpose(IMAGENET_MEAN_FULL, [1,2,0])


Dataset = tf.data.Dataset
class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, txt_file, mode, batch_size, num_classes, shuffle=True,
                 buffer_size=1000, size=None, fulval=True):
        """Create a new ImageDataGenerator.

        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train
        e.g. a convolutional neural network.

        Args:
            txt_file: Path to the text file.
            mode: Either 'training' or 'validation'. Depending on this value,
                different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Wether or not to shuffle the data in the dataset and the
                initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.
            fulval: if false use a fixed subset of target file as validat-
                ion file. 

        Raises:
            ValueError: If an invalid mode is passed.

        """
        self.fulval = fulval
        self.txt_file = txt_file
        self.num_classes = num_classes
        self.size = size

        # retrieve the data from the text file
        self._read_txt_file()

        # number of samples in the dataset
        self.data_size = len(self.labels)

        # initial shuffling of the file and label lists (together!)
        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)

        # create dataset
        data = Dataset.from_tensor_slices((self.img_paths, self.labels))

        # distinguish between train/infer. when calling the parsing functions
        if mode == 'training':
            data = data.map(self._parse_function_train, num_parallel_calls=8)

        elif mode == 'inference':
            data = data.map(self._parse_function_inference)

        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        # create a new dataset with batches of images
        data = data.batch(batch_size)

        self.data = data

    def _read_txt_file(self):
        """Read the content of the text file and store it into lists."""
        self.img_paths = []
        self.labels = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split(' ')
                self.img_paths.append(items[0])
                self.labels.append(int(items[1]))
        if not self.fulval:
            idx = np.random.RandomState(seed=42).choice(len(self.labels), 1000)
            self.labels = list(np.array(self.labels)[idx])
            self.img_paths = list(np.array(self.img_paths)[idx])
        if self.size and self.size>len(self.labels):
            idx = np.random.choice(len(self.labels), self.size)
            self.labels = list(np.array(self.labels)[idx])
            self.img_paths = list(np.array(self.img_paths)[idx])
    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        path = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.labels.append(labels[i])

    def _parse_function_train(self, filename, label):
        """Input parser for samples of the training set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_jpeg(img_string, channels=3, dct_method="INTEGER_ACCURATE")
        img_decoded = tf.image.resize_images(img_decoded, [256,256])
        img_decoded = img_decoded[:, :, ::-1]
        img_decoded = tf.subtract(tf.to_float(img_decoded), IMAGENET_MEAN_FULL)
        img_resized = tf.to_float(tf.random_crop(img_decoded, [227, 227, 3]))
        #img_resized = tf.image.random_flip_left_right(img_resized)
        """
        Dataaugmentation comes here.
        """
        img_bgr = img_resized#tf.subtract(img_resized, IMAGENET_MEAN)


        return img_bgr, one_hot

    def _parse_function_inference(self, filename, label):
        """Input parser for samples of the validation/test set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        #img_string = tf.read_file("test_"+filename)
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_jpeg(img_string, channels=3, dct_method="INTEGER_ACCURATE")
        img_decoded = tf.image.resize_images(img_decoded, [227,227])
        img_decoded = img_decoded[:, :, ::-1]
        img_bgr = tf.subtract(tf.to_float(img_decoded), IMAGENET_MEAN)
        return img_bgr, one_hot
