"""Sample code for reading bin files"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import random
import struct

from tensorflow.core.example import example_pb2


def example_generator(data_path, single_pass):
	"""Generates tf.Examples from data files.
		Binary data format: <length><blob>. <length> represents the byte size
		of <blob>. <blob> is serialized tf.Example proto. The tf.Example contains
		the tokenized article text and summary. Supports both reading data continuously
		for training and doing a single pass for testing.
	Args:
		data_path: Path to the .bin data files. Can include wildcards,
		 e.g. if you have several training data chunk files
		 train_001.bin, train_002.bin, etc, then pass
		 data_path=train_* to access them all.
		single_pass: Boolean. If True, go through the dataset exactly once,
		 generating examples in the order they appear, then return.
		 Otherwise, generate random examples indefinitely.
	Yields:
		Deserialized tf.Example.
	"""
	while True:
		filelist = glob.glob(data_path) # get the list of datafiles
		if single_pass:
			filelist = sorted(filelist)
		else:
			random.shuffle(filelist)
		for f in filelist:
			reader = open(f, 'rb')
			while True:
				len_bytes = reader.read(8)
				if not len_bytes: break # finished reading this file
				str_len = struct.unpack('q', len_bytes)[0]
				example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
				yield example_pb2.Example.FromString(example_str)
		if single_pass:
			# print("example_generator completed reading all datafiles. No more data.")
			break