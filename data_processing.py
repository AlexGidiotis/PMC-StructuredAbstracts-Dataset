"""Data preprocessing for PMC-SA"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import logging
import argparse
import struct as strc
import sys

from lxml import etree

import pyspark
from pyspark.sql import functions as F
from pyspark.sql import types as spark_types
from pyspark.ml import feature as ml_feature

from nltk import word_tokenize, sent_tokenize

import tensorflow as tf
from tensorflow.core.example import example_pb2


# Some special tokens
SENT_START = '<s>'
SENT_END = '</s>'
TITLE_START = '<t>'
TITLE_END = '</t>'
SEC_START = '<sec>'
SEC_END = '</sec>'


def create_filelist(in_path, out_path):
	"""Reads the dir tree under in_path and creates
	 a list of all nxml file.
	
	Args:
		in_path: The path to the top level of the data
		out_path: The filename to write the filelist
	"""
	nxml_ext_regex = re.compile('\.nxml$')
	with open(out_path, 'w') as of:
		for path, _, files in os.walk(in_path):
			for file in files:
				if nxml_ext_regex.search(file) is not None:
					file_name = path + '/' + file
					of.write(file_name)
					of.write('\n')
	return


def read_files(
		sc,
		filelist,
		num_part):
	"""Read the files from filelist into a dataframe.
	
	Args:
		sc: An active spark context
		filelist: The filename of the filelist txt
		num_part: The number of partitions

	Returns:
		df: A dataframe with the read files
	"""
	path_rdd = sc.textFile(filelist, minPartitions=num_part)
	df = path_rdd.map(
			lambda x: pyspark.sql.Row(
				file_name=os.path.basename(x),
				**read_file(x))) \
		.toDF() \
		.filter((F.col('abstract').isNotNull()) \
			& (F.col('pmc_id').isNotNull()) \
			& (F.col('full_text').isNotNull()))

	return df


def read_file(file_path):
	"""Read a single file.
	This will be called as map function by spark.

	Args:
		file_path: The filename

	Returns:
		data_dict: An object with fields:
		 pmc_id: string or None
		 abstract: string or None
		 full_text: string or None
	"""
	# We skip files we cannot read correctly.
	try:
		tree = read_xml(file_path, nxml=True)
	except: # If we fail to read format
		data_dict = {
			'pmc_id': None,
			'abstract': None,
			'full_text': None
		}
		return data_dict

	# Ignore malformed abstracts
	try:
		# Read abstract
		abstracts = []
		abstract_tree = tree.findall('.//abstract')[0]
		if len(abstract_tree) > 1: # Else there is no abstract
			for abstr in abstract_tree:
				abs_section_text = stringify_children(abstr)
				if len(abs_section_text) < 5:
					continue
				abs_section_text = re.sub('\s+', ' ', abs_section_text) \
					.replace('\n', ' ') \
					.replace('\t', ' ') \
					.strip()

				# Split into sentences
				abs_sents = sent_tokenize(abs_section_text)
				abs_proc_text = []
				for sent in abs_sents:
					proc_sent = SENT_START + sent \
					 + ' ' + SENT_END
					abs_proc_text.append(proc_sent)
				abs_proc_text = ' '.join(abs_proc_text)
				abs_section = SEC_START + abs_proc_text \
				 + SEC_END
				abstracts.append(abs_section)

			if not abstracts:
				abstract = None
			else:
				abstract = ' '.join(abstracts)
				# We don't want very short abstracts
				if len(abstract) < 5:
					abstract = None
		else:
			abstract = None
	except: # If anything fails we ignore.
		abstract = None

	# Ignore malformed full texts.
	try:
		# Read body
		body = tree.xpath('.//body')[0]
		parts = []
		for i, part in enumerate(body):
			section_text = stringify_children(part)
			section_text = re.sub('\s+', ' ', section_text)
			# Discard short sections
			if len(section_text.split()) > 5:
				section = SEC_START \
				 + section_text + ' ' + SEC_END
				parts.append(section)

		full_text = ' '.join(parts).lstrip()
		full_text = re.sub('\s+', ' ', full_text)
		full_text = re.sub('\s\.', '.', full_text)
		full_text = re.sub('\s,', ',', full_text)

		# We don't want bad texts
		if len(full_text) < 10:
			full_text = None
	# Sometimes there is no text body
	except:
		full_text = None

	# Get the pmc id
	try:
		pmid_tree = tree.xpath('.//article-id')
		for ar_id in pmid_tree:
			if ar_id.attrib['pub-id-type'] == 'pmc':
				pmcid = str(ar_id.xpath("text()")[0])
	except:
		pmcid = None

	data_dict = {
		'pmc_id': pmcid,
		'abstract': abstract,
		'full_text': full_text
	}

	return data_dict


def read_xml(path, nxml=False):
	"""Parse tree from given XML path."""
	try:
		tree = etree.parse(path)
	except:
		try:
			tree = etree.fromstring(path)
		except Exception as e:
			print("Error: it was not able to read a path, a file-like object, or a string as an XML")
			raise
	if '.nxml' in path or nxml:
		remove_namespace(tree)
	return tree


def remove_namespace(tree):
	"""Strip namespace from parsed XML."""
	for node in tree.iter():
		try:
			has_namespace = node.tag.startswith('{')
		except AttributeError:
			continue
		if has_namespace:
			node.tag = node.tag.split('}', 1)[1]


def stringify_children(node):
	"""Read and stringify the children of each nxml node."""
	section_parts = []
	for ch in node.getchildren():
		string_text = ''
		ch_tag = ch.tag
		if ((ch_tag == 'title') or (ch_tag == 'p')):
			sec_tree = ch.xpath("text()")
			for txt in sec_tree:
				txt = txt.rstrip()
				if len(txt) > 1:
					if ch_tag == 'title': #headers
						string_text += ' ' + TITLE_START \
						 + txt.rstrip('(') \
						 	.lstrip(')') \
						 	.rstrip('[') \
						 	.lstrip(']') + TITLE_END
					else: # body
						string_text += ' ' + txt.rstrip('(') \
							.lstrip(')') \
							.rstrip('[') \
							.lstrip(']')
			section_parts.append(string_text)
		else: # Start recursion
			string_text += ' ' + stringify_children(ch)
			section_parts.append(string_text)
	return ' '.join(filter(None, section_parts))


def process_text():
	"""UDF wrapper for process_text_"""
	def process_text_(text):
		"""Processing for string columns."""
		text = text.encode('utf8')
		try:
			processed_text = word_tokenize(text.decode('utf8'))
			processed_text = [wrd.lower() for wrd in processed_text]
		except:
			processed_text = None
			return processed_text

		processed_text = ' '.join(processed_text)
		# Those are split by tokenizer
		processed_text = re.sub('< sec >', '<sec>', processed_text)
		processed_text = re.sub('< /sec >', '</sec>', processed_text)
		processed_text = re.sub('< s >', '<s>', processed_text)
		processed_text = re.sub('< /s >', '</s>', processed_text)
		processed_text = re.sub('< t >', '<t>', processed_text)
		processed_text = re.sub('< /t >', '</t>', processed_text)

		return processed_text

	return F.udf(process_text_, spark_types.StringType())


def split_data(df):
	"""Three-way split data."""
	train_df, test_df = df.randomSplit([0.9, 0.1], seed=45)
	val_df, test_df = test_df.randomSplit([0.5, 0.5], seed=45)

	return train_df, val_df, test_dfs


def create_vocab(df):
	"""Create a vocabulary from a dataframe.
	Also removes some special tokens.
	
	Args:
		df: A dataframe with columns'processed_abstract'
		 and 'processed_full_text'

	Return:
		vocab: A wordlist sorted by frequency
	"""
	concat_udf = F.udf(
		lambda cols: " ".join([col for col in cols]),
		spark_types.StringType())
	df = df.withColumn(
		'all_text',
		concat_udf(F.array(
			'processed_abstract',
			'processed_full_text')))
	tokenizer = ml_feature.Tokenizer(
		inputCol='all_text',
		outputCol='tokens')
	df = tokenizer.transform(df)
	cv = ml_feature.CountVectorizer(
		inputCol='tokens',
		outputCol='vectors',
		vocabSize=200000)
	cv_model = cv.fit(df)

	# wrd_list is sorted by frequency
	vocab = cv_model.vocabulary
	vocab.remove(SENT_START)
	vocab.remove(SENT_END)
	vocab.remove(SEC_START)
	vocab.remove(SEC_END)

	return vocab


def write_ids(
		df,
		out_path,
		flag):
	"""Write pmc ids into a text file.

	Args:
		df: A dataframe with column 'pmc_id'
		out_path: The path to write the ids
		flag: One of 'train', 'val', 'test'
	"""	
	id_file = os.path.join(out_path, flag + '_ids.txt')
	with open(id_file, 'w') as writer:
		for pmcid in df.select('pmc_id').collect():
			try:
				writer.write('pmc' + pmcid.pmc_id + '\n')
			except:
				pass
	return


def write_vocab(vocab, out_path):
	"""Write the vocab to text file.
	
	Args:
		vocab: A list of tokens sorted by frequency
		out_path: The path to write the vocab
	"""
	with open(os.path.join(out_path, 'vocab'), 'w') as writer:
		for i, word in enumerate(vocab):
			writer.write(word.encode('utf8') + ' ' + str(i) + '\n')
	tf.logging.info('Finished writing vocabulary')

	return


def write_bin(output_fname):
	"""Partition function wrapper for write_bin."""
	def write_bin_(part_idx, partition):
		"""Write a partition into a .bin file in tf.exmple format."""
		partition_data = list(partition)
		num_items = len(partition_data)
		if not partition_data:
			return partition

		output_file = output_fname + '-' + str(part_idx) + '.bin'
		with open(output_file, 'wb') as writer:
			for item in partition_data:
				abstract = item.processed_abstract.encode('utf8')
				article = item.processed_full_text.encode('utf8')

				tf_example = example_pb2.Example()
				tf_example \
					.features \
					.feature['article'] \
					.bytes_list.value.extend([article])
				tf_example \
					.features \
					.feature['abstract'] \
					.bytes_list.value.extend([abstract])
				tf_example_str = tf_example.SerializeToString()
				str_len = len(tf_example_str)
				writer.write(strc.pack('q', str_len))
				writer.write(strc.pack('%ds' % str_len, tf_example_str))
		return partition

	return write_bin_


if __name__ == '__main__':

	logging.basicConfig(
		filename='data_preparation.log',
		level=logging.INFO)

	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--input_path',
		'-i',
		help='Path to the data files')
	parser.add_argument(
		'--output_path',
		'-o',
		help='Path to write output')
	parser.add_argument(
		'--num_partitions',
		'-np',
		help='The number of spark partitions (default is 1000)',
		default=1000)

	args = parser.parse_args()
	input_path = args.input_path
	output_path = args.output_path
	num_partitions = int(args.num_partitions)

	output_data_path = os.path.join(output_path, 'chunked')
	output_data_path = os.path.join(output_path, 'chunked')
	output_train_path = os.path.join(output_data_path, 'train')
	output_val_path = os.path.join(output_data_path, 'val')
	output_test_path = os.path.join(output_data_path, 'test')
	if not os.path.exists(output_path):
		os.makedirs(output_path)
		os.makedirs(output_data_path)
		os.makedirs(output_train_path)
		os.makedirs(output_val_path)
		os.makedirs(output_test_path)

	tf.logging.info('Creating filelist...')
	filelist_file = os.path.join(output_path, 'filelist.txt')
	create_filelist(input_path, filelist_file)
	sc = pyspark.SparkContext()
	spark = pyspark.sql.SparkSession(sc)

	tf.logging.info('Reading data files...')
	df = read_files(sc, filelist_file, num_partitions) \
		.repartition(num_partitions)

	proc_df = process_data(df)
	proc_df = proc_df.repartition(num_partitions) \
		.persist(pyspark.StorageLevel.MEMORY_AND_DISK_SER)

	train_df, val_df, test_df = split_data(proc_df)

	vocab = create_vocab(train_df)
	tf.logging.info('Vocabulary size: %d' % len(vocab))
	write_vocab(vocab, output_path)

	write_ids(train_df, output_path, 'train')
	write_ids(val_df, output_path, 'val')
	write_ids(test_df, output_path, 'test')

	# This is hacky. Since write_bin is a partition function it will need an
	# action to trigger execution. 
	# We trigger the execution of write_bin by calling the count action.
	train_df.repartition(num_partitions) \
		.rdd.mapPartitionsWithIndex(
			write_bin(os.path.join(output_train_path, "train"))) \
		.count()
	test_df.repartition(num_partitions) \
		.rdd.mapPartitionsWithIndex(
			write_bin(os.path.join(output_test_path, "test"))) \
		.count()
	val_df.repartition(num_partitions) \
		.rdd.mapPartitionsWithIndex(
			write_bin(os.path.join(output_val_path, "val"))) \
		.count()