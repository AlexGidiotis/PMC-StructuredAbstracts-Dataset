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

from pyspark.sql import SparkSession, Row
from pyspark import SparkContext, StorageLevel
from pyspark.sql import functios as F
from pyspark.sql import types as spark_types
from pyspark.ml.feature import CountVectorizer, Tokenizer

from nltk import word_tokenize, sent_tokenize

import tensorflow as tf
from tensorflow.core.example import example_pb2

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

TITLE_START = '<t>'
TITLE_END = '</t>'

SECTION_START = '<sec>'
SECTION_END = '</sec>'

KEYWORDS = {
	'introduction': 'i',
	'case': 'i',
	'purpose': 'i',
	'objective': 'i',
	'objectives': 'i',
	'aim': 'i',
	'summary': 'i',
	'findings': 'l',
	'background': 'i',
	'background/aims': 'i',
	'literature': 'l',
	'studies': 'l',
	'methods': 'm',
	'method': 'm',
	'techniques': 'm',
	'methodology': 'm',
	'results': 'r',
	'result': 'r',
	'experiment': 'r',
	'experiments': 'r',
	'experimental': 'r',
	'discussion': 'd',
	'limitations': 'd',
	'conclusion': 'c',
	'conclusions': 'c',
	'concluding': 'c'
}


def remove_namespace(tree):
	"""
	Strip namespace from parsed XML
	"""
	for node in tree.iter():
		try:
			has_namespace = node.tag.startswith('{')
		except AttributeError:
			continue
		if has_namespace:
			node.tag = node.tag.split('}', 1)[1]


def read_xml(path, nxml=False):
	"""
	Parse tree from given XML path
	"""
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


def stringify_children(node):
	"""
	Recursively read the section headers and text into string.
	"""
	parts = []
	for ch in node.getchildren():
		text = ''
		if (ch.tag == 'title') or (ch.tag == 'p'):
			for t in ch.xpath("text()"):
				t = t.rstrip()
				if len(t) > 1:
					if ch.tag == 'title':
						text += ' ' + TITLE_START \
						 + t.rstrip('(') \
						 	.lstrip(')') \
						 	.rstrip('[') \
						 	.lstrip(']') + TITLE_END
					else:
						text += ' ' + t.rstrip('(') \
							.lstrip(')') \
							.rstrip('[') \
							.lstrip(']')
			parts.append(text)
		else:
			text += ' ' + stringify_children(ch)
			parts.append(text)
	return ' '.join(filter(None, parts))


def create_filelist(in_path, out_path):
	"""Create a list of files to read"""
	files_read = 0
	nxml_sign = re.compile('\.nxml$')
	file_list = os.path.join(out_path, 'filelist.txt')
	with open(file_list, 'w') as of:
		for path, dirs, files in os.walk(in_path):
			for f in files:
				if nxml_sign.search(f) is not None:
					file_name = path + '/' + f
					files_read += 1
					of.write(file_name)
					of.write('\n')

	return file_list


def read_data(sc, file_list, num_partitions):
	"""Read the filelist into a spark dataframe."""
	path_rdd = sc.textFile(file_list, minPartitions=num_partitions)
	df = path_rdd.map(
			lambda x: Row(
				file_name=os.path.basename(x),
				**read_file(x))) \
		.toDF() \
		.where((F.col('abstract').isNotNull()) \
			& (F.col('pmc_id').isNotNull()) \
			& (F.col('full_text').isNotNull()))

	return df


def read_file(file_path):
	"""Read a data file and extract abstract, full text and pmc id."""
	# Skip bad files.
	try:
		tree = read_xml(file_path, nxml=True)
	except:
		return_dict = {
			'pmc_id': None,
			'abstract': None,
			'full_text': None
		}

		return return_dict
	# Get the pmc id
	try:
		pmid_tree = tree.xpath('.//article-id')
		for ar_id in pmid_tree:
			if ar_id.attrib['pub-id-type'] == 'pmc':
				pmcid = str(ar_id.xpath("text()")[0])
	except:
		pmcid = None

	# Skip bad abstracts.
	try:
		# Read abstract
		abstracts = []
		abstract_tree = tree.findall('.//abstract')[0]
		if len(abstract_tree) > 1:
			for a in abstract_tree:
				abs_section_text = stringify_children(a)
				if len(abs_section_text) < 3:
					continue
				abs_section_text = re.sub('\s+', ' ', abs_section_text) \
					.replace('\n', ' ') \
					.replace('\t', ' ') \
					.strip()
				header = re.findall('<t>(.*?)<\/t>', abs_section_text)
				# Check header for sec id.
				abs_sec_id = 'o'
				for head in header:
					for wrd in head.split():
						try:
							abs_sec_id = KEYWORDS[wrd.lower()]
							break
						except:
							continue

				section_id = '<' + abs_sec_id + '>'

				# Split into sentences sentences
				abs_sents = sent_tokenize(abs_section_text)
				abs_proc_text = []
				for sent in abs_sents:
					proc_sent = SENTENCE_START + sent \
					 + ' ' + SENTENCE_END
					abs_proc_text.append(proc_sent)
				abs_proc_text = ' '.join(abs_proc_text)
				abs_section = SECTION_START + section_id \
				 + abs_proc_text + SECTION_END
				abstracts.append(abs_section)
			if len(abstracts) > 0:
				abstract = ' '.join(abstracts)
				# We don't want bad abstracts
				if len(abstract) < 5:
					abstract = None
			else:
				abstract = None
		else:
			abstract = None
	except:
		abstract = None

	# Skip bad full texts.
	try:
		# Read body
		body = tree.xpath('.//body')[0]
		parts = []
		for i, part in enumerate(body):
			section_text = stringify_children(part)
			section_text = re.sub('\s+', ' ', section_text)
			# Discard short sections
			if len(section_text.split()) > 5:
				header = re.findall('<t>(.*?)<\/t>', section_text)
				# Check header for sec id.
				sec_id = 'o'
				for head in header:
					for wrd in head.split():
						try:
							sec_id = KEYWORDS[wrd.lower()]
							break
						except:
							continue

				section_id = '<' + sec_id + '>'
				section = SECTION_START + section_id \
				 + section_text + ' ' + SECTION_END
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

	return_dict = {
		'pmc_id': pmcid,
		'abstract': abstract,
		'full_text': full_text
	}

	return return_dict


def process_text():
	"""UDF wrapper for process_text_"""
	def process_text_(text):
		"""Tokenize, lowercase and process a string column."""
		text = text.encode('utf8')
		try:
			proc_text = word_tokenize(text.decode('utf8'))
			proc_text = [wrd.lower() for wrd in proc_text]
		except:
			proc_text = None
			return proc_text

		proc_text = ' '.join(proc_text)
		# Those are split during tokenization so we have to fix them
		proc_text = re.sub('< sec >', '<sec>', proc_text)
		proc_text = re.sub('< /sec >', '</sec>', proc_text)
		proc_text = re.sub('< s >', '<s>', proc_text)
		proc_text = re.sub('< /s >', '</s>', proc_text)
		proc_text = re.sub('< t >', '<t>', proc_text)
		proc_text = re.sub('< /t >', '</t>', proc_text)
		proc_text = re.sub(
			'< [ilmrdco] >',
			lambda m: '<' + m.group()[2] + '>',
			proc_text)

		return proc_text

	return F.udf(process_text_, spark_types.StringType())


def create_vocab(df):
	"""Create the vocanulary."""
	concat_udf = F.udf(
		lambda cols: " ".join([x for x in cols]),
		spark_types.StringType())
	df = df.withColumn(
		'all_text', concat_udf(F.array(
			'processed_abstract',
			'processed_full_text')))
	tokenizer = Tokenizer(
		inputCol='all_text',
		outputCol='tokens')
	df = tokenizer.transform(df)
	cv = CountVectorizer(
		inputCol='tokens',
		outputCol='vectors',
		vocabSize=200000)
	cv_model = cv.fit(df)
	# wrd_list is already sorted
	wrd_list = cv_model.vocabulary
	wrd_list.remove(SENTENCE_START)
	wrd_list.remove(SENTENCE_END)
	wrd_list.remove(SECTION_START)
	wrd_list.remove(SECTION_END)

	for k in set(KEYWORDS.values() + ['o']):
		sec_id = '<' + k + '>'
		try:
			wrd_list.remove(sec_id)
		except:
			pass

	return wrd_list


def write_vocab(wrd_list, out_path):
	"""Write the vocabulary."""
	tf.logging.info('Writing vocab file...')
	with open(os.path.join(out_path, 'vocab'), 'w') as writer:
		for i, word in enumerate(wrd_list):
			writer.write(word.encode('utf8') + ' ' + str(i) + '\n')
	tf.logging.info('Finished writing vocab file')

	return


def write_ids(df, out_path, flag):
	"""Write the pmc_ids to txt."""	
	with open(os.path.join(out_path, flag + '_ids.txt'), 'w') as writer:
		for id_item in df.select('pmc_id').collect():
			try:
				writer.write('pmc' + id_item.pmc_id + '\n')
			except:
				pass

	return


def write_bin(output_file_name):
	"""UDF wrapper for write_bin."""
	def write_bin_(part_idx, partition):
		"""Write each partition to a binary file."""
		partition_data = list(partition)
		num_items = len(partition_data)
		if num_items == 0:
			return partition

		output_file = output_file_name + '-' + str(part_idx) + '.bin'
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
		filename='data_prep_logs.log',
		level=logging.INFO)

	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--input_path',
		'-i',
		help='Path to the .nxml data files')
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
	file_list = create_filelist(
		input_path,
		output_path)
	sc = SparkContext()
	spark = SparkSession(sc)

	tf.logging.info('Reading data files...')
	proc_df = read_data(sc, file_list, num_partitions) \
		.repartition(num_partitions) \
		.withColumn('processed_abstract',
			process_text()('abstract')) \
		.withColumn('processed_full_text',
			process_text()('full_text')) \
		.where(
			(F.col('processed_full_text').isNotNull()) \
			& (F.col('processed_abstract').isNotNull()) \
			& (F.col('pmc_id').isNotNull())) \
		.select(
			'pmc_id',
			'processed_abstract',
			'processed_full_text') \
		.repartition(num_partitions) \
		.persist(StorageLevel.MEMORY_AND_DISK_SER)

	proc_df.show()

	train_df, test_df = proc_df.randomSplit([0.9, 0.1], seed=45)
	val_df, test_df = test_df.randomSplit([0.5, 0.5], seed=45)

	write_ids(train_df, output_path, 'train')
	write_ids(val_df, output_path, 'val')
	write_ids(test_df, output_path, 'test')

	wrd_list = create_vocab(train_df)
	tf.logging.info('Vocabulary size: %d' % len(wrd_list))
	write_vocab(wrd_list, output_path)

	train_df.repartition(num_partitions) \
		.rdd.mapPartitionsWithIndex(write_bin(os.path.join(output_train_path, "train"))) \
		.count()
	test_df.repartition(num_partitions) \
		.rdd.mapPartitionsWithIndex(write_bin(os.path.join(output_test_path, "test"))) \
		.count()
	val_df.repartition(num_partitions) \
		.rdd.mapPartitionsWithIndex(write_bin(os.path.join(output_val_path, "val"))) \
		.count()
