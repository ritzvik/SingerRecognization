import sys
import glob
import os
import librosa
import numpy as np 
import tensorflow as tf 
import sklearn
from multiprocessing import Process, Queue, Lock
from export_builder import parse, one_hot_encode

def importAdditionalData():
	singer_names = []
	F=open('trained/additional.txt')
	n=int(F.readline()[:-1])
	for i in range(0,n):
		singer_names.append(F.readline()[:-1])
	n=int(F.readline()[:-1])
	n_hidden_units_i = []
	for i in range(0,n):
		n_hidden_units_i.append(int(F.readline()[:-1]))
	#
	return singer_names,n_hidden_units_i

def getTestFiles():
	test_files = []
	for i,fname in enumerate(os.listdir()):
		if fname.startswith('t-') and fname.endswith('.wav'):
			test_files.append(fname)
	#
	return test_files

def singer_index(singer_name,singer_names):
	return singer_names.index(singer_name)


def mainprog():
	#
	#
	singer_names, n_hidden_units_i = importAdditionalData()
	sr_global =18000
	threads = int(input('No of threads : '))
	test_files=getTestFiles()
	#
	ts_features, ts_labels = parse(test_files,sr_global,threads,singer_names)
	ts_labels = one_hot_encode(ts_labels)
	#
	#
	#
	#
	n_dim = ts_features.shape[1]
	n_classes = len(singer_names)
	n_hidden_layers = len(n_hidden_units_i)
	n_hidden_units_i.insert(0,n_dim)
	sd = 1/np.sqrt(n_dim)
	#
	#
	# X = tf.placeholder(dtype=tf.float32,shape=[None,n_dim])
	# Y = tf.placeholder(dtype=tf.float32,shape=[None,n_classes])
	#
	#
	sess = tf.Session()
	saver = tf.train.import_meta_graph('trained/model.meta')
	saver.restore(sess,'trained/model')
	#
	graph=tf.get_default_graph()
	X = graph.get_tensor_by_name('X:0')
	Y = graph.get_tensor_by_name('Y:0')
	op = graph.get_tensor_by_name('y_:0')
	#
	y_pred = sess.run(tf.argmax(op,1), feed_dict={X: ts_features})
	y_true = sess.run(tf.argmax(ts_labels,1))
	#
	#
	#
	#
	relation =  [[singer_names[i],i] for i in range(0,len(singer_names))]
	print(relation)
	print(y_true)
	print(y_pred)
	p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average='micro')
	print ("F-Score:", round(f,3))
	#
	#

if __name__=='__main__':
	mainprog()