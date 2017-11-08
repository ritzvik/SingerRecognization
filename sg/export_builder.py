#REF : https://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/
#REF : https://docs.python.org/2/library/multiprocessing.html
#REF : https://www.tensorflow.org/serving/serving_basic

import sys
import glob
import os
import librosa
# import librosa.display
import numpy as np 
import tensorflow as tf 
import sklearn
from multiprocessing import Process, Queue, Lock

def extract_feature(file_name, srate):
	print(file_name)
	X, sample_rate = librosa.load(file_name,sr=srate)
	stft = np.abs(librosa.stft(X))
	mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
	chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
	mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
	contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate,fmin=10).T,axis=0)
	tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)
	return mfccs,chroma,mel,contrast,tonnetz,get_name(file_name)

def segregate():
	train_files =[]
	test_files =[]
	for i,fname in enumerate(os.listdir()):
		if fname.endswith('.wav'):
			if fname[:2]=='t-':
				if fname[-5]=='t':
					test_files.append(fname)
			else:
				train_files.append(fname)
	return train_files,test_files

def get_name(file_name):
	if file_name[:2]=='t-':
		return file_name[2:].split('_')[0]
	else:
		return file_name.split('_')[0]


def return_singer_index(singer_name,singer_names):
	if singer_name in singer_names:
		return singer_names.index(singer_name)
	else:
		singer_names.append(singer_name)
		return singer_names.index(singer_name)

def parse_extension(fname, srate, out_q, lock):
	lock.acquire()
	mfccs,chroma,mel,contrast,tonnetz,sname = extract_feature(fname, srate)
	out_q.put([mfccs,chroma,mel,contrast,tonnetz,sname])
	lock.release()

def parse(file_names_list, srate, n_threads, singer_names):
	out_q = Queue()
	features, labels = np.empty((0,193)), np.empty(0)
	locks = [Lock() for i in range(0,n_threads)]
	#
	for i,fname in enumerate(file_names_list):
		Process(target=parse_extension, args=(fname,srate,out_q,locks[i%n_threads])).start()
	#
	for i in range(0,n_threads):
		locks[i].acquire(block=True, timeout=3.0)
	#
	print("~~~")
	for i,fname in enumerate(file_names_list):
		tmp = out_q.get()
		#mfccs,chroma,mel,contrast,tonnetz,sname = tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],tmp[5]
		ext_features = np.hstack(tmp[:-1])
		features = np.vstack([features,ext_features])
		labels = np.append(labels,return_singer_index(tmp[5],singer_names))
	return np.array(features), np.array(labels, dtype=np.int)

def one_hot_encode(labels):
	n_labels = len(labels)
	n_unique_labels = len(np.unique(labels))
	one_hot_encode = np.zeros((n_labels,n_unique_labels))
	one_hot_encode[np.arange(n_labels), labels] = 1
	return one_hot_encode

def exportAdditionalData(singer_names,n_hidden_units_i):
	#
	F=open('trained/additional.txt','w')
	F.write(str(len(singer_names))+'\n')
	for singer_name in singer_names:
		F.write(singer_name+'\n')
	F.write(str(len(n_hidden_units_i)-1)+'\n')
	for units in n_hidden_units_i[1:]:
		F.write(str(units)+'\n')


def main():
	sr_global = 18000
	train_files,test_files = segregate()
	#
	singer_names = []
	threads = int(input('No of threads : '))
	#
	#
	ts_features, ts_labels = parse(test_files,sr_global,threads,singer_names)
	tr_features, tr_labels = parse(train_files,sr_global,threads,singer_names)
	#
	ts_labels = one_hot_encode(ts_labels)
	tr_labels = one_hot_encode(tr_labels)
	#print(ts_labels); print(tr_labels);
	#
	training_epochs = 1999
	n_dim = tr_features.shape[1]
	n_classes = len(singer_names)
	n_hidden_layers = int(input('Give no of Hidden Layers : '))
	n_hidden_units_i = [n_dim]

	for i in range(0,n_hidden_layers):
		n_hidden_units_i.append(int(input('Units for Layer %d : '%(i))))

	sd = 1/np.sqrt(n_dim)
	learning_rate = 0.01



	X = tf.placeholder(dtype=tf.float32,shape=[None,n_dim],name='X')
	Y = tf.placeholder(dtype=tf.float32,shape=[None,n_classes],name='Y')


	ht = X
	#hiddenLayerVars = []

	for i in range(0,n_hidden_layers):
		W_i = tf.Variable(tf.random_normal([n_hidden_units_i[i],n_hidden_units_i[i+1]], mean = 0, stddev=sd),name='Wh_%d'%(i))
		b_i = tf.Variable(tf.random_normal([n_hidden_units_i[i+1]], mean = 0, stddev=sd),name='bh_%d'%(i))
		if i%2:
			h_i = tf.nn.sigmoid(tf.matmul(ht,W_i) + b_i,name='hh_%d'%(i))
		else:
			h_i = tf.nn.tanh(tf.matmul(ht,W_i) + b_i,name='hh_%d'%(i))
		#hiddenLayerVars.append([W_i,b_i,h_i])
		ht = h_i

	# W = tf.Variable(tf.random_normal([n_hidden_units_three,n_classes], mean = 0, stddev=sd))
	W = tf.Variable(tf.random_normal([n_hidden_units_i[n_hidden_layers],n_classes], mean = 0, stddev=sd),name='W')
	b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd),name='b')
	#y_ = tf.nn.softmax(tf.matmul(h_3,W) + b)
	y_ = tf.nn.softmax(tf.matmul(ht,W) + b,name='y_')

	init = tf.initialize_all_variables()


	cost_function = -tf.reduce_sum(Y * tf.log(y_))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

	correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	saver = tf.train.Saver()


	cost_history = np.empty(shape=[1],dtype=float)
	y_true, y_pred = None, None
	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(training_epochs):
			_,cost = sess.run([optimizer,cost_function],feed_dict={X:tr_features,Y:tr_labels})
			cost_history = np.append(cost_history,cost)
		#
		saver.save(sess, 'trained/model')
		y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: ts_features})
		y_true = sess.run(tf.argmax(ts_labels,1))
		print('Test accuracy: ',round(sess.run(accuracy, feed_dict={X: ts_features, Y: ts_labels}) , 3))

	exportAdditionalData(singer_names,n_hidden_units_i)

	relation =  [[singer_names[i],i] for i in range(0,len(singer_names))]
	print (relation)
	print (y_true)
	print (y_pred)
	p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average='micro')
	print ("F-Score:", round(f,3))
	#
	#

if __name__=='__main__':
	main()
	#