import glob
import os
import librosa
import librosa.display
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import tensorflow as tf 
import sklearn
from multiprocessing import Process, Queue, Lock

def extract_feature(file_name):
	print(file_name)
	X, sample_rate = librosa.load(file_name,sr=sr_global)
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
				test_files.append(fname)
			else:
				train_files.append(fname)
	return train_files,test_files

def get_name(file_name):
	if file_name[:2]=='t-':
		return file_name[2:].split('_')[0]
	else:
		return file_name.split('_')[0]


def return_singer_index(singer_name):
	global singers; global singer_names;
	if singer_name in singer_names:
		return singer_names.index(singer_name)
	else:
		singer_names.append(singer_name)
		singers += 1
		return singer_names.index(singer_name)

def parse_extension(fname, out_q, lock):
	lock.acquire()
	mfccs,chroma,mel,contrast,tonnetz,sname = extract_feature(fname)
	out_q.put([mfccs,chroma,mel,contrast,tonnetz,sname])
	lock.release()

def parse(file_names_list):
	global threads
	out_q = Queue()
	features, labels = np.empty((0,193)), np.empty(0)
	locks = [Lock() for i in range(0,threads)]
	#
	for i,fname in enumerate(file_names_list):
		Process(target=parse_extension, args=(fname,out_q,locks[i%threads])).start()
	#
	for i in range(0,threads):
		locks[i].acquire(block=True)
	#
	print("~~~")
	for i,fname in enumerate(file_names_list):
		tmp = out_q.get()
		#mfccs,chroma,mel,contrast,tonnetz,sname = tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],tmp[5]
		ext_features = np.hstack(tmp[:-1])
		features = np.vstack([features,ext_features])
		labels = np.append(labels,return_singer_index(tmp[5]))
	return np.array(features), np.array(labels, dtype=np.int)

def one_hot_encode(labels):
	n_labels = len(labels)
	n_unique_labels = len(np.unique(labels))
	one_hot_encode = np.zeros((n_labels,n_unique_labels))
	one_hot_encode[np.arange(n_labels), labels] = 1
	return one_hot_encode


if __name__=='__main__':
	sr_global = 18000
	
	train_files,test_files = segregate()
	singers = 0
	singer_names = []
	threads = int(input('No of threads : '))


	ts_features, ts_labels = parse(test_files)
	tr_features, tr_labels = parse(train_files)

	tr_labels = one_hot_encode(tr_labels)
	ts_labels = one_hot_encode(ts_labels)




	training_epochs = 2000
	n_dim = tr_features.shape[1]
	n_classes = len(singer_names)
	n_hidden_layers = int(input('Give no of Hidden Layers : '))
	n_hidden_units_i = [n_dim]

	for i in range(0,n_hidden_layers):
		n_hidden_units_i.append(int(input('Units for Layer %d : '%(i))))

	sd = 1/np.sqrt(n_dim)
	learning_rate = 0.01



	X = tf.placeholder(tf.float32,[None,n_dim])
	Y = tf.placeholder(tf.float32,[None,n_classes])


	ht = X

	for i in range(0,n_hidden_layers):
		W_i = tf.Variable(tf.random_normal([n_hidden_units_i[i],n_hidden_units_i[i+1]], mean = 0, stddev=sd))
		b_i = tf.Variable(tf.random_normal([n_hidden_units_i[i+1]], mean = 0, stddev=sd))
		if i%2:
			h_i = tf.nn.sigmoid(tf.matmul(ht,W_i) + b_i)
		else:
			h_i = tf.nn.tanh(tf.matmul(ht,W_i) + b_i)
		ht = h_i

	# W = tf.Variable(tf.random_normal([n_hidden_units_three,n_classes], mean = 0, stddev=sd))
	W = tf.Variable(tf.random_normal([n_hidden_units_i[n_hidden_layers],n_classes], mean = 0, stddev=sd))
	b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))
	#y_ = tf.nn.softmax(tf.matmul(h_3,W) + b)
	y_ = tf.nn.softmax(tf.matmul(ht,W) + b)

	init = tf.initialize_all_variables()


	cost_function = -tf.reduce_sum(Y * tf.log(y_))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

	correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




	cost_history = np.empty(shape=[1],dtype=float)
	y_true, y_pred = None, None
	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(training_epochs):
			_,cost = sess.run([optimizer,cost_function],feed_dict={X:tr_features,Y:tr_labels})
			cost_history = np.append(cost_history,cost)
		
		y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: ts_features})
		y_true = sess.run(tf.argmax(ts_labels,1))
		print('Test accuracy: ',round(sess.run(accuracy, feed_dict={X: ts_features, Y: ts_labels}) , 3))




	relation =  [[singer_names[i],i] for i in range(0,len(singer_names))]
	print (relation)
	print (y_true)
	print (y_pred)
	p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average='micro')
	print ("F-Score:", round(f,3))
