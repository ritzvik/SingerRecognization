import sys
import glob
import os
import numpy as np
import tensorflow as tf 
import sklearn
from sklearn import metrics
#from multiprocessing import Process, Queue, Lock
import pickle


def exportAdditionalData(singer_names,n_hidden_units_i):
	#
	F=open('trained/additional.txt','w')
	F.write(str(len(singer_names))+'\n')
	for singer_name in singer_names:
		F.write(singer_name+'\n')
	F.write(str(len(n_hidden_units_i)-1)+'\n')
	for units in n_hidden_units_i[1:]:
		F.write(str(units)+'\n')
	F.close()



def main():
	varfile=open('objs.pkl','rb')
	sr_global,singer_names,tr_features,tr_labels,ts_features,ts_labels = pickle.load(varfile)
	varfile.close()
	#
	#
	training_epochs = 2000
	n_dim = tr_features.shape[1]
	n_classes = len(singer_names)
	n_hidden_layers = int(input('Give no of Hidden Layers : '))
	n_hidden_units_i = [n_dim]
	#
	for i in range(0,n_hidden_layers):
		n_hidden_units_i.append(int(input('Units for Layer %d : '%(i))))
	#
	sd = 1/np.sqrt(n_dim)
	learning_rate = 0.01
	#
	#
	#
	X = tf.placeholder(dtype=tf.float32,shape=[None,n_dim],name='X')
	Y = tf.placeholder(dtype=tf.float32,shape=[None,n_classes],name='Y')
	#
	#
	ht = X
	#hiddenLayerVars = []
	#
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
	p,r,f,s = metrics.precision_recall_fscore_support(y_true, y_pred, average='micro')
	print ("F-Score:", round(f,3))
	#
	#

if __name__=='__main__':
	main()
	#