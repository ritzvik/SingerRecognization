#REF : https://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/
#REF : https://docs.python.org/2/library/multiprocessing.html
#REF : https://www.tensorflow.org/serving/serving_basic

import sys
import glob
import os
import librosa
# import librosa.display
import numpy as np 
#import tensorflow as tf 
#import sklearn
from multiprocessing import Process, Queue, Lock
import pickle

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
	#
	f=open('objs.pkl','wb')
	pickle.dump([sr_global,singer_names,tr_features,tr_labels,ts_features,ts_labels],f)
	f.close()


if __name__=='__main__':
	main()
	#