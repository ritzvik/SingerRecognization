# SingerRecognization

This Project aims to recognize singer from audio clips of approximately 12 seconds.

The Training and Test WAV files should reside in the same folder as the .py files
The Project uses TensorFlow and Librosa libraries
Much of the audio processing code is took from https://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/

There are four .py files:

  recog.py --> vanilla version\n
  multirecog.py --> multithreaded version (significantly faster)\n
  export_builder.py --> multithreaded version (make model and export)\n
  import_learner.py --> import saved model and run on test data
  
How To Run (Train real time and test)??

  -> Make sure all the audio files are in the same folder as the .py files.
  
  -> Training Set files should have \<person name\>_i.wav format
  
  -> Test Set files should have t-\<person name\>_i.wav format
  
  
How to export model ??

  -> Run export_builder.py. It will train on files of format \<person name\>_i.wav and run the model on files of format t-\<person name\>_it.wav format.
  
  -> Run import_builder.py. It will run the files of format t-\<person name\>_i.wav on exported model.
