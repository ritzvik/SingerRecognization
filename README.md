# SingerRecognization

This Project aims to recognize singer from audio clips of approximately 12 seconds.

The Training and Test WAV files should reside in the same folder as the .py files
The Project uses TensorFlow and Librosa libraries
Much of the audio processing code is took from https://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/

There are two .py files:

  recog.py --> vanilla version
  
  multirecog.py --> multithreaded version (significantly faster)
  
  
How To Run ??

  -> Make sure all the audio files are in the same folder as the .py files.
  
  -> Training Set files should have \<person name\>_i.wav format
  
  -> Test Set files should have t-\<person name\>_i.wav format
  
  
