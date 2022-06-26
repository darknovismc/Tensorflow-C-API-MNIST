# Tensorflow-C-API-MNIST
Console application written in C++ in Visual Studio 2010. 
It uses tensorflow C API to learn from MNIST handwritten 60000 image databaseand then predict 10000 images category.

Usage:
1.Download the database itself from : http://yann.lecun.com/exdb/mnist/ and extract next to the project
2.Install Tensorflow C api (tested on versions 2.8.0 and 2.9.0) either CPU  or a GPU versions.
3.Copy tensorflow.lib file to Microsoft Visual Studio\VC\lib\amd64 
4.Copy the tensorflow.dll to where the .exe is
5.Build and run project in any Visual Studio. Make sure you are building x64 application.

Files in project:
modelCompat.py - model source definition in Python.Exectute the file to get .graph.pb file.You can test different models like LSTMs, CNNs etc.
graph.pb - model's graph itself.Read by tensorflow API and trained by you.
TrainingData.cpp - C++ helper class used to encapsulate reading the training and testing data from 4 MNIST files
TensorflowModel.cpp - C++ class encapsulating all Tensorflow functionality. 
void TensorflowModel::Train(const float*inputData,const float*targetData,int batch_size)  function for batch training
void TensorflowModel::Predict(const float*inputData,const float*targetData) to feedforward the data
void TensorflowModel::Checkpoint(int type) to save or load model weights
TensorFlowTest.cpp - main console application to test all the functionality
