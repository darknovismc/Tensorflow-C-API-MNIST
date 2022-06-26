// TensorFlowTest.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include "TensorflowModel.h"
#include "TrainingData.h"

bool Prompt(const char* text)
{
	std::cout<<std::endl << text;
	char c =std::getchar();
	std::cin.clear();
	fflush(stdin);
	if(c=='y')
		return true;
	return false;
}
#define BATCH_SIZE 10
int main() 
{
	srand((unsigned int)time( NULL ));
	clock_t begin_time=0;
	try
	{
		TrainingData td;
		float inputData[28*28*BATCH_SIZE];
		float targetData[10*BATCH_SIZE];
		TensorflowModel tf;
		if(Prompt("Would you like to load the model ?:")) 
			tf.Load();
		else
			tf.Init();
		if(Prompt("Would you like to train the model ?:")) 
		{
			std::cout << "Started training.." << std::endl;
			for (int i = 0; i < td.TrainDataSize; i+=BATCH_SIZE) 
			{
				for(int j=0;j<BATCH_SIZE;j++)
					td.FillTrainData(&inputData[j*28*28],&targetData[j*10],i+j);
				tf.Train(inputData,targetData,BATCH_SIZE);
				if(float(clock() - begin_time) /  CLOCKS_PER_SEC>2)
				{
					begin_time = clock();
					std::cout <<"Progress: "<< i*100/(float)td.TrainDataSize<< "%"<< std::endl;
				}
			}
			std::cout << "Finished training."<<std::endl;
		}
		std::cout << "Testing the model with "<<td.TestDataSize <<" data." <<std::endl;
		for(int i=0;i<td.TestDataSize;i++)
		{
			td.FillTestData(inputData,targetData,(td.TestDataSize-1)*rand()/RAND_MAX);
			tf.Predict(inputData,targetData);
			if(float(clock() - begin_time) /  CLOCKS_PER_SEC>2)
			{
				begin_time = clock();
				std::cout <<"Progress: "<< i*100/(float)td.TestDataSize<< "%"<< std::endl;
			}
		}
		tf.PrintStats();
		if(Prompt("Would you like to save the model ?:")) 
			tf.Save();
	}
	catch(std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
	std::cout << ("Application finished.\n");
	std::getchar();
	return 0;
}