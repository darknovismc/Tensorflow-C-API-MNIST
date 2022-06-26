#pragma once
#include "tensorflow/c/c_api.h"
#include <string>
class TensorflowModel
{
private:
	int guessed;
	int inputDataSize;
	int targetDataSize;
	TF_Graph* graph;
	TF_Session* session;
	TF_Status* status;
	TF_Output input,hidden,target, output;
	TF_Operation *init_op, *train_op, *save_op, *restore_op;
	TF_Output checkpoint_file;
	enum SaveOrRestore { SAVE, RESTORE };
	void Okay(){if (TF_GetCode(status) != TF_OK) throw std::runtime_error("ERROR: " + std::string(TF_Message(status)));}
	TF_Buffer* ReadFile(const char* filename);
	TF_Tensor* ScalarStringTensor(const char* data, TF_Status* status);
	void Checkpoint(int type);
public:
	TensorflowModel();
	~TensorflowModel();
	void PrintStats(){std::cout << "Guessed:" << guessed << std::endl;}
	void Init();
	void Predict(const float*inputData,const float*targetData);
	void Train(const float*inputData,const float*targetData,int batch_size);
	void Save(){Checkpoint(SAVE);}
	void Load(){Checkpoint(RESTORE);}
};