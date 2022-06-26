#include "stdafx.h"
#include "TensorflowModel.h"
#include <iostream>
#include <fstream>

TensorflowModel::TensorflowModel():guessed(0)
{
	std::cout << "TensorFlow Version: " << TF_Version() << std::endl;
	status = TF_NewStatus();
	graph = TF_NewGraph();
	{
		// Create the session.
		TF_SessionOptions* opts = TF_NewSessionOptions();
		session = TF_NewSession(graph, opts, status);
		TF_DeleteSessionOptions(opts);
		Okay(); 
		// Import the graph.
		TF_Buffer* graph_def = ReadFile("graph.pb");
		if (graph_def == NULL) 
			throw std::runtime_error("Error loading graph file!\n");
		std::cout << "Read model graph of size " << graph_def->length << " bytes\n";
		TF_ImportGraphDefOptions* optsGraph = TF_NewImportGraphDefOptions();
		TF_GraphImportGraphDef(graph, graph_def, optsGraph, status);
		TF_DeleteImportGraphDefOptions(optsGraph);
		TF_DeleteBuffer(graph_def);
		Okay();
	}
	// Handles to the interesting operations in the graph.
	input.oper = TF_GraphOperationByName(graph, "input");
	input.index = 0;
	hidden.oper = TF_GraphOperationByName(graph, "hidden");
	hidden.index = 0;
	target.oper = TF_GraphOperationByName(graph, "target");
	target.index = 0;
	output.oper = TF_GraphOperationByName(graph, "output");
	output.index = 0;
	int64_t dims[2];
	TF_GraphGetTensorShape(graph,input,dims,2,status);
	Okay();
	inputDataSize = (int)dims[1];
	TF_GraphGetTensorShape(graph,output,dims,2,status);
	Okay();
	targetDataSize = (int)dims[1];

	init_op = TF_GraphOperationByName(graph, "init");
	train_op = TF_GraphOperationByName(graph, "train");
	save_op = TF_GraphOperationByName(graph, "save/control_dependency");
	restore_op = TF_GraphOperationByName(graph, "save/restore_all");
	checkpoint_file.oper = TF_GraphOperationByName(graph, "save/Const");
	checkpoint_file.index = 0;
}

TensorflowModel::~TensorflowModel()
{
	TF_DeleteSession(session, status);
	TF_DeleteGraph(graph);
	TF_DeleteStatus(status);
}

TF_Buffer* TensorflowModel::ReadFile(const char* filename) 
{
	std::ifstream file(filename,std::ios::binary);

	if(!file.is_open())
		throw std::runtime_error("File not opened!\n");
	file.seekg(0,std::ios::end);
	std::streamoff size = file.tellg();       
    file.seekg(0,std::ios::beg);
	if(size<=0)
		throw std::runtime_error("Wrong file size!\n");

	char* buf = new char[(unsigned int)size+1];
	file.read(buf,size);
    buf[size] = NULL;
	file.close();

	TF_Buffer* ret = TF_NewBufferFromString(buf, size);
	delete[] buf;
	return ret;
}

TF_Tensor* TensorflowModel::ScalarStringTensor(const char* str, TF_Status* status) 
{
	size_t nbytes = 8 + strlen(str);
	TF_Tensor* t = TF_AllocateTensor(TF_STRING, NULL, 0, nbytes);
	void* data = TF_TensorData(t);
	memset(data, 0, 8);
	return t;
}

void TensorflowModel::Init() 
{
	std::cout << ("Initializing model weights\n");
	TF_SessionRun(session, NULL,
		/* No inputs */
		NULL, NULL, 0,
		/* No outputs */
		NULL, NULL, 0,
		&init_op, 1,
		NULL, status);
	Okay();
}

void TensorflowModel::Checkpoint(int type) 
{
	const char* checkpoint_prefix = "./checkpoints/checkpoint";
	TF_Tensor* t = ScalarStringTensor(checkpoint_prefix, status);
	if(type == RESTORE)
		std::cout << ("Restoring weights from checkpoint\n");
	else
		std::cout << ("Saving checkpoint\n");
	if (TF_GetCode(status) != TF_OK) 
	{
		TF_DeleteTensor(t);
		throw std::runtime_error("ERROR: " + std::string(TF_Message(status)));
	}
	TF_Output inputs[1] = {checkpoint_file};
	TF_Tensor* input_values[1] = {t};
	const TF_Operation* op[1] = {type == SAVE ? save_op
		: restore_op};
	TF_SessionRun(session, NULL, inputs, input_values, 1,
		/* No outputs */
		NULL, NULL, 0,
		/* The operation */
		op, 1, NULL, status);
	TF_DeleteTensor(t);
	Okay();
}

void TensorflowModel::Predict(const float*inputData,const float*targetData) 
{
	// batch consists of 4 2x1 matrices.
	const int64_t dims[2] = {1, inputDataSize};
	const size_t nbytes = dims[0]*dims[1] * sizeof(float);
	TF_Tensor* t = TF_AllocateTensor(TF_FLOAT, dims, 2, nbytes);
	memcpy(TF_TensorData(t), inputData, nbytes);
	TF_Output inputs[1] = {input};
	TF_Tensor* input_values[1] = {t};
	TF_Output outputs[1] = {output};
	TF_Tensor* output_values[1] = {NULL};

	TF_SessionRun(session, NULL, inputs, input_values, 1, outputs,
		output_values, 1,
		/* No target operations to run */
		NULL, 0, NULL, status);
	TF_DeleteTensor(t);
	Okay();
	int outputTensorSize = targetDataSize*sizeof(float);
	if (TF_TensorByteSize(output_values[0]) != outputTensorSize) 
	{
		std::string tensorSize = std::to_string((long long)TF_TensorByteSize(output_values[0]));
		TF_DeleteTensor(output_values[0]);
		throw std::runtime_error(std::string("ERROR: Expected predictions tensor to have ") + std::to_string((long long)outputTensorSize)  + std::string(" bytes, but it has ") + tensorSize + std::string(" bytes.\n"));
	}
	float* predictions = new float[targetDataSize];
	memcpy(predictions, TF_TensorData(output_values[0]), outputTensorSize);
	TF_DeleteTensor(output_values[0]);

	int idx1 =0;
	while(targetData[idx1]==0)
		idx1++;
	int idx2 =0;
	float max = predictions[0];
	for(int i=0;i<targetDataSize;i++)
	{
		if(predictions[i]>max)
		{
			max = predictions[i];
			idx2 = i;
		}
	}
	if(idx1==idx2)
		guessed++;
	delete[] predictions;
}

void TensorflowModel::Train(const float*inputData,const float*targetData,int batch_size) 
{
	TF_Tensor *x, *y;
	const int64_t dims1[] = {batch_size, inputDataSize};
	size_t nbytes1 = dims1[0]*dims1[1] * sizeof(float);
	const int64_t dims2[] = {batch_size, targetDataSize};
	size_t nbytes2 = dims2[0]*dims2[1] * sizeof(float);
	x = TF_AllocateTensor(TF_FLOAT, dims1, 2, nbytes1);
	y = TF_AllocateTensor(TF_FLOAT, dims2, 2, nbytes2);
	memcpy(TF_TensorData(x), inputData, nbytes1);
	memcpy(TF_TensorData(y), targetData, nbytes2);

	TF_Output inputs[2] = {input, target};
	TF_Tensor* input_values[2] = {x, y};
	TF_SessionRun(session, NULL, inputs, input_values, 2,
		/* No outputs */
		NULL, NULL, 0, &train_op, 1, NULL, status);
	TF_DeleteTensor(x);
	TF_DeleteTensor(y);
	Okay();
}