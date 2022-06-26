#include <fstream>
#include <memory>
class TrainingData
{
private:
	std::unique_ptr<unsigned char[]> imagesTrain,labelsTrain,imagesTest,labelsTest;
	static unsigned char* readDataFile(const char * fileName);
	void checkMagicNumber(std::unique_ptr<unsigned char[]>& data,unsigned short correct);
	void drawDigits();
public:
	TrainingData();
	void FillTestData(float* inputData,float* outputData,int imgNum);
	void FillTrainData(float* inputData,float* outputData,int imgNum);
	int TrainDataSize,TestDataSize;
};