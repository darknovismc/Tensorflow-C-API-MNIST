#include "stdafx.h"
#include "TrainingData.h"
const int IMG_RES = 28;

void TrainingData::checkMagicNumber(std::unique_ptr<unsigned char[]>& data,unsigned short correct)
{
	int magicNumber = data[3] | data[2]<<8 | data[1]<<16 | data[0]<<24;
	if(magicNumber!=correct)
		throw std::runtime_error("Wrong file magic number !");
}

TrainingData::TrainingData()
{
	imagesTest=std::unique_ptr<unsigned char[]>(readDataFile("../../ImgDataSet/t10k-images.idx3-ubyte"));
	checkMagicNumber(imagesTest,0x803);
	labelsTest=std::unique_ptr<unsigned char[]>(readDataFile("../../ImgDataSet/t10k-labels.idx1-ubyte"));
	checkMagicNumber(labelsTest,0x801);
	imagesTrain=std::unique_ptr<unsigned char[]>(readDataFile("../../ImgDataSet/train-images.idx3-ubyte"));
	checkMagicNumber(imagesTrain,0x803);
	labelsTrain=std::unique_ptr<unsigned char[]>(readDataFile("../../ImgDataSet/train-labels.idx1-ubyte"));
	checkMagicNumber(labelsTrain,0x801);
	TrainDataSize = imagesTrain[7] | imagesTrain[6]<<8 | imagesTrain[5]<<16 | imagesTrain[4]<<24;
	TestDataSize =  imagesTest[7]  | imagesTest[6]<<8  | imagesTest[5]<<16  | imagesTest[4]<<24;
	//drawDigits();
}

unsigned char* TrainingData::readDataFile(const char * fileName)
{
	std::cout <<"Opening: " << fileName << std::endl;
	std::ifstream file(fileName,std::ios::binary);

	if(!file.is_open())
		throw std::runtime_error("File not opened!");
	file.seekg(0,std::ios::end);
	std::streamoff size = file.tellg();       
    file.seekg(0,std::ios::beg);

	char* buf = new char[(unsigned int)size+1];
	file.read(buf,size);
    buf[size] = NULL;
	file.close();
	return (unsigned char*)buf;
}

void TrainingData::drawDigits()
{
	system("mode CON: COLS=50");
	HWND console = GetConsoleWindow();
	HDC hDC = GetDC(console);
	for(int yNum=0;yNum<10;yNum++)
	{
		for(int xNum=0;xNum<10;xNum++)
		{
			for(int col=0;col<IMG_RES;col++)
			{
				for(int row=0;row<IMG_RES;row++)
				{
					unsigned char data = imagesTest[16+row+col*IMG_RES+xNum*IMG_RES*IMG_RES+yNum*10*IMG_RES*IMG_RES];
					SetPixel(hDC,xNum*IMG_RES+row,yNum*IMG_RES+col,RGB(data,data,data));
				}
			}
		}
	}
	for(int i=0;i<25;i++)
		std::cout << std::endl;
	for(int yNum=0;yNum<10;yNum++)
	{
		for(int xNum=0;xNum<10;xNum++)
			std::cout << (char)(labelsTest[xNum+yNum*10+8]+'0');
		std::cout << std::endl;
	}
}

void TrainingData::FillTestData(float* inputData,float* outputData,int imgNum)
{
	if(imgNum >= TestDataSize)
		throw std::runtime_error("Image number greater than train data size !");
	for(int col=0;col<IMG_RES;col++)
	{
		for(int row=0;row<IMG_RES;row++)
			inputData[row+col*IMG_RES]=imagesTest[16+row+col*IMG_RES+imgNum*IMG_RES*IMG_RES]/255.0f;
	}
	for(int i=0;i<10;i++)
		outputData[i]=0;
	outputData[labelsTest[imgNum+8]]=1.0f;
}

void TrainingData::FillTrainData(float* inputData,float* outputData,int imgNum)
{
	if(imgNum >= TrainDataSize)
		throw std::runtime_error("Image number greater than test data size !");
	for(int col=0;col<IMG_RES;col++)
	{
		for(int row=0;row<IMG_RES;row++)
			inputData[row+col*IMG_RES]=imagesTrain[16+row+col*IMG_RES+imgNum*IMG_RES*IMG_RES]/255.0f;
	}
	for(int i=0;i<10;i++)
		outputData[i]=0;
	outputData[labelsTrain[imgNum+8]]=1.0f;
}