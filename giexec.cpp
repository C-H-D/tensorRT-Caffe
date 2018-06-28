/*
Dan 2018.6.27
*/ 
#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include <time.h>
#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvCaffeParser.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

#define CHECK(status)									\
{														\
	if (status != 0)									\
	{													\
		std::cout << "Cuda failure: " << status;		\
		abort();										\
	}													\
}

// stuff we know about the network and the caffe input/output blobs
static const int INPUT_D = 3;	//input 3 channels(BGR)
static const int INPUT_H = 227;	//input image height
static const int INPUT_W = 227;	//input image width
static const int OUTPUT_SIZE = 3;	//output size(3 types)

const char* INPUT_BLOB_NAME = "data";	//input name(defined in prototxt)
const char* OUTPUT_BLOB_NAME = "prob";	//output name(defined in prototxt)

// Logger for GIE info/warning/errors
class Logger : public ILogger			
{
	void log(Severity severity, const char* msg) override
	{
		// suppress info-level messages
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
} gLogger;


// JPG reader using stb_image
void readJPG(const std::string& fileName,  unsigned char*& buffer)
{
	int w,h,bpp;
	buffer = stbi_load(fileName.c_str(), &w, &h, &bpp, 3);
}

void caffeToGIEModel(const std::string& deployFile,				// name for caffe prototxt
					 const std::string& modelFile,				// name for model 
					 const std::vector<std::string>& outputs,   // network outputs
					 unsigned int maxBatchSize,					// batch size - NB must be at least as large as the batch we want to run with)
					 IHostMemory *&gieModelStream)    // output buffer for the GIE model
{
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);

	// parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();
	const IBlobNameToTensor* blobNameToTensor = parser->parse(deployFile.c_str(),
		                          			  modelFile.c_str(),
								  *network,
							  DataType::kFLOAT);

	// specify which tensors are outputs
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1 << 20);

	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	// serialize the engine, then close everything down
	gieModelStream = engine->serialize();
	engine->destroy();
	builder->destroy();
	shutdownProtobufLibrary();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
	const ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	assert(engine.getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME), 
		outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

	// create GPU buffers and a stream
	CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_D * INPUT_H * INPUT_W * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));
	cudaEvent_t start, end;	//calculate run time
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&end));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	float ms;
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_D * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	cudaEventRecord(start, stream);
	context.enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaEventRecord(end, stream);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&ms, start, end);
	cudaStreamSynchronize(stream);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	std::cout<<"execution time:"<<ms<<"ms."<<std::endl;
	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}


int main(int argc, char** argv)
{
	// create a GIE model from the caffe model and serialize it to a stream
    	IHostMemory *gieModelStream{nullptr};
	std::cout<<"Converting from caffe model..."<<std::endl;
   	caffeToGIEModel("/home/nvidia/caffemodel/deploy.prototxt", "/home/nvidia/caffemodel/mycaffenet_train_iter_450000.caffemodel", std::vector < std::string > { OUTPUT_BLOB_NAME }, 1, gieModelStream);
	std::cout<<"Convertion successful!"<<std::endl;
	unsigned char* fileData;
	std::cout<<"Reading jpg image..."<<std::endl;
	std::string fname;
	std::ifstream in("file.txt");
	in>>fname;
	readJPG(fname, fileData);
	std::cout<<"Reading successful!"<<std::endl;

	std::cout<<"Converting data..."<<std::endl;
	unsigned char* newData = (unsigned char*)malloc(INPUT_D*INPUT_H*INPUT_W*sizeof(unsigned char));
	//reformatting 
	int start = 2;
	for(int i=0;i<227*227;i++)
	{
		newData[i] = fileData[start]-97.25115522;	//magic, don't modify!(mean image)
		start += 3;
	}
	start = 1;
	for(int i=227*227;i<227*227*2;i++)
	{
		newData[i] = fileData[start]-108.84620235;	//magic, don't modify!(mean image)
		start += 3;
	}
	start = 0;
	for(int i=227*227*2;i<227*227*3;i++)
	{
		newData[i] = fileData[start]-116.72652473;	//magic, don't modify!(mean image)
		start += 3;
	}
	float data[INPUT_D*INPUT_H*INPUT_W];
	for (int i = 0; i < INPUT_D*INPUT_H*INPUT_W; i++)
		data[i] = float(newData[i]);
	std::cout<<"Converted."<<std::endl;

	std::cout<<"Deserializing the engine..."<<std::endl;
	// deserialize the engine 
	IRuntime* runtime = createInferRuntime(gLogger);
	ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), nullptr);
    if (gieModelStream) gieModelStream->destroy();

	IExecutionContext *context = engine->createExecutionContext();
	std::cout<<"Deserialized."<<std::endl;
	// run inference
	float prob[OUTPUT_SIZE];
	std::cout<<"Running inference..."<<std::endl;
	doInference(*context, data, prob, 1);
	std::cout<<"Inference successful!"<<std::endl;
	// destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();
	// print a histogram of the output distribution
	float maxP = 0.0;
	int pos = 0;
	for (unsigned int i = 0; i < 3; i++)
    	{
		std::cout<<prob[i]<<std::endl;
        	if(prob[i]>maxP)
		{
			maxP = prob[i];
			pos = i;
		}
   	}
	std::ofstream out("type.txt");
	out<<pos<<std::endl;
	return 0;
}
