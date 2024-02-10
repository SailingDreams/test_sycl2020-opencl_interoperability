#include<CL/sycl.hpp>
#include<iostream>
#include<stdlib.h>
#include <chrono>

const char* Convolution3x3Source =
R"CLC(

__kernel void convolution2D_3x3(
    __global float* restrict pOut,
    int2 outOffsetPitch,
    __global const float* restrict pIn,
    int2 inOffsetPitch,
    __constant const float* pFilter,
    int2 dim)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    const int width = dim.x;
    const int height = dim.y;

    if (x >= width || y >= height) return;

    pOut += outOffsetPitch.x;
    pIn += inOffsetPitch.x;

    float value = 0.0f;
    int cIdx = 0;

    for (int l = -1; l <= 1; l++)
    {
        const int yl = clamp(y + l, 0, height - 1);

        for (int k = -1; k <= 1; k++)
        {
            int xk = clamp(x + k, 0, width - 1);

            float v = pIn[yl * inOffsetPitch.y + xk];
            float c = pFilter[cIdx++];
            value += v * c;
        }
    }

    pOut[y * outOffsetPitch.y + x] = value;
}
    )CLC";

//using namespace std;
const char* addKernelSource =
R"CLC(
    kernel void add(global int* data) {
    int index = get_global_id(0);
    data[index] = data[index] + 1;
    }
    )CLC";

constexpr size_t gSize = 4 * 1024 * 1024;
std::array<int, gSize> gData;
std::array<int, gSize> gDataOut;

constexpr size_t calSize = 4; // calibration: make really small to measure cl queue time
std::array<int, calSize> data;
/*
* 
*/
const char* oclErrorToString(cl_int error)
{
    switch (error)
    {
    case CL_SUCCESS: return "Success";
    case CL_DEVICE_NOT_FOUND: return "Device Not Found";
    case CL_DEVICE_NOT_AVAILABLE: return "Device Not Available";
    case CL_COMPILER_NOT_AVAILABLE: return "Compiler Not Available";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "Memory Object Allocation Failure";
    case CL_OUT_OF_RESOURCES: return "Out of Resources";
    case CL_OUT_OF_HOST_MEMORY: return "Out of Host Memory";
    case CL_PROFILING_INFO_NOT_AVAILABLE: return "Profiling Information Not Available";
    case CL_MEM_COPY_OVERLAP: return "Memory Copy Overlap";
    case CL_IMAGE_FORMAT_MISMATCH: return "Image Format Mismatch";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "Image Format Not Supported";
    case CL_BUILD_PROGRAM_FAILURE: return "Build Program Failure";
    case CL_MAP_FAILURE: return "Map Failure";
    case CL_INVALID_VALUE: return "Invalid Value";
    case CL_INVALID_DEVICE_TYPE: return "Invalid Device Type";
    case CL_INVALID_PLATFORM: return "Invalid Platform";
    case CL_INVALID_DEVICE: return "Invalid Device";
    case CL_INVALID_CONTEXT: return "Invalid Context";
    case CL_INVALID_QUEUE_PROPERTIES: return "Invalid Queue Properties";
    case CL_INVALID_COMMAND_QUEUE: return "Invalid Command Queue";
    case CL_INVALID_HOST_PTR: return "Invalid Host Pointer";
    case CL_INVALID_MEM_OBJECT: return "Invalid Memory Object";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "Invalid Image Format Descriptor";
    case CL_INVALID_IMAGE_SIZE: return "Invalid Image Size";
    case CL_INVALID_SAMPLER: return "Invalid Sampler";
    case CL_INVALID_BINARY: return "Invalid Binary";
    case CL_INVALID_BUILD_OPTIONS: return "Invalid Build Options";
    case CL_INVALID_PROGRAM: return "Invalid Program";
    case CL_INVALID_PROGRAM_EXECUTABLE: return "Invalid Program Executable";
    case CL_INVALID_KERNEL_NAME: return "Invalid Kernel Name";
    case CL_INVALID_KERNEL_DEFINITION: return "Invalid Kernel Definition";
    case CL_INVALID_KERNEL: return "Invalid Kernel";
    case CL_INVALID_ARG_INDEX: return "Invalid Argument Index";
    case CL_INVALID_ARG_VALUE: return "Invalid Argument Value";
    case CL_INVALID_ARG_SIZE: return "Invalid Argument Size";
    case CL_INVALID_KERNEL_ARGS: return "Invalid Kernel Arguments";
    case CL_INVALID_WORK_DIMENSION: return "Invalid Work Dimension";
    case CL_INVALID_WORK_GROUP_SIZE: return "Invalid Work Group Size";
    case CL_INVALID_WORK_ITEM_SIZE: return "Invalid Work Item Size";
    case CL_INVALID_GLOBAL_OFFSET: return "Invalid Global Offset";
    case CL_INVALID_EVENT_WAIT_LIST: return "Invalid Event Wait List";
    case CL_INVALID_EVENT: return "Invalid Event";
    case CL_INVALID_OPERATION: return "Invalid Operation";
    case CL_INVALID_GL_OBJECT: return "Invalid GL Object";
    case CL_INVALID_BUFFER_SIZE: return "Invalid Buffer Size";
    case CL_INVALID_MIP_LEVEL: return "Invalid MIP Level";
    case CL_INVALID_GLOBAL_WORK_SIZE: return "Invalid Global Work Size";
#ifdef CL_VERSION_1_2
    case CL_COMPILE_PROGRAM_FAILURE: return "Compile Program Failure";
    case CL_LINKER_NOT_AVAILABLE: return "Linker Not Available";
    case CL_LINK_PROGRAM_FAILURE: return "Link Program Failure";
    case CL_DEVICE_PARTITION_FAILED: return "Device Partition Failed";
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE: return "Kernel Argument Info Not Available";
    case CL_INVALID_PROPERTY: return "Invalid Property";
    case CL_INVALID_IMAGE_DESCRIPTOR: return "Invalid Image Descriptor";
    case CL_INVALID_COMPILER_OPTIONS: return "Invalid Compiler Options";
    case CL_INVALID_LINKER_OPTIONS: return "Invalid Linker Options";
    case CL_INVALID_DEVICE_PARTITION_COUNT: return "Invalid Device Partition Count";
#endif // CL_VERSION_1_2
#ifdef CL_VERSION_2_0
    case CL_INVALID_PIPE_SIZE: return "Invalid Pipe Size";
    case CL_INVALID_DEVICE_QUEUE: return "Invalid Device Queue";
#endif
    default:
        return "Unknown";
    }
}

/**
* 
*/
#if 1
void setEnvVar(bool gpuSelected) {
    if (gpuSelected) {
        _putenv_s("SYCL_DEVICE_FILTER", "OPENCL:GPU");
        //std::setenv("SYCL_DEVICE_FILTER", "", true);
    }
    else {
        _putenv_s("SYCL_DEVICE_FILTER", "");
        //setenv("SYCL_DEVICE_FILTER", "OPENCL:GPU", true);
    }

    char* a;
    a = getenv("SYCL_DEVICE_FILTER");
    if (a == NULL) {
        std::cout << "SYCL_DEVICE_FILTER is not an env var" << std::endl;
    }
    else {
        std::cout << "SYCL_DEVICE_FILTER = " << a << std::endl;
    }//std::system("echo %SYCL_DEVICE_FILTER%"); // doesn't work
}

/*
* 
*/
void getKernel(cl_context ocl_ctx, cl_device_id ocl_dev,  
                  const char* kernelSource, cl_kernel &ocl_kernel,
                  const char * kernelName)
{
    cl_int err = CL_SUCCESS;

    cl_program ocl_program = clCreateProgramWithSource(ocl_ctx, 1, &kernelSource, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cout << "ERROR on clCreateProgramWithSource = " << oclErrorToString(err) << std::endl;
    }
    err = CL_SUCCESS;
    err = clBuildProgram(ocl_program, 1, &ocl_dev, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "ERROR on clBuildProgram = " << oclErrorToString(err) << std::endl;
    }
    err = CL_SUCCESS;
    ocl_kernel = clCreateKernel(ocl_program, kernelName, &err);
    if (err != CL_SUCCESS) {
        std::cout << "ERROR on clCreateKernel = " << oclErrorToString(err) << std::endl;
    }    
}
#endif
/*
* 
*/
bool checkArgs(int argc, char** argv, bool &gpuSelected, int &iterations, 
               bool &verbose, bool &enableTrace) {

    if (argc != 5) {
        std::cout << "Not enough arguments." << std::endl;
        std::cout << "syntax: syclOpenclBenchmark.exe device[cpu|gpu] iterations[> 100] verbose[true,false] enableTrace[true|false]" << std::endl;
        return false;
    }
    std::string cpuOrGpu(argv[1]);
    iterations = atoi(argv[2]); // The larger the number, the better the processing time estimate
    std::string verboseString(argv[3]);
    std::string enableTraceString(argv[4]);

    if (cpuOrGpu == "cpu") {
        gpuSelected = false;
    }
    else if (cpuOrGpu == "gpu") {
        gpuSelected = true;
    }
    else {
        std::cout << "ERROR:  argument incorrect. Either true or false" << std::endl;
        return false;
    }

    verbose = false;
    if (verboseString == "true") {
        verbose = true;
    }
    else if (verboseString == "false") {
        verbose = false;
    }
    else {
        std::cout << "ERROR: verbose argument incorrect. Either true or false" << std::endl;
        return false;
    }

    enableTrace = false;
    if (enableTraceString == "true") {
        enableTrace = true;
    }
    else if (enableTraceString == "false") {
        enableTrace = false;
    }
    else {
        std::cout << "ERROR: enableTrace argument incorrect. Either true or false" << std::endl;
        return false;
    }

    return true;
}
/**
* 
*/
int main(int argc, char** argv)
{
    bool gpuSelected = false;
    int iterations = atoi(argv[2]); // The larger the number, the better the processing time estimate
    bool verbose = false;
    bool enableTrace = true;

    bool pass = checkArgs(argc, argv, gpuSelected, iterations, verbose, enableTrace);
    if (!pass) { return -1; }
    
    //int iterations = 100; 

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    cl_int err = CL_SUCCESS;

    for (int i = 0; i < gSize; i++) {
      gData[i] = i;
    }

    // Set trace to see what opencl device is used SYCL_PI_TRACE = 1
    if (enableTrace) {
        _putenv_s("SYCL_PI_TRACE", "1");
    }
    else {
        _putenv_s("SYCL_PI_TRACE", "");
    }

    int (*selector)(const sycl::_V1::device&);

    if (gpuSelected) {
        selector = sycl::gpu_selector_v;
    }
    else {
        selector = sycl::cpu_selector_v;
    }
    
    setEnvVar(gpuSelected);

    sycl::device dev(selector); // eddie added
    sycl::context ctx=sycl::context(dev);

    sycl::backend myBackend = dev.get_backend();
    // This will output "opencl" irregarless of cpu_selector_v or gpu_selector_v. Why?
    // std::cout << "The selected backend is " << myBackend << std::endl;

    cl_device_id ocl_dev;
    try {
        ocl_dev = sycl::get_native<cl::sycl::backend::opencl, sycl::device>(dev);
    }
    catch (const char e) {
        std::cout << "An exception is caught while getting ocl device.\n";
        std::cout << "The exception output = " << e << std::endl;
        std::terminate();
    }

    auto ocl_ctx=sycl::get_native<cl::sycl::backend::opencl,sycl::context>(ctx);  // ejp todo Write try catch for this 

    cl_command_queue ocl_queue = clCreateCommandQueueWithProperties(ocl_ctx, ocl_dev,0,&err);
    sycl::queue q=sycl::make_queue<sycl::backend::opencl>(ocl_queue,ctx); 

    std::cout << "Running on device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "sizeof(cl_mem) = " << sizeof(cl_mem) << std::endl;

    // Create to measure cl queue time.
    cl_mem ocl_buf_cal = clCreateBuffer(ocl_ctx,CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, calSize * sizeof(int), &data[0],&err);
    // create buffer for gpu speed test
    cl_mem ocl_buf = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, gSize * sizeof(int), &gData[0], &err);
    if (err != CL_SUCCESS || ocl_buf == nullptr)
    {
        std::cout << "Failed to allocate device buffer of size " << gSize << ". Error: " << oclErrorToString(err) << std::endl;
        return -1;
    }


    sycl::buffer<int, 1> bufferCal = sycl::make_buffer<sycl::backend::opencl, int>(ocl_buf_cal, ctx);
    sycl::buffer<int, 1> buffer =sycl::make_buffer<sycl::backend::opencl, int>(ocl_buf, ctx);


    cl_kernel ocl_kernel_conv3x3;
    getKernel(ocl_ctx, ocl_dev, Convolution3x3Source, ocl_kernel_conv3x3, "convolution2D_3x3");
    sycl::kernel concolution3x3_kernel = sycl::make_kernel<sycl::backend::opencl>(ocl_kernel_conv3x3, ctx);

    // ejp todo: put make_kernel in getKernel function
    cl_kernel ocl_kernel;
    getKernel(ocl_ctx, ocl_dev, addKernelSource, ocl_kernel,"add");
    sycl::kernel add_kernel = sycl::make_kernel<sycl::backend::opencl>(ocl_kernel, ctx);


    // Measure setup and tear down size   
    size_t CumulativeTime = 0;

    for (int i = 0; i < iterations; i++) {
        begin = std::chrono::steady_clock::now();
        q.submit([&](sycl::handler& h) {
            auto data_acc = bufferCal.get_access<sycl::access_mode::read_write,
            sycl::target::device>(h);
            h.set_args(data_acc);
            h.parallel_for(calSize, add_kernel);
            }).wait();
        end = std::chrono::steady_clock::now();
        auto queuingTimeNs = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        if (verbose) {
            std::cout << "Queuing Time " << queuingTimeNs.count() << "[ns]" << std::endl;
        }
        CumulativeTime += queuingTimeNs.count();
    }
    auto avgQueuingTime = CumulativeTime / iterations;
    std::cout << "Avg Queuing Time " << avgQueuingTime << "[ns]" << std::endl;

    // Measuring processing time using the big array. Reset values used for function call timing
    std::cout << "Precessing array size = " << (double)gSize / (1024.0 * 1024.0) << " M ints ";
    std::cout << " for " << iterations << " iterations." << std::endl;

    CumulativeTime = 0;
    for (int i = 0; i < iterations; i++) {
        begin = std::chrono::steady_clock::now();
        q.submit([&](sycl::handler& h) {
            auto data_acc = buffer.get_access<sycl::access_mode::read_write,
            sycl::target::device>(h);
            h.set_args(data_acc);
            h.parallel_for(gSize, add_kernel);
            }).wait();

        end = std::chrono::steady_clock::now();

        auto processingPlusQueuingTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        if (verbose) {
            std::cout << "Processing Plus Queuing Time = " << processingPlusQueuingTime.count() << "[ns]" << std::endl;
        }
        CumulativeTime += processingPlusQueuingTime.count();

        // check results
        auto isBlocking = CL_TRUE; // use blocking reads/writes
        clEnqueueReadBuffer(ocl_queue, ocl_buf, CL_TRUE, 0, gSize * sizeof(int), &gDataOut[0], 0, NULL, NULL);

        bool resultsOk = true;
        for (int i = 0; i < gSize; i++) {
            //std::cout<<data[i]<<std::endl;
            if (gDataOut[i] != i + 1) {
                std::cout << "Results did not validate at index " << i << "!\n";
                resultsOk = false;
                return -1;
            }
        }
        if (resultsOk && verbose) {
            std::cout << "Success!\n";
        }

        // Reset buffer for next iteration
        //q.memcpy(buffer, &gData[0], gSize * sizeof(int)).wait();
        clEnqueueWriteBuffer(ocl_queue, ocl_buf, isBlocking, 0, gSize * sizeof(int), &gData[0], 0, NULL, NULL);

        if(i % 10 == 0) {
            std::cout << ".";
        }
    }
    std::cout << std::endl;

    auto avgProcessingPlusQueuingTime = CumulativeTime / iterations;
    std::cout << "Avg Processing Plus Queuing Time = " << avgProcessingPlusQueuingTime << "[ns]" << std::endl;
    auto avgProcessingTime = avgProcessingPlusQueuingTime - avgQueuingTime;
    if (avgProcessingPlusQueuingTime < avgQueuingTime) {
        std::cout << "Bad results. avgProcessingPlusQueuingTime < avgQueuingTime. Use more iterations" << std::endl;
    }
    else {
        std::cout << "Avg Processing Time = " << avgProcessingTime << "[ns]" << std::endl;
    }
    


    return 0;
}

