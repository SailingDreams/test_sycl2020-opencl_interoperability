# 3x3 opencl convolution using syscl interoperability

## Tools
Visual Studio 2022
2020 oneAPI Base Toolkit

## Usage
Powershell syntax
cd x64\Debug
 .\syclOpenClBenchmark.exe gpu 1000 false false
.\syclOpenClBenchmark.exe cpu 1000 false false

  Help
  syclOpenclBenchmark.exe device[cpu|gpu] iterations[> 100] verbose[true,false] enableTrace[true|false]

## Results
Running with the gcc compiler is much slower than Visual C++ Compiler.  
CPU - 480 usec
GPU - 400 usec
The example doesn't reflect the fact that if you had a long processing chain, you wouldn't use the wait() at each stage. Furthermore, the CPU can be used for other things while the GPU processing queue finishes. Typically, a lot of img processing algorithms would be queued before getting the final img.

### Output
3x3 Conv Kernel on 0.4 MFloat img
**********CPU ************************************************************
SYCL_PI_TRACE[all]:   platform: Intel(R) OpenCL
SYCL_PI_TRACE[all]:   device: Intel(R) Core(TM) i5-7200U CPU @ 2.50GHz
Running on device: Intel(R) Core(TM) i5-7200U CPU @ 2.50GHz
***********************************************************
Avg Queuing Time 192399[ns]
***********************************************************
Processing img 512 x 1014 (0.495117 M floats ) for 20000 iterations.
....................................................................................................
Img: Avg Processing Plus Queuing Time = 673263[ns]
Img: Avg Processing Time = 480864[ns]

**********GPU ************************************************************
SYCL_DEVICE_FILTER = OPENCL:GPU
SYCL_PI_TRACE[all]: Selected device: -> final score = 1000
SYCL_PI_TRACE[all]:   platform: Intel(R) OpenCL HD Graphics
SYCL_PI_TRACE[all]:   device: Intel(R) HD Graphics 620
Running on device: Intel(R) HD Graphics 620
***********************************************************
Avg Queuing Time 188638[ns]
***********************************************************
Precessing img 512 x 1014 (0.495117 M floats ) for 20000 iterations.
....................................................................................................
Img: Avg Processing Plus Queuing Time = 593083[ns]
Img: Avg Processing Time = 404445[ns]

## Credits
Thanks to C at Intel for help debugging. 

## Reference
Data Parallel C++
Mastering DPC++ for Programming of Heterogeneous Systems using C++ and SYCL
https://link.springer.com/book/10.1007/978-1-4842-5574-2
