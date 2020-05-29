//  Author: Arjun Ramaswami

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CL_VERSION_2_0
#include <CL/cl_ext_intelfpga.h> // to disable interleaving & transfer data to specific banks - CL_CHANNEL_1_INTELFPGA
#include "CL/opencl.h"

#include "transpose_fpga.h"
#include "opencl_utils.h"
#include "helper.h"

#ifndef KERNEL_VARS
#define KERNEL_VARS
static cl_platform_id platform = NULL;
static cl_device_id *devices;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_program program = NULL;
static cl_command_queue queue1 = NULL, queue2 = NULL, queue3 = NULL;
#endif

static void queue_setup();
void queue_cleanup();

/** 
 * @brief Allocate memory of single precision complex floating points
 * @param sz  : size_t : size to allocate
 * @param svm : 1 if svm
 * @return void ptr or NULL
 */
void* fpgaf_complex_malloc(size_t sz, int svm){
  if(svm == 1){
    fprintf(stderr, "Working in progress\n");
    return NULL;
    // return aocl_mmd_shared_mem_alloc(svm_handle, sizeof(double2) * sz, inData, device_ptr);
  }
  else if(sz == 0){
    return NULL;
  }
  else{
    return ((float2 *)alignedMalloc(sz));
  }
}

/** 
 * @brief Initialize FPGA
 * @param platform name: string - name of the OpenCL platform
 * @param path         : string - path to binary
 * @param use_svm      : 1 if true 0 otherwise
 * @param use_emulator : 1 if true 0 otherwise
 * @return 0 if successful 
 */
int fpga_initialize(const char *platform_name, const char *path, int use_svm, int use_emulator){
  cl_int status = 0;

#ifdef VERBOSE
  printf("\tInitializing FPGA ...\n");
#endif

  if(path == NULL || strlen(path) == 0){
    fprintf(stderr, "Path to binary missing\n");
    return 1;
  }

  // Check if this has to be sent as a pointer or value
  // Get the OpenCL platform.
  platform = findPlatform(platform_name);
  if(platform == NULL){
    fprintf(stderr,"ERROR: Unable to find %s OpenCL platform\n", platform_name);
    return 1;
  }
  // Query the available OpenCL devices.
  cl_uint num_devices;
  devices = getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices);
  if(devices == NULL){
    fprintf(stderr, "ERROR: Unable to find devices for %s OpenCL platform\n", platform_name);
    return 1;
  }

  // use the first device.
  device = devices[0];

  // Create the context.
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
  checkError(status, "Failed to create context");

#ifdef VERBOSE
  printf("\tGetting program binary from path %s ...\n", path);
#endif
  // Create the program.
  program = getProgramWithBinary(context, &device, 1, path);
  if(program == NULL) {
    fprintf(stderr, "Failed to create program\n");
    fpga_final();
    return 1;
  }

#ifdef VERBOSE
  printf("\tBuilding program ...\n");
#endif
  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  return 0;
}

/** 
 * @brief Release FPGA Resources
 */
void fpga_final(){

#ifdef VERBOSE
  printf("\tCleaning up FPGA resources ...\n");
#endif
  if(program) 
    clReleaseProgram(program);
  if(context)
    clReleaseContext(context);
  free(devices);
}

/**
 * \brief  compute an complex single precision matrix transposition on the FPGA
 * \param  N   : length of the matrix
 * \param  inp : pointer to input matrix
 * \param  out : pointer to output matrix
 * \param  batch : number of transposes to perform in a batched mode
 * \param  use_svm : 1 if pcie transfers are SVM based
 * \param  isND : 1 if kernel is ND Range
 * \retval fpga_t : time taken in milliseconds for data transfers and execution
 */
fpga_t mTranspose(int N, float2 *inp, float2 *out, int batch, int use_svm, int isND){
  fpga_t mTranspose_time = {0.0, 0.0, 0.0, 0};
  cl_kernel fetch_kernel = NULL, transpose_kernel = NULL, store_kernel = NULL;
  cl_int status = 0;

  // if N is not a power of 2
  if(inp == NULL || out == NULL || ((N & (N-1)) !=0)){
    return mTranspose_time;
  }

  queue_setup();

  //unsigned num_pts = batch * N * N;
  size_t buf_sz = sizeof(float2) * batch * N * N;

  // Create device buffers 
  cl_mem d_inData, d_outData;
  d_inData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_1_INTELFPGA, buf_sz, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");

  d_outData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, buf_sz, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

 // Copy data from host to device
  mTranspose_time.pcie_write_t = getTimeinMilliSec();

  status = clEnqueueWriteBuffer(queue1, d_inData, CL_TRUE, 0, buf_sz, inp, 0, NULL, NULL);

  mTranspose_time.pcie_write_t = getTimeinMilliSec() - mTranspose_time.pcie_write_t;
  checkError(status, "Failed to copy data to device");

  // create kernel
  fetch_kernel = clCreateKernel(program, "fetch", &status);
  checkError(status, "Failed to create fetch kernel");
  transpose_kernel = clCreateKernel(program, "transpose", &status);
  checkError(status, "Failed to create transpose kernel");
  store_kernel = clCreateKernel(program, "store", &status);
  checkError(status, "Failed to create store kernel");

  // kernel args
  status = clSetKernelArg(fetch_kernel, 0, sizeof(cl_mem), (void *)&d_inData);
  checkError(status, "Failed to set fetch kernel arg 0");
  status = clSetKernelArg(fetch_kernel, 1, sizeof(cl_int), (void*)&batch);
  checkError(status, "Failed to set fetch kernel arg 1");

  status = clSetKernelArg(transpose_kernel, 0, sizeof(cl_int), (void*)&batch);
  checkError(status, "Failed to set fetch kernel arg");

  status = clSetKernelArg(store_kernel, 0, sizeof(cl_mem), (void *)&d_outData);
  checkError(status, "Failed to set transpose kernel arg 0");
  status = clSetKernelArg(store_kernel, 1, sizeof(cl_int), (void*)&batch);
  checkError(status, "Failed to set fetch kernel arg 1");

  double start = getTimeinMilliSec();
  if(isND){
    size_t lws_transfer[] = {N};
    size_t gws_transfer[] = {batch * N * N / 8}; 

    status = clEnqueueNDRangeKernel(queue1, fetch_kernel, 1, 0, gws_transfer, lws_transfer, 0, NULL, NULL);
    checkError(status, "Failed to launch kernel");

    size_t lws_transpose_kernel[] = {N * N / 8};
    size_t gws_transpose_kernel[] = {batch * N * N / 8}; 
    status = clEnqueueNDRangeKernel(queue2, transpose_kernel, 1, 0, gws_transpose_kernel, lws_transpose_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch kernel");

    status = clEnqueueNDRangeKernel(queue3, store_kernel, 1, 0, gws_transfer, lws_transfer, 0, NULL, NULL);
    checkError(status, "Failed to launch kernel");
  }
  else{
    printf("fetch kernel \n");
    status = clEnqueueTask(queue1, fetch_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch fetch kernel");

    printf("transpose kernel \n");
    status = clEnqueueTask(queue2, transpose_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch transpose kernel");

    printf("store kernel \n");
    status = clEnqueueTask(queue3, store_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch store kernel");
    printf("finished store kernel \n");
  }

  // Wait for all command queues to complete pending events
  printf("Finishing Queue1\n");
  status = clFinish(queue1);
  checkError(status, "failed to finish");
  printf("Finishing Queue2\n");
  status = clFinish(queue2);
  checkError(status, "failed to finish");
  printf("Finishing Queue3\n");
  status = clFinish(queue3);
  checkError(status, "failed to finish");

  double stop = getTimeinMilliSec();
  mTranspose_time.exec_t = stop - start;

  mTranspose_time.pcie_read_t = getTimeinMilliSec();

  printf("Reading from buffer %d %d\n", N, batch);
  status = clEnqueueReadBuffer(queue1, d_outData, CL_TRUE, 0, buf_sz, out, 0, NULL, NULL);

  printf("finished Reading from buffer \n");
  mTranspose_time.pcie_read_t = getTimeinMilliSec() - mTranspose_time.pcie_read_t;
  checkError(status, "Failed to read data from device");

  queue_cleanup();

  printf("Queue Cleanup \n");
  if (d_inData)
  	clReleaseMemObject(d_inData);
  if (d_outData)
  	clReleaseMemObject(d_outData);

  printf("Kernel Cleanup \n");
  if(fetch_kernel) 
    clReleaseKernel(fetch_kernel);  
  if(transpose_kernel) 
    clReleaseKernel(transpose_kernel);  
  if(store_kernel) 
    clReleaseKernel(store_kernel);  

  mTranspose_time.valid = 1;
  return mTranspose_time;
}

/**
 * \brief Create a command queue for each kernel
 */
void queue_setup(){
  cl_int status = 0;
  // Create one command queue for each kernel.
  queue1 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue1");
  queue2 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue2");
  queue3 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue3");
}

/**
 * \brief Release all command queues
 */
void queue_cleanup() {
  if(queue1) 
    clReleaseCommandQueue(queue1);
  if(queue2) 
    clReleaseCommandQueue(queue2);
  if(queue3) 
    clReleaseCommandQueue(queue3);
}