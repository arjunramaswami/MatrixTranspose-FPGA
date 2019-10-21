/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

// global dependencies
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// common dependencies
#include "CL/opencl.h"
#include "../common/opencl_utils.h"
#include "transpose_api.h"
#include "helper.h"
#include <CL/cl_ext_intelfpga.h> // to disable interleaving & transfer data to specific banks - CL_CHANNEL_1_INTELFPGA

// host variables
#ifndef KERNEL_VARS
#define KERNEL_VARS
static cl_platform_id platform = NULL;
static cl_device_id *devices;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_program program = NULL;
static cl_command_queue queue1 = NULL, queue2 = NULL, queue3 = NULL;
#endif

// Function prototypes
int init();
void cleanup();
static void cleanup_program();
static void init_program(int N[2], char *data_path);
static void queue_setup();
void queue_cleanup();
static double fpga_run(int N[2], float2 *c_in, unsigned iter);

// --- CODE -------------------------------------------------------------------

int fpga_initialize_(){
   return init();
}

void fpga_final_(){
   cleanup();
}

/******************************************************************************
 * \brief  check whether matrix transposition can be computed on the FPGA or not. This depends 
 *         on the availability of bitstreams whose sizes are for now listed here 
 *         If the matrix sizes are found and the FPGA is not setup before, it is done
 * \param  data_path - path to the data directory
 * \param  N - integer pointer to the size of the matrix
 * \retval true if matrix size supported
 *****************************************************************************/
int fpga_check_bitstream_(char *data_path, int N[2]){
    static int transpose_size[2] = {0, 0};

    // check the supported sizes
    if( (N[0] == 8 && N[1] == 8) ||
        (N[0] == 16 && N[1] == 16) ||
        (N[0] == 32 && N[1] == 32) ||
        (N[0] == 64 && N[1] == 64) ||
        (N[0] == 128 && N[1] == 128)  ){

        // if first time
        if( transpose_size[0] == 0 && transpose_size[1] == 0 ){
          transpose_size[0] = N[0];
          transpose_size[1] = N[1];

          init_program(transpose_size, data_path);
        }
        else if( transpose_size[0] == N[0] && transpose_size[1] == N[1] ){
          // if same matrix size as previous
          // dont do anything
        }
        else{
            // else if different matrix size as previous
            // cleanup and initialize
          transpose_size[0] = N[0];
          transpose_size[1] = N[1];

          cleanup_program();
          init_program(transpose_size, data_path);
        }

        return 1;
    }
    else{
        return 0;
    } 
}

/******************************************************************************
 * \brief   compute an in-place single precision matrix transposition on the FPGA
 * \param   N   : integer pointer to size of matrix
 * \param   din : complex input/output single precision data pointer 
 * \retval double : time taken for transposition
 *****************************************************************************/
double fpga_matrix_transpose_(int N[2], float2 *din, unsigned iter) {
  return fpga_run(N, din, iter);
}

/******************************************************************************
 * \brief   Transpose a 2d matrix
 * \param   N       : integer pointer to size of matrix
 * \param   din     : complex input/output single precision data pointer 
 * \retval double : time taken for matrix transposition
 *****************************************************************************/
static double fpga_run(int N[2], float2 *c_in, unsigned iter) {
  cl_int status = 0;
  cl_kernel fetch_kernel = NULL, transpose_kernel = NULL, store_kernel = NULL;
  unsigned index = 0;
  const unsigned buf_size = N[0] * N[1] * iter;
  int iterations = iter;
  float2 *c_in_buf = (float2 *)alignedMalloc(sizeof(float2) * buf_size);
  if (c_in_buf == NULL){
    printf("Unable to allocate host memory to local buffers\n");
    exit(1);
  }
  float2 *c_out_buf = (float2 *)alignedMalloc(sizeof(float2) * buf_size);
  if (c_out_buf == NULL){
    printf("Unable to allocate host memory to local buffers\n");
    exit(1);
  }

  // Host Buffers : Distribute Data 
  memcpy(&c_in_buf[index], &c_in[index], sizeof(float2) * buf_size);

  // Create the kernel - name passed in here must match kernel name in the
  // original CL file, that was compiled into an AOCX file using the AOC tool
  fetch_kernel = clCreateKernel(program, "fetch", &status);
  checkError(status, "Failed to create fetch kernel");
  transpose_kernel = clCreateKernel(program, "transpose", &status);
  checkError(status, "Failed to create transpose kernel");
  store_kernel = clCreateKernel(program, "store", &status);
  checkError(status, "Failed to create store kernel");

  // Device memory buffers
  cl_mem d_inData, d_outData;
  d_inData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_1_INTELFPGA, sizeof(float2) * buf_size, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");
  d_outData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * buf_size, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  queue_setup();

  // Copy data from host to device
  status = clEnqueueWriteBuffer(queue1, d_inData, CL_TRUE, 0, sizeof(float2) * buf_size, c_in_buf, 0, NULL, NULL);
  checkError(status, "Failed to copy data to device");
  status = clFinish(queue1);
  checkError(status, "failed to finish copying to buffer 1");
  
  status = clSetKernelArg(fetch_kernel, 0, sizeof(cl_mem), (void *)&d_inData);
  checkError(status, "Failed to set fetch kernel arg");
  status = clSetKernelArg(transpose_kernel, 0, sizeof(cl_int), (void*)&iterations);
  checkError(status, "Failed to set fetch kernel arg");
  status = clSetKernelArg(store_kernel, 0, sizeof(cl_mem), (void *)&d_outData);
  checkError(status, "Failed to set transpose kernel arg");

  double start = getTimeinMilliSec();
  /*
  status = clEnqueueTask(queue1, fetch_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fetch kernel");
  */
  size_t lws_transpose[] = {N[0]};
  size_t gws_transpose[] = {iter * N[0] * N[1] / 8}; 

  status = clEnqueueNDRangeKernel(queue1, fetch_kernel, 1, 0, gws_transpose, lws_transpose, 0, NULL, NULL);
  checkError(status, "Failed to launch kernel");

  size_t lws_transpose_kernel[] = {N[0] * N[1] / 8};
  size_t gws_transpose_kernel[] = {iter * N[0] * N[1] / 8}; 
  status = clEnqueueNDRangeKernel(queue2, transpose_kernel, 1, 0, gws_transpose_kernel, lws_transpose_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch kernel");

  /*
  status = clEnqueueTask(queue2, transpose_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch transpose kernel");
  status = clEnqueueTask(queue3, store_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch store kernel");
  */
  status = clEnqueueNDRangeKernel(queue3, store_kernel, 1, 0, gws_transpose, lws_transpose, 0, NULL, NULL);
  checkError(status, "Failed to launch kernel");

  // Wait for all command queues to complete pending events
  status = clFinish(queue1);
  checkError(status, "failed to finish");
  status = clFinish(queue2);
  checkError(status, "failed to finish");
  status = clFinish(queue3);
  checkError(status, "failed to finish");

  double stop = getTimeinMilliSec();
  double fpga_runtime = stop - start;
   
  // Copy results from device to host
  status = clEnqueueReadBuffer(queue1, d_outData, CL_TRUE, 0, sizeof(float2) * buf_size, c_in_buf, 0, NULL, NULL);
  checkError(status, "Failed to read data from device");

  memcpy(&c_in[index], &c_in_buf[index], sizeof(float2) * buf_size);

  queue_cleanup();

  if (c_in_buf)
	  free(c_in_buf);

  if (d_inData)
  	clReleaseMemObject(d_inData);
  if (d_outData)
  	clReleaseMemObject(d_outData);

  if(fetch_kernel) 
    clReleaseKernel(fetch_kernel);  
  if(transpose_kernel) 
    clReleaseKernel(transpose_kernel);  
  if(store_kernel) 
    clReleaseKernel(store_kernel);  

  return fpga_runtime;
}


/******************************************************************************
 * \brief   Initialize the program - select device, create context and program
 *****************************************************************************/
void init_program(int N[2], char *data_path){
  cl_int status = 0;

  // use the first device.
  device = devices[0];

  // Create the context.
  context = clCreateContext(NULL, 1, &device, &openCLContextCallBackFxn, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the program.
  program = getProgramWithBinary(context, &device, 1, N, data_path);
  if(program == NULL) {
    printf("Failed to create program");
    exit(1);
  }
  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

}

/******************************************************************************
 * \brief   Free resources allocated during program initialization
 *****************************************************************************/
void cleanup_program(){
  if(program) 
    clReleaseProgram(program);
  if(context)
    clReleaseContext(context);
}

/******************************************************************************
 * \brief   Initialize the OpenCL FPGA environment - platform and devices
 * \retval  true if error in initialization
 *****************************************************************************/
int init() {
  cl_int status = 0;

  // Get the OpenCL platform.
  platform = findPlatform("Intel(R) FPGA");
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform\n");
    return 1;
  }
  // Query the available OpenCL devices.
  cl_uint num_devices;
  devices = getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices);

  return 0;
}

/******************************************************************************
 * \brief   Free resources allocated during initialization - devices
 *****************************************************************************/
void cleanup(){
  cleanup_program();
  free(devices);
}

/******************************************************************************
 * \brief   Create a command queue for each kernel
 *****************************************************************************/
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

/******************************************************************************
 * \brief   Release all command queues
 *****************************************************************************/
void queue_cleanup() {
  if(queue1) 
    clReleaseCommandQueue(queue1);
  if(queue2) 
    clReleaseCommandQueue(queue2);
  if(queue3) 
    clReleaseCommandQueue(queue3);
}