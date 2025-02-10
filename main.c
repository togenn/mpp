#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

#include <CL/cl.h>

#define MAX_PLATFORMS 10
#define MAX_INFO_SIZE 1024
#define SIZE 100

void displayPlatformInfo(cl_platform_id platform, cl_platform_info param_name, const char* param_str) {
    char info[MAX_INFO_SIZE];
    size_t info_size;

    clGetPlatformInfo(platform, param_name, sizeof(info), info, &info_size);
    info[info_size] = '\0'; // Ensure null termination
    printf("%s: %s\n", param_str, info);
}

float* allocate_matrix() {
    return (float*)malloc(SIZE * SIZE * sizeof(float));
}

void initialize_matrix(float* matrix) {
    for (int i = 0; i < SIZE * SIZE; i++) {
        matrix[i] = (float)(i % 10);
    }
}

void add_Matrix(float* matrix_1, float* matrix_2, float* result) {
    for (int i = 0; i < SIZE * SIZE; i++) {
        result[i] = matrix_1[i] + matrix_2[i];
    }
}

void print_matrix(float* matrix) {
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            printf("%.2f ", matrix[i * SIZE + j]);
        }
        printf("\n");
    }
}

#ifndef _WIN32
double get_time() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec / 1e6;
}
#else
// Timing function for Windows
double get_time() {
    LARGE_INTEGER frequency, start;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    return (double)start.QuadPart / frequency.QuadPart;
}
#endif

void check_error(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        printf("Error during operation '%s' (Error Code: %d)\n", operation, err);
        exit(1);
    }
}

int matrix_addtion_test() {

    cl_platform_id platforms[MAX_PLATFORMS];
    cl_uint num_platforms;

    // Get available platforms
    cl_int err = clGetPlatformIDs(MAX_PLATFORMS, platforms, &num_platforms);
    if (err != CL_SUCCESS) {
        printf("Failed to get OpenCL platforms.\n");
        return 1;
    }

    printf("Found %u OpenCL platform(s):\n", num_platforms);

    // Display platform information
    for (cl_uint i = 0; i < num_platforms; i++) {
        printf("\nPlatform %u:\n", i + 1);
        displayPlatformInfo(platforms[i], CL_PLATFORM_NAME, "Platform Name");
        displayPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, "Vendor");
        displayPlatformInfo(platforms[i], CL_PLATFORM_VERSION, "Version");
        displayPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, "Profile");
        displayPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, "Extensions");
    }


    // Allocate memory for matrices
    float* matrix_1 = allocate_matrix();
    float* matrix_2 = allocate_matrix();
    float* result = allocate_matrix();

    if (!matrix_1 || !matrix_2 || !result) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    // Initialize matrices
    initialize_matrix(matrix_1);
    initialize_matrix(matrix_2);

    // Start timing
    double start_time = get_time();

    // Perform element-wise addition
    add_Matrix(matrix_1, matrix_2, result);

    // Stop timing
    double end_time = get_time();

    // Print the resulting matrix
    printf("Resulting Matrix:\n");
    print_matrix(result);

    // Print execution time
    printf("Host execution Time: %f milliseconds\n", (end_time - start_time) * 1000);


    // Get OpenCL platform
    cl_platform_id platform;
    check_error(clGetPlatformIDs(1, &platform, &num_platforms), "Getting platform");

    // Get OpenCL device
    cl_device_id device;
    cl_uint num_devices;
    check_error(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices), "Getting device");

    // Create OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    // Create command queue with profiling enabled
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, (cl_queue_properties[]) { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 }, NULL);

    // Read kernel file
    FILE* file = fopen("add_matrix.cl", "r");
    if (!file) {
        printf("Failed to open kernel file.\n");
        exit(1);
    }
    char* kernel_source = (char*)malloc(8192);
    size_t kernel_size = fread(kernel_source, 1, 8192, file);
    fclose(file);

    // Create program and build
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_size, NULL);
    check_error(clBuildProgram(program, 1, &device, NULL, NULL, NULL), "Building program");

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "add_matrix", NULL);

    // Allocate device memory
    cl_mem d_matrix_1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        SIZE * SIZE * sizeof(float), matrix_1, NULL);
    cl_mem d_matrix_2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        SIZE * SIZE * sizeof(float), matrix_2, NULL);
    cl_mem d_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SIZE * SIZE * sizeof(float), NULL, NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_matrix_1);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_matrix_2);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_result);

    // Define global work size
    size_t global_size[2] = { SIZE, SIZE };

    // Create event for profiling
    cl_event event;

    // Execute kernel with profiling
    check_error(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, NULL, 0, NULL, &event), "Running kernel");

    // Wait for completion
    clWaitForEvents(1, &event);

    // Get profiling information
    cl_ulong start_time_cl, end_time_cl;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time_cl, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time_cl, NULL);

    // Compute execution time in milliseconds
    double execution_time_ms = (end_time_cl - start_time_cl) / 1e6;

    // Copy result back to host
    check_error(clEnqueueReadBuffer(queue, d_result, CL_TRUE, 0, SIZE * SIZE * sizeof(float), result, 0, NULL, NULL), "Reading result");

    // Print a portion of the result
    printf("Result Matrix (First 10x10 elements):\n");
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            printf("%.2f ", result[i * SIZE + j]);
        }
        printf("\n");
    }

    // Print execution time
    printf("Device Execution Time: %.6f ms\n", execution_time_ms);

    // Cleanup
    clReleaseEvent(event);
    clReleaseMemObject(d_matrix_1);
    clReleaseMemObject(d_matrix_2);
    clReleaseMemObject(d_result);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(matrix_1);
    free(matrix_2);
    free(result);
    free(kernel_source);

    return 0;
}

int main() {
    return matrix_addtion_test();
}