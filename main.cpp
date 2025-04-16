#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <limits>
#include <cfloat>

#include <omp.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

#include <CL/cl.h>
#include "lodepng/lodepng.h"
#define MAX_PLATFORMS 10
#define MAX_INFO_SIZE 1024
#define TEST_MATRIX_SIZE 100

#define WINDOW_SIZE 9
#define MAX_DISPARITY (260 / 4)

void displayPlatformInfo(cl_platform_id platform, cl_platform_info param_name, const char* param_str) {
    char info[MAX_INFO_SIZE];
    size_t info_size;

    clGetPlatformInfo(platform, param_name, sizeof(info), info, &info_size);
    info[info_size] = '\0';
    printf("%s: %s\n", param_str, info);
}

float* allocate_matrix() {
    return (float*)malloc(TEST_MATRIX_SIZE * TEST_MATRIX_SIZE * sizeof(float));
}

void initialize_matrix(float* matrix) {
    for (int i = 0; i < TEST_MATRIX_SIZE * TEST_MATRIX_SIZE; i++) {
        matrix[i] = (float)(i % 10);
    }
}

void add_Matrix(float* matrix_1, float* matrix_2, float* result) {
    for (int i = 0; i < TEST_MATRIX_SIZE * TEST_MATRIX_SIZE; i++) {
        result[i] = matrix_1[i] + matrix_2[i];
    }
}

void print_matrix(float* matrix) {
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            printf("%.2f ", matrix[i * TEST_MATRIX_SIZE + j]);
        }
        printf("\n");
    }
}

unsigned char* ReadImage(const char* filename, unsigned* width, unsigned* height) {
    unsigned char* image = NULL;
    unsigned error = lodepng_decode32_file(&image, width, height, filename);
    if (error) {
        printf("Error reading image %s: %s\n", filename, lodepng_error_text(error));
        exit(1);
    }
    return image;
}

unsigned char* ResizeImage(const unsigned char* image, unsigned width, unsigned height, unsigned* new_width, unsigned* new_height) {
    *new_width = width / 4;
    *new_height = height / 4;
    unsigned char* resized_image = (unsigned char*)malloc(*new_width * *new_height * 4 * sizeof(unsigned char));

    if (resized_image == NULL) {
        printf("Memory allocation failed for resized_image.\n");
        exit(1);
    }

    for (unsigned y = 0; y < *new_height; y++) {
        for (unsigned x = 0; x < *new_width; x++) {
            unsigned src_index = (y * 4 * width + x * 4) * 4;
            unsigned dst_index = (y * *new_width + x) * 4;
            resized_image[dst_index] = image[src_index];         // R
            resized_image[dst_index + 1] = image[src_index + 1]; // G
            resized_image[dst_index + 2] = image[src_index + 2]; // B
            resized_image[dst_index + 3] = image[src_index + 3]; // A
        }
    }
    return resized_image;
}

unsigned char* GrayScaleImage(const unsigned char* image, unsigned width, unsigned height) {
    unsigned char* gray_image = (unsigned char*)malloc(width * height * sizeof(unsigned char));

    if (gray_image == NULL) {
        printf("Memory allocation failed for gray_image.\n");
        exit(1);
    }

    for (unsigned y = 0; y < height; y++) {
        for (unsigned x = 0; x < width; x++) {
            unsigned index = (y * width + x) * 4;
            unsigned char r = image[index];
            unsigned char g = image[index + 1];
            unsigned char b = image[index + 2];
            gray_image[y * width + x] = (unsigned char)(0.2126 * r + 0.7152 * g + 0.0722 * b);
        }
    }
    return gray_image;
}


unsigned char* ApplyFilter(const unsigned char* image, unsigned width, unsigned height) {
    unsigned char* filtered_image = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    int kernel[5][5] = {
        {1, 4, 6, 4, 1},
        {4, 16, 24, 16, 4},
        {6, 24, 36, 24, 6},
        {4, 16, 24, 16, 4},
        {1, 4, 6, 4, 1}
    };
    int kernel_sum = 256;

    if (filtered_image == NULL) {
        printf("Memory allocation failed for gray_image.\n");
        exit(1);
    }

    for (unsigned y = 2; y < height - 2; y++) {
        for (unsigned x = 2; x < width - 2; x++) {
            int sum = 0;
            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    sum += image[(y + ky) * width + (x + kx)] * kernel[ky + 2][kx + 2];
                }
            }
            filtered_image[y * width + x] = (unsigned char)(sum / kernel_sum);
        }
    }
    return filtered_image;
}

// Function to write an image using LodePNG
void WriteImage(const char* filename, const unsigned char* image, unsigned width, unsigned height) {
    unsigned error = lodepng_encode_file(filename, image, width, height, LCT_GREY, 8);
    if (error) {
        printf("Error writing image %s: %s\n", filename, lodepng_error_text(error));
        exit(1);
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


//more logging for build errors
void check_build_error(cl_int err, cl_program program, cl_device_id device, const char* operation) {
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Error during operation '%s' (Error Code: %d)\n", operation, err);
        printf("Build Log:\n%s\n", log);
        free(log);
        exit(1);
    }
}

// foolproofed reading of kernel holy hell
char* read_kernel_source(const char* filename, size_t* size) {
    FILE* file = nullptr;
    errno_t err = fopen_s(&file, filename, "rb");  // Open in binary mode

    if (err != 0 || !file) {
        printf("Failed to open kernel file: %s\n", filename);
        exit(1);
    }


    fseek(file, 0, SEEK_END);
    *size = ftell(file);
    if (*size == (size_t)-1) {
        printf("Error determining file size: %s\n", filename);
        fclose(file);
        exit(1);
    }
    rewind(file);

    char* source = (char*)malloc(*size + 1);
    if (!source) {
        printf("Memory allocation failed for kernel source.\n");
        fclose(file);
        exit(1);
    }

    size_t bytesRead = fread(source, 1, *size, file);
    if (bytesRead != *size) {
        printf("Error: Only read %zu out of %zu bytes from kernel file: %s\n", bytesRead, *size, filename);
        free(source);
        fclose(file);
        exit(1);
    }

    source[*size] = '\0';
    fclose(file);
    return source;
}

cl_program build_opencl_program(cl_context context, cl_device_id device, const char* filename) {
    size_t source_size;
    char* source_str = read_kernel_source(filename, &source_size);
    cl_int err;
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, &source_size, &err);
    free(source_str);

    if (err != CL_SUCCESS) {
        printf("Failed to create OpenCL program.\n");
        exit(1);
    }

    // Build the OpenCL program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    if (err != CL_SUCCESS) {
        printf("Failed to build OpenCL program.\n");

        // Get the build log size
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the build log
        char* build_log = (char*)malloc(log_size);
        if (build_log == NULL) {
            printf("Failed to allocate memory for build log.\n");
            exit(1);
        }

        // Get the build log
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);

        // Print the build log to help with debugging
        printf("OpenCL Build Log:\n%s\n", build_log);

        // Free the build log memory
        free(build_log);

        exit(1);
    }

    return program;
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
    const cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, NULL);


    FILE* file = nullptr;
    errno_t add_matrix_err = fopen_s(&file, "add_matrix.cl", "r");  // Open in text mode

    if (add_matrix_err != 0 || !file) {
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
        TEST_MATRIX_SIZE * TEST_MATRIX_SIZE * sizeof(float), matrix_1, NULL);
    cl_mem d_matrix_2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        TEST_MATRIX_SIZE * TEST_MATRIX_SIZE * sizeof(float), matrix_2, NULL);
    cl_mem d_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, TEST_MATRIX_SIZE * TEST_MATRIX_SIZE * sizeof(float), NULL, NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_matrix_1);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_matrix_2);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_result);

    // Define global work size
    size_t global_size[2] = { TEST_MATRIX_SIZE, TEST_MATRIX_SIZE };

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
    check_error(clEnqueueReadBuffer(queue, d_result, CL_TRUE, 0, TEST_MATRIX_SIZE * TEST_MATRIX_SIZE * sizeof(float), result, 0, NULL, NULL), "Reading result");

    // Print a portion of the result
    printf("Result Matrix (First 10x10 elements):\n");
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            printf("%.2f ", result[i * TEST_MATRIX_SIZE + j]);
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

int read_image_test() {
    const char* input_filenames[] = { "im0.png", "im1.png" };
    const char* output_filenames[] = { "image_0_bw.png", "image_1_bw.png" };

    for (int i = 0; i < 2; i++) {
        const char* input_filename = input_filenames[i];
        const char* output_filename = output_filenames[i];
        unsigned width, height, new_width, new_height;

        double total_start_time = get_time();

		// Read image
        unsigned char* image = ReadImage(input_filename, &width, &height);

		// Resize image
        unsigned char* resized_image = ResizeImage(image, width, height, &new_width, &new_height);

		// Convert to grayscale
        unsigned char* gray_image = GrayScaleImage(resized_image, new_width, new_height);

		// Apply 5x5 filter
        unsigned char* filtered_image = ApplyFilter(gray_image, new_width, new_height);

		// Write image
        WriteImage(output_filename, filtered_image, new_width, new_height);

		// Free memory
        free(image);
        free(resized_image);
        free(gray_image);
        free(filtered_image);

		// Total time
        double total_end_time = get_time();
        printf("total time for processing image %s: %.6f seconds\n", input_filename, total_end_time - total_start_time);
        printf("\n");
    }

    return 0;
}

void process_image_opencl(const char* input_filename, const char* output_filename) {
    unsigned width, height, new_width, new_height;
    unsigned char* image = ReadImage(input_filename, &width, &height);
    new_width = width / 4;
    new_height = height / 4;
    size_t image_size = width * height * 4;
    size_t gray_size = new_width * new_height;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_int err;

    double total_start_time = get_time();

    err = clGetPlatformIDs(1, &platform, NULL);
    check_error(err, "Getting Platform");
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    check_error(err, "Getting Device");
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check_error(err, "Creating Context");

    const cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };

    queue = clCreateCommandQueueWithProperties(context, device, props, &err);

    check_error(err, "Creating Queue");

    // create buffers
    cl_mem d_image = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, image_size, image, &err);
    check_error(err, "Creating Image Buffer");
    cl_mem d_gray = clCreateBuffer(context, CL_MEM_WRITE_ONLY, gray_size * sizeof(cl_uchar4), NULL, &err);
    check_error(err, "Creating Gray Buffer");
    cl_mem d_filtered = clCreateBuffer(context, CL_MEM_WRITE_ONLY, gray_size * sizeof(cl_uchar), NULL, &err);
    check_error(err, "Creating Filter Buffer");

    // define filter kernel and create buffer
    int filter[5][5] = {
        {1, 4, 6, 4, 1},
        {4, 16, 24, 16, 4},
        {6, 24, 36, 24, 6},
        {4, 16, 24, 16, 4},
        {1, 4, 6, 4, 1}
    };

    cl_mem d_kernel = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * 5 * 5, filter, &err);
    check_error(err, "Creating Kernel Buffer");

    // load and build resize kernel
    size_t resize_size;
    char* resize_source = read_kernel_source("resize.cl", &resize_size);


    cl_program resize_program = clCreateProgramWithSource(context, 1, (const char**)&resize_source, NULL, &err);
    check_error(err, "Creating Resize Program");
    free(resize_source);
    err = clBuildProgram(resize_program, 1, &device, NULL, NULL, NULL);
    check_build_error(err, resize_program, device, "Building Resize Program");
    cl_kernel resize_kernel = clCreateKernel(resize_program, "resize_image", &err);
    check_error(err, "Creating Resize Kernel");

    // execute resize kernel
    size_t global_size[2] = { new_width, new_height };
    clSetKernelArg(resize_kernel, 0, sizeof(cl_mem), &d_image);
    clSetKernelArg(resize_kernel, 1, sizeof(cl_mem), &d_gray);
    clSetKernelArg(resize_kernel, 2, sizeof(int), &width);
    clSetKernelArg(resize_kernel, 3, sizeof(int), &height);
    clSetKernelArg(resize_kernel, 4, sizeof(int), &new_width);
    clSetKernelArg(resize_kernel, 5, sizeof(int), &new_height);
    cl_event resize_event;
    err = clEnqueueNDRangeKernel(queue, resize_kernel, 2, NULL, global_size, NULL, 0, NULL, &resize_event);
    check_error(err, "Running Resize Kernel");
    clWaitForEvents(1, &resize_event);

    // load and build grayscale kernel
    size_t grayscale_size;
    char* grayscale_source = read_kernel_source("grayscale.cl", &grayscale_size);
    cl_program grayscale_program = clCreateProgramWithSource(context, 1, (const char**)&grayscale_source, NULL, &err);
    check_error(err, "Creating Grayscale Program");
    free(grayscale_source);
    err = clBuildProgram(grayscale_program, 1, &device, NULL, NULL, NULL);
    check_build_error(err, grayscale_program, device, "Building Grayscale Program");
    cl_kernel grayscale_kernel = clCreateKernel(grayscale_program, "grayscale_image", &err);
    check_error(err, "Creating Grayscale Kernel");

    // execute grayscale kernel
    clSetKernelArg(grayscale_kernel, 0, sizeof(cl_mem), &d_gray);
    clSetKernelArg(grayscale_kernel, 1, sizeof(cl_mem), &d_filtered);
    clSetKernelArg(grayscale_kernel, 2, sizeof(int), &new_width);
    clSetKernelArg(grayscale_kernel, 3, sizeof(int), &new_height);
    cl_event grayscale_event;
    err = clEnqueueNDRangeKernel(queue, grayscale_kernel, 2, NULL, global_size, NULL, 0, NULL, &grayscale_event);
    check_error(err, "Running Grayscale Kernel");
    clWaitForEvents(1, &grayscale_event);

    // load and build filter kernel
    size_t filter_size;
    char* filter_source = read_kernel_source("filter.cl", &filter_size);
    cl_program filter_program = clCreateProgramWithSource(context, 1, (const char**)&filter_source, NULL, &err);
    check_error(err, "Creating Filter Program");
    free(filter_source);
    err = clBuildProgram(filter_program, 1, &device, NULL, NULL, NULL);
    check_build_error(err, filter_program, device, "Building Filter Program");
    cl_kernel filter_kernel = clCreateKernel(filter_program, "apply_filter", &err);
    check_error(err, "Creating Filter Kernel");

    // execute filter kernel
    clSetKernelArg(filter_kernel, 0, sizeof(cl_mem), &d_filtered);
    clSetKernelArg(filter_kernel, 1, sizeof(cl_mem), &d_filtered);
    clSetKernelArg(filter_kernel, 2, sizeof(cl_mem), &d_kernel);
    clSetKernelArg(filter_kernel, 3, sizeof(int), &new_width);
    clSetKernelArg(filter_kernel, 4, sizeof(int), &new_height);
    cl_event filter_event;
    err = clEnqueueNDRangeKernel(queue, filter_kernel, 2, NULL, global_size, NULL, 0, NULL, &filter_event);
    check_error(err, "Running Filter Kernel");
    clWaitForEvents(1, &filter_event);

    // read the output image
    unsigned char* output_image = (unsigned char*)malloc(gray_size);
    clEnqueueReadBuffer(queue, d_filtered, CL_TRUE, 0, gray_size, output_image, 0, NULL, NULL);
    WriteImage(output_filename, output_image, new_width, new_height);

    // end total timing for the entire process
    double total_end_time = get_time();
    printf("Total time for processing image %s: %.6f seconds\n", input_filename, total_end_time - total_start_time);
    printf("\n");

    // cleanup
    clReleaseMemObject(d_image);
    clReleaseMemObject(d_gray);
    clReleaseMemObject(d_filtered);
    clReleaseMemObject(d_kernel);
    clReleaseKernel(resize_kernel);
    clReleaseKernel(grayscale_kernel);
    clReleaseKernel(filter_kernel);
    clReleaseProgram(resize_program);
    clReleaseProgram(grayscale_program);
    clReleaseProgram(filter_program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(image);
    free(output_image);
}

float compute_mean(const unsigned char* img, unsigned width, unsigned x, unsigned y, unsigned win_size) {
    float sum = 0.0;
    for (int j = -(int)win_size / 2; j <= (int)win_size / 2; j++) {
        for (int i = -(int)win_size / 2; i <= (int)win_size / 2; i++) {
            sum += img[(y + j) * width + (x + i)];
        }
    }
    return sum / (win_size * win_size);
}

float compute_zncc(const unsigned char* left, const unsigned char* right, unsigned width, unsigned x, unsigned y, int d, unsigned win_size) {
    float meanL = compute_mean(left, width, x, y, win_size);
    float meanR = compute_mean(right, width, x - d, y, win_size);

    float numerator = 0.0f, denomL = 0.0f, denomR = 0.0f;
    for (int j = -(int)win_size / 2; j <= (int)win_size / 2; j++) {
        for (int i = -(int)win_size / 2; i <= (int)win_size / 2; i++) {
            float l_val = left[(y + j) * width + (x + i)] - meanL;
            float r_val = right[(y + j) * width + (x + i - d)] - meanR;
            numerator += l_val * r_val;
            denomL += l_val * l_val;
            denomR += r_val * r_val;
        }
    }
    return numerator / (sqrtf(denomL * denomR) + 1e-5f);
}

unsigned char* compute_disparity(const unsigned char* left, const unsigned char* right, unsigned width, unsigned height, int reverse) {
    unsigned char* disparity_map = (unsigned char*)malloc(width * height * sizeof(unsigned char));

    #pragma omp parallel for collapse(3)
    for (int y = WINDOW_SIZE / 2; y < height - WINDOW_SIZE / 2; y++) {
        for (unsigned x = WINDOW_SIZE / 2; x < width - WINDOW_SIZE / 2; x++) {
            float max_zncc = -FLT_MAX;
            unsigned char best_d = 0;

            for (unsigned d = 0; d <= MAX_DISPARITY; d++) {
                if ((!reverse && x < d + (int)(WINDOW_SIZE / 2)) || (reverse && x + d >= width - (int)(WINDOW_SIZE / 2))) continue;
                float zncc_val = compute_zncc(
                    reverse ? right : left,
                    reverse ? left : right,
                    width,
                    x,
                    y,
                    reverse ? -(int)d : (int)d,
                    WINDOW_SIZE
                );
                if (zncc_val > max_zncc) {
                    max_zncc = zncc_val;
                    best_d = (unsigned char)(d * 255 / MAX_DISPARITY);
                }
            }
            disparity_map[y * width + x] = best_d;
        }
    }
    return disparity_map;
}

unsigned char* cross_check_disparity(const unsigned char* left_disparity, const unsigned char* right_disparity, unsigned width, unsigned height, unsigned threshold) {
    unsigned char* consolidated_map = (unsigned char*)malloc(width * height * sizeof(unsigned char));

    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int y = 0; y < height; y++) {
        for (unsigned x = 0; x < width; x++) {
            unsigned char dL = left_disparity[y * width + x];
            unsigned char dR = right_disparity[y * width + x];
            consolidated_map[y * width + x] = (abs(dL - dR) > threshold) ? 0 : dL;
        }
    }
    return consolidated_map;
}

void occlusion_filling(unsigned char* disparity_map, unsigned width, unsigned height) {
    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int y = 0; y < height; y++) {
        for (unsigned x = 0; x < width; x++) {
            if (disparity_map[y * width + x] == 0) {
                for (int dx = -5; dx <= 5; dx++) {
                    int nx = x + dx;
                    if (nx >= 0 && nx < (int)width && disparity_map[y * width + nx] > 0) {
                        disparity_map[y * width + x] = disparity_map[y * width + nx];
                        break;
                    }
                }
            }
        }
    }
}

void stereo_disparity_cpp() {
    unsigned width, height, new_width, new_height;
    unsigned char* left_image = ReadImage("im0.png", &width, &height);
    unsigned char* right_image = ReadImage("im1.png", &width, &height);

    double start_time = omp_get_wtime();
    unsigned char* left_resized = ResizeImage(left_image, width, height, &new_width, &new_height);
    unsigned char* right_resized = ResizeImage(right_image, width, height, &new_width, &new_height);
    unsigned char* left_gray = GrayScaleImage(left_resized, new_width, new_height);
    unsigned char* right_gray = GrayScaleImage(right_resized, new_width, new_height);
    unsigned char* left_filtered = ApplyFilter(left_gray, new_width, new_height);
    unsigned char* right_filtered = ApplyFilter(right_gray, new_width, new_height);

    unsigned char* left_disparity = compute_disparity(left_filtered, right_filtered, new_width, new_height, 0);
    unsigned char* right_disparity = compute_disparity(left_filtered, right_filtered, new_width, new_height, 1);

    unsigned char* consolidated_disparity = cross_check_disparity(left_disparity, right_disparity, new_width, new_height, 8);
    occlusion_filling(consolidated_disparity, new_width, new_height);

    double end_time = omp_get_wtime();
    printf("Execution time %f seconds\n", end_time - start_time);
    // Execution time 9.3 seconds with ryzen 7 5800x3d
    // 1.3 seconds with opm

    WriteImage("disparity_map.png", left_disparity, new_width, new_height);
    WriteImage("disparity_map2.png", right_disparity, new_width, new_height);
    WriteImage("final_disparity.png", consolidated_disparity, new_width, new_height);

    free(left_image); free(right_image);
    free(left_resized); free(right_resized);
    free(left_gray); free(right_gray);
    free(left_disparity); free(right_disparity);
    free(consolidated_disparity);
}

double stereo_disparity_opencl(const char* left_image_path, const char* right_image_path, const char* output_path) {
    unsigned width = 0, height = 0;
    unsigned new_width = 0, new_height = 0;
    unsigned int window_size = WINDOW_SIZE;
    unsigned char* left_image = ReadImage(left_image_path, &width, &height);
    unsigned char* right_image = ReadImage(right_image_path, &width, &height);

    new_width = width / 4;
    new_height = height / 4;
    size_t global_work_size[2] = { new_width, new_height };
    int max_disp = MAX_DISPARITY;

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    check_error(err, "Getting Platform");
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    check_error(err, "Getting Device");
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check_error(err, "Creating Context");
    const cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    queue = clCreateCommandQueueWithProperties(context, device, props, &err);
    check_error(err, "Creating Queue");

    // load and compile kernel programs
    cl_program resize_program = build_opencl_program(context, device, "resize.cl");
    cl_program grayscale_program = build_opencl_program(context, device, "grayscale.cl");
    cl_program zncc_program = build_opencl_program(context, device, "zncc_kernel.cl");
	cl_program cross_check_program = build_opencl_program(context, device, "cross_check_kernel.cl");
    cl_program occlusion_program = build_opencl_program(context, device, "occlusion_fill_kernel.cl");

    // create kernels
    cl_kernel resize_kernel = clCreateKernel(resize_program, "resize_image", &err);
    check_error(err, "Creating Resize Kernel");
    cl_kernel grayscale_kernel = clCreateKernel(grayscale_program, "grayscale_image", &err);
    check_error(err, "Creating Grayscale Kernel");
    cl_kernel zncc_kernel = clCreateKernel(zncc_program, "zncc_kernel", &err);
    check_error(err, "Creating ZNCC Kernel");
	cl_kernel cross_check_kernel = clCreateKernel(cross_check_program, "cross_check_disparity", &err);
	check_error(err, "Creating Cross Check Kernel");
	cl_kernel occlusion_kernel = clCreateKernel(occlusion_program, "occlusion_filling", &err);
	check_error(err, "Creating Occlusion Fill Kernel");

    // create buffers
    cl_mem left_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, width * height * sizeof(cl_uchar4), left_image, &err);
    check_error(err, "Creating Left Buffer");
    cl_mem right_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, width * height * sizeof(cl_uchar4), right_image, &err);
    check_error(err, "Creating Right Buffer");

    cl_mem resized_left_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, new_width * new_height * sizeof(cl_uchar4), NULL, &err);
    check_error(err, "Creating Resized Left Buffer");
    cl_mem resized_right_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, new_width * new_height * sizeof(cl_uchar4), NULL, &err);
    check_error(err, "Creating Resized Right Buffer");

    cl_mem gray_left_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, new_width * new_height * sizeof(unsigned char), NULL, &err);
    check_error(err, "Creating Grayscale Left Buffer");
    cl_mem gray_right_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, new_width * new_height * sizeof(unsigned char), NULL, &err);
    check_error(err, "Creating Grayscale Right Buffer");

    cl_mem disparity_buf_LR = clCreateBuffer(context, CL_MEM_WRITE_ONLY, new_width * new_height * sizeof(unsigned char), NULL, &err);
    check_error(err, "Creating Left-to-Right Disparity Buffer");
    cl_mem disparity_buf_RL = clCreateBuffer(context, CL_MEM_WRITE_ONLY, new_width * new_height * sizeof(unsigned char), NULL, &err);
    check_error(err, "Creating Right-to-Left Disparity Buffer");

    cl_mem cross_checked_disparity_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, new_width * new_height * sizeof(unsigned char), NULL, &err);
    check_error(err, "Creating Cross-Checked Disparity Buffer");

	cl_mem occlusion_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, new_width * new_height * sizeof(unsigned char), NULL, &err);
	check_error(err, "Creating Occlusion Buffer");

    double start_time = omp_get_wtime();

    // resize left image
    clSetKernelArg(resize_kernel, 0, sizeof(cl_mem), &left_buf);
    clSetKernelArg(resize_kernel, 1, sizeof(cl_mem), &resized_left_buf);
    clSetKernelArg(resize_kernel, 2, sizeof(int), &width);
    clSetKernelArg(resize_kernel, 3, sizeof(int), &height);
    clSetKernelArg(resize_kernel, 4, sizeof(int), &new_width);
    clSetKernelArg(resize_kernel, 5, sizeof(int), &new_height);
    check_error(clEnqueueNDRangeKernel(queue, resize_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL), "Running Resize Kernel on Left Image");

    // resize right image
    clSetKernelArg(resize_kernel, 0, sizeof(cl_mem), &right_buf);
    clSetKernelArg(resize_kernel, 1, sizeof(cl_mem), &resized_right_buf);
    check_error(clEnqueueNDRangeKernel(queue, resize_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL), "Running Resize Kernel on Right Image");

    // convert left image to grayscale
    clSetKernelArg(grayscale_kernel, 0, sizeof(cl_mem), &resized_left_buf);
    clSetKernelArg(grayscale_kernel, 1, sizeof(cl_mem), &gray_left_buf);
    clSetKernelArg(grayscale_kernel, 2, sizeof(int), &new_width);
    clSetKernelArg(grayscale_kernel, 3, sizeof(int), &new_height);
    check_error(clEnqueueNDRangeKernel(queue, grayscale_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL), "Running Grayscale Kernel on Left Image");

    // convert right image to grayscale
    clSetKernelArg(grayscale_kernel, 0, sizeof(cl_mem), &resized_right_buf);
    clSetKernelArg(grayscale_kernel, 1, sizeof(cl_mem), &gray_right_buf);
    clSetKernelArg(grayscale_kernel, 2, sizeof(int), &new_width);
    clSetKernelArg(grayscale_kernel, 3, sizeof(int), &new_height);
    check_error(clEnqueueNDRangeKernel(queue, grayscale_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL), "Running Grayscale Kernel on Right Image");

    size_t local_work_size_zncc[2] = { 16, 16 };
    size_t global_work_size_zncc[2] = { ((new_width + 15) / 16) * 16, ((new_height + 15) / 16) * 16 };

    // compute disparity left to right
    int reverse = 0;
    clSetKernelArg(zncc_kernel, 0, sizeof(cl_mem), &gray_left_buf);
    clSetKernelArg(zncc_kernel, 1, sizeof(cl_mem), &gray_right_buf);
    clSetKernelArg(zncc_kernel, 2, sizeof(cl_mem), &disparity_buf_LR);
    clSetKernelArg(zncc_kernel, 3, sizeof(int), &new_width);
    clSetKernelArg(zncc_kernel, 4, sizeof(int), &new_height);
    clSetKernelArg(zncc_kernel, 5, sizeof(unsigned int), &window_size);
    clSetKernelArg(zncc_kernel, 6, sizeof(int), &max_disp);
    clSetKernelArg(zncc_kernel, 7, sizeof(int), &reverse);
    check_error(clEnqueueNDRangeKernel(queue, zncc_kernel, 2, NULL, global_work_size_zncc, local_work_size_zncc, 0, NULL, NULL), "Running ZNCC Kernel (Left->Right)");

    /*
    unsigned char* disparity_map_LR = (unsigned char*)malloc(new_width * new_height * sizeof(unsigned char));
    check_error(clEnqueueReadBuffer(queue, disparity_buf_LR, CL_TRUE, 0, new_width* new_height * sizeof(unsigned char), disparity_map_LR, 0, NULL, NULL), "Reading Left->Right Disparity Buffer");
	WriteImage("disparity_LR.png", disparity_map_LR, new_width, new_height);
    */

    // compute disparity right to left
	reverse = 1;
    clSetKernelArg(zncc_kernel, 0, sizeof(cl_mem), &gray_right_buf);
    clSetKernelArg(zncc_kernel, 1, sizeof(cl_mem), &gray_left_buf);
    clSetKernelArg(zncc_kernel, 2, sizeof(cl_mem), &disparity_buf_RL);
    clSetKernelArg(zncc_kernel, 3, sizeof(int), &new_width);
    clSetKernelArg(zncc_kernel, 4, sizeof(int), &new_height);
    clSetKernelArg(zncc_kernel, 5, sizeof(unsigned int), &window_size);
    clSetKernelArg(zncc_kernel, 6, sizeof(int), &max_disp);
    clSetKernelArg(zncc_kernel, 7, sizeof(int), &reverse);
    check_error(clEnqueueNDRangeKernel(queue, zncc_kernel, 2, NULL, global_work_size_zncc, local_work_size_zncc, 0, NULL, NULL), "Running ZNCC Kernel (Right->Left)");

    /*
    unsigned char* disparity_map_RL = (unsigned char*)malloc(new_width * new_height * sizeof(unsigned char));
    check_error(clEnqueueReadBuffer(queue, disparity_buf_RL, CL_TRUE, 0, new_width* new_height * sizeof(unsigned char), disparity_map_RL, 0, NULL, NULL), "Reading Right->Left Disparity Buffer");
	WriteImage("disparity_RL.png", disparity_map_RL, new_width, new_height);
    */

	// cross-check disparity maps
    int threshold = 128;
    clSetKernelArg(cross_check_kernel, 0, sizeof(cl_mem), &disparity_buf_LR);
    clSetKernelArg(cross_check_kernel, 1, sizeof(cl_mem), &disparity_buf_RL);
    clSetKernelArg(cross_check_kernel, 2, sizeof(cl_mem), &cross_checked_disparity_buf);
    clSetKernelArg(cross_check_kernel, 3, sizeof(int), &new_width);
    clSetKernelArg(cross_check_kernel, 4, sizeof(int), &new_height);
    clSetKernelArg(cross_check_kernel, 5, sizeof(int), &threshold);
    check_error(clEnqueueNDRangeKernel(queue, cross_check_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL), "Running Cross-Check Kernel");

    /*
	unsigned char* cross_checked_disparity_map = (unsigned char*)malloc(new_width * new_height * sizeof(unsigned char));
	check_error(clEnqueueReadBuffer(queue, cross_checked_disparity_buf, CL_TRUE, 0, new_width* new_height * sizeof(unsigned char), cross_checked_disparity_map, 0, NULL, NULL), "Reading Cross-Checked Disparity Buffer");
	WriteImage("cross_checked_disparity.png", cross_checked_disparity_map, new_width, new_height);
    */
   
    // apply occlusion filling
    clSetKernelArg(occlusion_kernel, 0, sizeof(cl_mem), &cross_checked_disparity_buf);
	clSetKernelArg(occlusion_kernel, 1, sizeof(cl_mem), &occlusion_buf);
    clSetKernelArg(occlusion_kernel, 2, sizeof(int), &new_width);
    clSetKernelArg(occlusion_kernel, 3, sizeof(int), &new_height);
    check_error(clEnqueueNDRangeKernel(queue, occlusion_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL), "Running Occlusion Fill Kernel");

    // read disparity map
    unsigned char* disparity_map = (unsigned char*)malloc(new_width * new_height * sizeof(unsigned char));
    check_error(clEnqueueReadBuffer(queue, occlusion_buf, CL_TRUE, 0, new_width* new_height * sizeof(unsigned char), disparity_map, 0, NULL, NULL), "Reading Occlusion Buffer");

    // execution time 0.15 seconds with RTX 3060ti
    double end_time = omp_get_wtime();
    double run_time = end_time - start_time;
    printf("Opencl execution time %f seconds\n", end_time - start_time);

	// save output image
    WriteImage(output_path, disparity_map, new_width, new_height);

    // cleaning up
    free(left_image);
    free(right_image);
    free(disparity_map);
    clReleaseMemObject(left_buf);
    clReleaseMemObject(right_buf);
    clReleaseMemObject(resized_left_buf);
    clReleaseMemObject(resized_right_buf);
    clReleaseMemObject(gray_left_buf);
    clReleaseMemObject(gray_right_buf);
    clReleaseMemObject(disparity_buf_LR);
    clReleaseMemObject(disparity_buf_RL);
    clReleaseMemObject(cross_checked_disparity_buf);
    clReleaseMemObject(occlusion_buf);
    clReleaseKernel(resize_kernel);
    clReleaseKernel(grayscale_kernel);
    clReleaseKernel(zncc_kernel);
    clReleaseKernel(cross_check_kernel);
    clReleaseKernel(occlusion_kernel);
    clReleaseProgram(resize_program);
    clReleaseProgram(grayscale_program);
    clReleaseProgram(zncc_program);
    clReleaseProgram(cross_check_program);
    clReleaseProgram(occlusion_program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return run_time;
}

int main() {
    double total_time = 0.0;

    for (int i = 0; i < 10; ++i) {
        printf("Run %d:\n", i + 1);
        double exec_time = stereo_disparity_opencl("im0.png", "im1.png", "disparity_opencl.png");
        total_time += exec_time;
    }

    double average_time = total_time / 10.0;
    printf("Average OpenCL execution time over 10 runs: %f ms\n", 1000 * average_time);

    return 0;
}
