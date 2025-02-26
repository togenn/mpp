#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <limits.h>
#include <float.h>

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
	FILE* file = fopen(filename, "rb"); // open in binary mode why not brrrrrrrr
    if (!file) {
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
    queue = clCreateCommandQueueWithProperties(context, device, (cl_queue_properties[]) { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 }, & err);
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

    float numerator = 0.0f;
    float denomL = 0.0f;
    float denomR = 0.0f;

    for (int j = -(int)win_size / 2; j <= (int)win_size / 2; j++) {
        for (int i = -(int)win_size / 2; i <= (int)win_size / 2; i++) {
            float l_val = left[(y + j) * width + (x + i)] - meanL;
            float r_val = right[(y + j) * width + (x + i - d)] - meanR;
            numerator += l_val * r_val;
            denomL += l_val * l_val;
            denomR += r_val * r_val;
        }
    }
    return numerator / (sqrtf(denomL * denomR));
}

unsigned char* compute_disparity(const unsigned char* left, const unsigned char* right, unsigned width, unsigned height, int reverse) {
    unsigned char* disparity_map = (unsigned char*)malloc(width * height * sizeof(unsigned char));

    for (unsigned y = WINDOW_SIZE / 2; y < height - WINDOW_SIZE / 2; y++) {
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
                    best_d = (unsigned char)(abs(d) * 255 / MAX_DISPARITY); // Store positive disparity
                }
            }

            disparity_map[y * width + x] = best_d;
        }
    }

    return disparity_map;
}

// Perform cross-checking to consolidate disparity maps
unsigned char* cross_check_disparity(const unsigned char* left_disparity, const unsigned char* right_disparity, unsigned width, unsigned height, unsigned threshold) {
    unsigned char* consolidated_map = (unsigned char*)malloc(width * height * sizeof(unsigned char));

    for (unsigned y = 0; y < height; y++) {
        for (unsigned x = 0; x < width; x++) {
            unsigned char dL = left_disparity[y * width + x];
            unsigned char dR = right_disparity[y * width + x];

            if (abs(dL - dR) > threshold) {
                consolidated_map[y * width + x] = 0;
            }
            else {
                consolidated_map[y * width + x] = dL;
            }
        }
    }

    return consolidated_map;
}


int is_valid_pixel(unsigned char val) {
    return val > 0;
}

void occlusion_filling(unsigned char* disparity_map, unsigned width, unsigned height) {
    unsigned char* filled_disparity = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    if (!filled_disparity) {
        printf("Memory allocation failed.\n");
        return;
    }

    memcpy(filled_disparity, disparity_map, width * height * sizeof(unsigned char));

    for (unsigned y = 0; y < height; y++) {
        for (unsigned x = 0; x < width; x++) {
            if (disparity_map[y * width + x] == 0) {
                unsigned char candidates[4] = { 0, 0, 0, 0 };
                float weights[4] = { 0, 0, 0, 0 };

                // Scan left
                for (int lx = x - 1; lx >= 0; lx--) {
                    if (is_valid_pixel(disparity_map[y * width + lx])) {
                        candidates[0] = disparity_map[y * width + lx];
                        weights[0] = 1.0f / (x - lx);
                        break;
                    }
                }

                // Scan right
                for (int rx = x + 1; rx < (int)width; rx++) {
                    if (is_valid_pixel(disparity_map[y * width + rx])) {
                        candidates[1] = disparity_map[y * width + rx];
                        weights[1] = 1.0f / (rx - x);
                        break;
                    }
                }

                // Scan up
                for (int uy = y - 1; uy >= 0; uy--) {
                    if (is_valid_pixel(disparity_map[uy * width + x])) {
                        candidates[2] = disparity_map[uy * width + x];
                        weights[2] = 1.0f / (y - uy);
                        break;
                    }
                }

                // Scan down
                for (int dy = y + 1; dy < (int)height; dy++) {
                    if (is_valid_pixel(disparity_map[dy * width + x])) {
                        candidates[3] = disparity_map[dy * width + x];
                        weights[3] = 1.0f / (dy - y);
                        break;
                    }
                }

                float weighted_sum = 0.0f;
                float total_weight = 0.0f;

                for (int i = 0; i < 4; i++) {
                    if (candidates[i] > 0) {
                        weighted_sum += candidates[i] * weights[i];
                        total_weight += weights[i];
                    }
                }

                if (total_weight > 0) {
                    filled_disparity[y * width + x] = (unsigned char)(weighted_sum / total_weight);
                }
            }
        }
    }

    memcpy(disparity_map, filled_disparity, width * height * sizeof(unsigned char));
    free(filled_disparity);
}

int main() {
    unsigned width, height, new_width, new_height;

    unsigned char* left_image = ReadImage("im0.png", &width, &height);
    unsigned char* right_image = ReadImage("im1.png", &width, &height);

    double start_time = get_time();

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

    double end_time = get_time();

    printf("Execution time %f seconds", end_time - start_time);
    // Execution time 37.296770 seconds with ryzen 7 5800x3d

    WriteImage("disparity_map.png", left_disparity, new_width, new_height);
    WriteImage("disparity_map2.png", right_disparity, new_width, new_height);
    WriteImage("final_disparity.png", consolidated_disparity, new_width, new_height);

    free(left_image);
    free(right_image);
    free(left_resized);
    free(right_resized);
    free(left_gray);
    free(right_gray);
    free(left_disparity);
    free(right_disparity);

    return 0;
}
