__kernel void apply_filter(__global const unsigned char* input, __global unsigned char* output, __global const int* filter, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= 2 && x < width - 2 && y >= 2 && y < height - 2) {
        int sum = 0;
        for (int ky = -2; ky <= 2; ky++) {
            for (int kx = -2; kx <= 2; kx++) {
                sum += input[(y + ky) * width + (x + kx)] * filter[(ky + 2) * 5 + (kx + 2)];
            }
        }
        output[y * width + x] = (uchar)(sum / 256);
    }
}