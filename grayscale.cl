__kernel void grayscale_image(__global const uchar4* input, __global uchar* output, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) {
        int idx = y * width + x;
        uchar4 pixel = input[idx];
        output[idx] = (uchar)(0.2126f * pixel.x + 0.7152f * pixel.y + 0.0722f * pixel.z);
    }
}