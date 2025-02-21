__kernel void resize_image(__global const uchar4* input, __global uchar4* output, int width, int height, int new_width, int new_height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < new_width && y < new_height) {
        int src_x = x * 4;
        int src_y = y * 4;
        output[y * new_width + x] = input[src_y * width + src_x];
    }
}