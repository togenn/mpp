__kernel void cross_check_disparity(
    __global const uchar* left_disparity,
    __global const uchar* right_disparity,
    __global uchar* consolidated_map,
    const unsigned int width,
    const unsigned int height,
    const unsigned int threshold) {

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) {
        int index = y * width + x;
        uchar dL = left_disparity[index];
        uchar dR = right_disparity[index];
        consolidated_map[index] = (abs(dL - dR) > threshold) ? 0 : dL;
    }
}

