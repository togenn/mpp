__kernel void occlusion_filling(
    __global uchar* disparity_map,
    __global uchar* occlusion_filled_map,
    const unsigned int width,
    const unsigned int height) {

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) {
        int index = y * width + x;
        occlusion_filled_map[index] = disparity_map[index];

        if (occlusion_filled_map[index] == 0) {
            for (int dx = -5; dx <= 5; dx++) {
                int nx = x + dx;
                if (nx >= 0 && nx < (int)width && disparity_map[y * width + nx] > 0) {
                    occlusion_filled_map[index] = disparity_map[y * width + nx];
                    break;
                }
            }
        }
    }
}

