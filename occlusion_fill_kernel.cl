__kernel void occlusion_filling(
    __global uchar* disparity_map,
    __global uchar* occlusion_filled_map,
    const unsigned int width,
    const unsigned int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int idx = y * width + x;
    const int RADIUS = 9;
	const int MAX_WINDOW_SIZE = (2 * RADIUS + 1) * (2 * RADIUS + 1);
	const int NUM_ITERATIONS = 3;

    if (x >= width || y >= height) return;

    // local pointers to swap buffers internally
    __global uchar* in_buf = disparity_map;
    __global uchar* out_buf = occlusion_filled_map;

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
		// barrier to synchronize threads
        barrier(CLK_LOCAL_MEM_FENCE);

        uchar center = in_buf[idx];

        if (center != 0) {
            out_buf[idx] = center;
        }
        else {
            // collect valid disparities from neighborhood
            uchar values[MAX_WINDOW_SIZE];
            int count = 0;

            for (int dy = -RADIUS; dy <= RADIUS; dy++) {
                for (int dx = -RADIUS; dx <= RADIUS; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;

                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        uchar val = in_buf[ny * width + nx];
                        if (val != 0) {
                            values[count++] = val;
                        }
                    }
                }
            }

            if (count > 0) {
                // bubble sort, valid for small arrays
                for (int i = 0; i < count - 1; i++) {
                    for (int j = 0; j < count - i - 1; j++) {
                        if (values[j] > values[j + 1]) {
                            uchar tmp = values[j];
                            values[j] = values[j + 1];
                            values[j + 1] = tmp;
                        }
                    }
                }
                out_buf[idx] = values[count / 2];
            }
            else {
                out_buf[idx] = 0;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // swap buffers for next iteration
        __global uchar* tmp = in_buf;
        in_buf = out_buf;
        out_buf = tmp;
    }

    disparity_map[idx] = in_buf[idx];
}
