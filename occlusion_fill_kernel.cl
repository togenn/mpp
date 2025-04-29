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
    const int WINDOW_SIZE = 2 * RADIUS + 1;
    const int MAX_WINDOW_SIZE = WINDOW_SIZE * WINDOW_SIZE;
    const int NUM_PASSES = 3;

    if (x >= width || y >= height) return;

    // Local memory for work-group's disparity values
    __local uchar local_disparity[16 + 2 * RADIUS][16 + 2 * RADIUS]; // 34x34 to cover 16x16 + halo
    int lx = get_local_id(0);
    int ly = get_local_id(1);

    // Load disparity map into local memory (with halo)
    for (int dy = -RADIUS; dy <= RADIUS; dy++) {
        for (int dx = -RADIUS; dx <= RADIUS; dx++) {
            int gx = x + dx;
            int gy = y + dy;
            int local_x = lx + dx + RADIUS;
            int local_y = ly + dy + RADIUS;
            if (local_x >= 0 && local_x < 16 + 2 * RADIUS && local_y >= 0 && local_y < 16 + 2 * RADIUS) {
                if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
                    local_disparity[local_y][local_x] = disparity_map[gy * width + gx];
                } else {
                    local_disparity[local_y][local_x] = 0; // Out-of-bounds
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Store intermediate results in local memory
    uchar current_value = disparity_map[idx];
    __local uchar local_result[16][16];
    local_result[ly][lx] = current_value;

    // Perform multiple passes locally
    for (int pass = 0; pass < NUM_PASSES; pass++) {
        barrier(CLK_LOCAL_MEM_FENCE); // Sync before processing
        current_value = local_result[ly][lx];

        if (current_value != 0) {
            if (pass == NUM_PASSES - 1) {
                occlusion_filled_map[idx] = current_value; // Write final result
            }
            continue;
        }

        // Collect valid disparities from local memory
        uchar values[MAX_WINDOW_SIZE];
        int count = 0;
        for (int dy = -RADIUS; dy <= RADIUS; dy++) {
            for (int dx = -RADIUS; dx <= RADIUS; dx++) {
                int local_x = lx + dx + RADIUS;
                int local_y = ly + dy + RADIUS;
                if (local_x >= 0 && local_x < 16 + 2 * RADIUS && local_y >= 0 && local_y < 16 + 2 * RADIUS) {
                    uchar val = local_disparity[local_y][local_x];
                    if (val != 0) {
                        values[count++] = val;
                    }
                }
            }
        }

        if (count > 0) {
            for (int i = 1; i < count; i++) {
                uchar key = values[i];
                int j = i - 1;
                while (j >= 0 && values[j] > key) {
                    values[j + 1] = values[j];
                    j--;
                }
                values[j + 1] = key;
            }
            current_value = values[count / 2]; // Update with median
            local_result[ly][lx] = current_value;
            if (pass == NUM_PASSES - 1) {
                occlusion_filled_map[idx] = current_value; // Write final result
            }
        } else {
            current_value = 0; // No valid neighbors
            local_result[ly][lx] = 0;
            if (pass == NUM_PASSES - 1) {
                occlusion_filled_map[idx] = 0; // Will be updated in next pass
            }
        }

        // Update local_disparity for next pass
        barrier(CLK_LOCAL_MEM_FENCE); // Sync before updating
        if (lx < 16 && ly < 16) {
            local_disparity[ly + RADIUS][lx + RADIUS] = local_result[ly][lx];
        }
    }
}