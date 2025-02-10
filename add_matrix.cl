__kernel void add_matrix(__global const float *matrix_1, 
                         __global const float *matrix_2, 
                         __global float *result) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int index = j * get_global_size(0) + i;  
    result[index] = matrix_1[index] + matrix_2[index];
}
