__kernel void zncc_kernel(
    __global const unsigned char* leftImg,
    __global const unsigned char* rightImg,
    __global unsigned char* disparityImg,
    const int width, const int height, 
    const int winSize, const int maxDisp, const int reverse)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < winSize / 2 || x >= width - winSize / 2 || 
        y < winSize / 2 || y >= height - winSize / 2)
    {
        disparityImg[y * width + x] = 0;
        return;
    }

    float maxZNCC = -FLT_MAX;
    unsigned char bestD = 0;

    for (int d = 0; d <= maxDisp; d++) {
        if ((!reverse && x < d + winSize / 2) || 
            (reverse && x + d >= width - winSize / 2)) continue;

        float sumL = 0.0, sumR = 0.0, sumL2 = 0.0, sumR2 = 0.0, sumLR = 0.0;
        int count = 0;

        for (int j = -winSize / 2; j <= winSize / 2; j++) {
            for (int i = -winSize / 2; i <= winSize / 2; i++) {
                int leftX = x + i;
                int leftY = y + j;
                int rightX = (reverse) ? leftX + d : leftX - d;

                if (rightX < 0 || rightX >= width) continue;

                float leftVal = leftImg[leftY * width + leftX];
                float rightVal = rightImg[leftY * width + rightX];

                sumL += leftVal;
                sumR += rightVal;
                sumL2 += leftVal * leftVal;
                sumR2 += rightVal * rightVal;
                sumLR += leftVal * rightVal;
                count++;
            }
        }

        if (count == 0) continue;

        float meanL = sumL / count;
        float meanR = sumR / count;

        float varL = sumL2 - count * meanL * meanL;
        float varR = sumR2 - count * meanR * meanR;
        float covLR = sumLR - count * meanL * meanR;

        float zncc = covLR / (sqrt(varL * varR) + 1e-5f);

        if (zncc > maxZNCC) {
            maxZNCC = zncc;
            bestD = (unsigned char)(d * 255 / maxDisp);
        }
    }

    disparityImg[y * width + x] = bestD;
}