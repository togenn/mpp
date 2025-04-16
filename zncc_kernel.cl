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
        y < winSize / 2 || y >= height - winSize / 2) {
        disparityImg[y * width + x] = 0;
        return;
    }
    float maxZNCC = -1.0f;
    unsigned char bestD = 0;
    for (int d = 0; d <= maxDisp; d++) {
        if ((!reverse && x < d + winSize / 2) ||
            (reverse && x + d >= width - winSize / 2)) continue;
        float4 sumL = 0.0f, sumR = 0.0f, sumL2 = 0.0f, sumR2 = 0.0f, sumLR = 0.0f;
        int count = 0;
        for (int j = -winSize / 2; j <= winSize / 2; j++) {
            for (int i = -winSize / 2; i <= winSize / 2 - 3; i += 4) {
                int leftX = x + i;
                int leftY = y + j;
                int rightX = reverse ? leftX + d : leftX - d;
                if (rightX < 0 || rightX + 3 >= width) continue;
                float4 leftVal = (float4)(
                    leftImg[leftY * width + leftX],
                    leftImg[leftY * width + leftX + 1],
                    leftImg[leftY * width + leftX + 2],
                    leftImg[leftY * width + leftX + 3]);
                float4 rightVal = (float4)(
                    rightImg[leftY * width + rightX],
                    rightImg[leftY * width + rightX + 1],
                    rightImg[leftY * width + rightX + 2],
                    rightImg[leftY * width + rightX + 3]);
                sumL += leftVal;
                sumR += rightVal;
                sumL2 += leftVal * leftVal;
                sumR2 += rightVal * rightVal;
                sumLR += leftVal * rightVal;
                count += 4;
            }
            // Handle remaining pixels
            for (int i = (winSize / 2 - 3) + 1; i <= winSize / 2; i++) {
                int leftX = x + i;
                int leftY = y + j;
                int rightX = reverse ? leftX + d : leftX - d;
                if (rightX < 0 || rightX >= width) continue;
                float leftVal = leftImg[leftY * width + leftX];
                float rightVal = rightImg[leftY * width + rightX];
                sumL.s0 += leftVal;
                sumR.s0 += rightVal;
                sumL2.s0 += leftVal * leftVal;
                sumR2.s0 += rightVal * rightVal;
                sumLR.s0 += leftVal * rightVal;
                count++;
            }
        }
        if (count == 0) continue;
        float meanL = (sumL.s0 + sumL.s1 + sumL.s2 + sumL.s3) / count;
        float meanR = (sumR.s0 + sumR.s1 + sumR.s2 + sumR.s3) / count;
        float varL = (sumL2.s0 + sumL2.s1 + sumL2.s2 + sumL2.s3) - count * meanL * meanL;
        float varR = (sumR2.s0 + sumR2.s1 + sumR2.s2 + sumR2.s3) - count * meanR * meanR;
        float covLR = (sumLR.s0 + sumLR.s1 + sumLR.s2 + sumLR.s3) - count * meanL * meanR;
        float zncc = covLR / (sqrt(varL * varR) + 1e-5f);
        if (zncc > maxZNCC) {
            maxZNCC = zncc;
            bestD = (unsigned char)(d * 255 / maxDisp);
        }
    }
    disparityImg[y * width + x] = bestD;
}