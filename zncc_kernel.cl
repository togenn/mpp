__kernel void zncc_kernel(
    global const unsigned char* leftImg,
    global const unsigned char* rightImg,
    global unsigned char* disparityImg,
    int width, int height, int winSize, int maxDisp)
{
    int idx = get_global_id(0);
    int x = idx % width;
    int y = idx / width;

    // only valid pixels
    if (x < winSize / 2 || x >= width - winSize / 2 || y < winSize / 2 || y >= height - winSize / 2) {
        return;
    }

    double maxZncc = -1.0;
    int bestDisp = 0;

    // calculate mean for the left image at pixel (x, y)
    double meanL = 0.0;
    int halfWin = winSize / 2;
    for (int j = -halfWin; j <= halfWin; j++) {
        for (int i = -halfWin; i <= halfWin; i++) {
            meanL += leftImg[(y + j) * width + (x + i)];
        }
    }
    meanL /= (winSize * winSize);

    if (maxDisp > 0) {
        // left image is reference
        for (int d = 0; d < maxDisp; d++) {
            if (x - d < 0) continue;

            // calculate mean for the right image at pixel (x, y)
            double meanR = 0.0;
            for (int j = -halfWin; j <= halfWin; j++) {
                for (int i = -halfWin; i <= halfWin; i++) {
                    meanR += rightImg[(y + j) * width + (x + i - d)];
                }
            }
            meanR /= (winSize * winSize);

            // compute the ZNCC between the left and right image patches
            double numerator = 0.0;
            double denomL = 0.0;
            double denomR = 0.0;
            for (int j = -halfWin; j <= halfWin; j++) {
                for (int i = -halfWin; i <= halfWin; i++) {
                    double l_val = leftImg[(y + j) * width + (x + i)] - meanL;
                    double r_val = rightImg[(y + j) * width + (x + i - d)] - meanR;
                    numerator += l_val * r_val;
                    denomL += l_val * l_val;
                    denomR += r_val * r_val;
                }
            }

            // calculate ZNCC score
            double znccVal = numerator / (sqrt(denomL * denomR) + 1e-5);

            if (znccVal > maxZncc) {
                maxZncc = znccVal;
                bestDisp = d;
            }
        }
    }
    else {
        // right image is reference
        for (int d = 0; d > maxDisp; d--) {
            if (x - d >= width) continue;

            // calculate mean for the right image at pixel (x, y)
            double meanR = 0.0;
            for (int j = -halfWin; j <= halfWin; j++) {
                for (int i = -halfWin; i <= halfWin; i++) {
                    meanR += rightImg[(y + j) * width + (x + i - d)];
                }
            }
            meanR /= (winSize * winSize);

            // calculate mean for the left image at pixel (x, y)
            double numerator = 0.0;
            double denomL = 0.0;
            double denomR = 0.0;
            for (int j = -halfWin; j <= halfWin; j++) {
                for (int i = -halfWin; i <= halfWin; i++) {
                    double l_val = leftImg[(y + j) * width + (x + i)] - meanL;
                    double r_val = rightImg[(y + j) * width + (x + i - d)] - meanR;
                    numerator += l_val * r_val;
                    denomL += l_val * l_val;
                    denomR += r_val * r_val;
                }
            }

            // calculate ZNCC score
            double znccVal = numerator / (sqrt(denomL * denomR) + 1e-5);

            if (znccVal > maxZncc) {
                maxZncc = znccVal;
                bestDisp = d;
            }
        }
    }

    disparityImg[idx] = (unsigned char)(bestDisp + abs(maxDisp));
}
