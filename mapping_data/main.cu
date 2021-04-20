#include <iostream>

#define CHANNELS 3

__global__
void colorToGreyscaleConversion(unsigned char *Pout, unsigned char *Pin, int width, int height) {
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;

    if (Col < width && Row < height) {
        // get 1D coordinate for the grayscale image.
        // This is the linearization of the picture 2D array.
        int greyOffset = Row * width + Col;

        int rgbOffset = greyOffset*CHANNELS;
        unsigned char r = Pin[rgbOffset + 1];
        unsigned char g = Pin[rgbOffset + 2];
        unsigned char b = Pin[rgbOffset + 3];

        // Perform the grey scale conversion and store it in Pout.
        Pout[greyOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

int main() {

    return 0;
}
