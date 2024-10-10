
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <complex>

const int SIZE_FRAGM = 2048;
const int HOP = 128;
const double PI = 3.141592;
const double scale = 0.6;

char file_source[] = "iamete-kudasai_vDfhVBw.wav";
char file_result[] = "result_kudasai.wav";

struct HEADER
{
    unsigned char riff[4];              // RIFF string
    unsigned int overall_size;          // overall size of file in bytes
    unsigned char wave[4];              // WAVE string
    unsigned char fmt_chunk_marker[4];  // fmt string with trailing null char
    unsigned int length_of_fmt;         // length of the format data
    unsigned int format_type;           // format type. 1-PCM, 3- IEEE float, 6 - 8bit A law, 7 - 8bit mu law
    unsigned int channels;              // no.of channels
    unsigned int sample_rate;           // sampling rate (blocks per second)
    unsigned int byterate;              // SampleRate * NumChannels * BitsPerSample/8
    unsigned int block_align;           // NumChannels * BitsPerSample/8
    unsigned int bits_per_sample;       // bits per sample, 8- 8bits, 16- 16 bits etc
    unsigned char data_chunk_header[4]; // DATA string or FLLR string
    unsigned int data_size;             // NumSamples * NumChannels * BitsPerSample/8 - size of the next chunk that will be read
};

// Read header and get data from wav file
void get_samples(int16_t **data_, int *size);

double window(int t)
{
    return sin(PI * sin(PI * t / SIZE_FRAGM) * sin(PI * t / SIZE_FRAGM));
    // return 0.5 * (1 - cos(2 * PI * t / SIZE_FRAGM));
}

typedef std::complex<double> Complex;
// Fast Furie Transform
void fft(std::vector<Complex> &a, bool invert)
{
    int n = a.size();
    if (n == 1)
        return;

    std::vector<Complex> a0(n / 2), a1(n / 2);
    for (int i = 0; 2 * i < n; i++)
    {
        a0[i] = a[2 * i];
        a1[i] = a[2 * i + 1];
    }
    fft(a0, invert);
    fft(a1, invert);

    double ang = 2 * PI / n * (invert ? -1 : 1);
    Complex w(1), wn(cos(ang), sin(ang));
    for (int i = 0; 2 * i < n; i++)
    {
        a[i] = a0[i] + w * a1[i];
        a[i + n / 2] = a0[i] - w * a1[i];
        if (invert)
        {
            a[i] /= 2;
            a[i + n / 2] /= 2;
        }
        w *= wn;
    }
}

struct HEADER header;

int main(int argc, char **argv)
{
    int16_t *full_data = 0x0;

    int size = 0;
    get_samples(&full_data, &size);

    double *result_data_double = 0x0;
    result_data_double = (double *)calloc(size / scale + HOP, sizeof(double));
    std::vector<Complex> first(SIZE_FRAGM);

    std::vector<Complex> second(SIZE_FRAGM);
#pragma omp parallel for
    for (int j = 0; j < SIZE_FRAGM; j++)
    {
        Complex tmp(full_data[j] * window(j), 0);
        first[j] = tmp;
    }
    fft(first, 0);
    for (int i = 1; i < (size - SIZE_FRAGM) / HOP; i++)
    {
        if (i % 100 == 0)
        {
            printf("%f\n", i * 1.0 / ((size - SIZE_FRAGM) / HOP ));
        }
#pragma omp parallel for
        for (int j = 0; j < SIZE_FRAGM; j++)
        {
            Complex tmp(full_data[j + i * HOP] * window(j), 0);
            second[j] = tmp;
        }

        fft(second, 0);

#pragma omp parallel for
        for (int j = 0; j < SIZE_FRAGM; j++)
        {

            double magnitude = sqrt(norm(second[j]));
            double phase1 = arg(first[j]);
            second[j] = std::polar(magnitude, (arg(second[j]) - phase1) * scale + phase1);
        }

        fft(second, 1);
        int new_hop = (int)(HOP / scale);
#pragma omp parallel for
        for (int j = 0; j < SIZE_FRAGM; j++)
        {
            result_data_double[j + i * new_hop] += second[j].real() / SIZE_FRAGM * HOP * 2; //segfault, increase size of result_data
            first[j] = second[j];
        }
        fft(first, 0);
    }

    int16_t *result_data = 0x0;
    result_data = (int16_t *)malloc(sizeof(int16_t) * size / scale);
#pragma omp parallel for
    for (int i = 0; i < (int)(size / scale); i++)
    {
        if (full_data[i] == EOF)
        {
            continue;
        }
        result_data[i] = full_data[i];
        if (i > 44)
        {
            result_data[i] = (int16_t)result_data_double[i];
            // std::cout << cmp_data[64] << " " << result_data_double[64] << " " << (int16_t)full_data[i] << std::endl;
        }
    }

    result_data[(int)(size / scale) - 1] = EOF;

    //write result
    FILE *ptr;
    ptr = fopen(file_source, "rb");
    fread(&header, sizeof(char), 44, ptr);
    fclose(ptr);
    ptr = fopen(file_result, "wb");
    header.data_size *= 1.0 / scale;
    //header.byterate *= scale * 1.0 / 1.9;
    header.sample_rate = 0;
    fwrite(&header, sizeof(char), 44, ptr);
    fwrite(result_data, sizeof(int16_t), size / scale, ptr);
    fclose(ptr);

    // free
    free(full_data);
    free(result_data);
    free(result_data_double);
    return 0;
}

void get_samples(int16_t **data_, int *size)
{
    unsigned char buffer4[4];
    unsigned char buffer2[2];

    FILE *ptr;

    ptr = fopen(file_source, "rb");
    if (ptr == NULL)
    {
        printf("Error opening file\\n");
        exit(1);
    }

    int read = 0;

    // read header parts

    read = fread(header.riff, sizeof(header.riff), 1, ptr);
    // printf("(1-4): %s \n", header.riff);

    read = fread(buffer4, sizeof(buffer4), 1, ptr);
    // printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);

    // convert little endian to big endian 4 byte int
    header.overall_size = buffer4[0] |
                          (buffer4[1] << 8) |
                          (buffer4[2] << 16) |
                          (buffer4[3] << 24);

    // printf("(5-8) Overall size: bytes:%u, Kb:%u \n", header.overall_size, header.overall_size / 1024);

    read = fread(header.wave, sizeof(header.wave), 1, ptr);
    // printf("(9-12) Wave marker: %s\n", header.wave);

    read = fread(header.fmt_chunk_marker, sizeof(header.fmt_chunk_marker), 1, ptr);
    // printf("(13-16) Fmt marker: %s\n", header.fmt_chunk_marker);

    read = fread(buffer4, sizeof(buffer4), 1, ptr);
    // printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);

    // convert little endian to big endian 4 byte integer
    header.length_of_fmt = buffer4[0] |
                           (buffer4[1] << 8) |
                           (buffer4[2] << 16) |
                           (buffer4[3] << 24);
    // printf("(17-20) Length of Fmt header: %u \n", header.length_of_fmt);

    read = fread(buffer2, sizeof(buffer2), 1, ptr);
    // printf("%u %u \n", buffer2[0], buffer2[1]);

    header.format_type = buffer2[0] | (buffer2[1] << 8);
    char format_name[10] = "";
    if (header.format_type == 1)
        strcpy(format_name, "PCM");
    else if (header.format_type == 6)
        strcpy(format_name, "A-law");
    else if (header.format_type == 7)
        strcpy(format_name, "Mu-law");

    // printf("(21-22) Format type: %u %s \n", header.format_type, format_name);

    read = fread(buffer2, sizeof(buffer2), 1, ptr);
    // printf("%u %u \n", buffer2[0], buffer2[1]);

    header.channels = buffer2[0] | (buffer2[1] << 8);
    // printf("(23-24) Channels: %u \n", header.channels);

    read = fread(buffer4, sizeof(buffer4), 1, ptr);
    // printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);

    header.sample_rate = buffer4[0] |
                         (buffer4[1] << 8) |
                         (buffer4[2] << 16) |
                         (buffer4[3] << 24);

    //printf("(25-28) Sample rate: %u\n", header.sample_rate);

    read = fread(buffer4, sizeof(buffer4), 1, ptr);
    // printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);

    header.byterate = buffer4[0] |
                      (buffer4[1] << 8) |
                      (buffer4[2] << 16) |
                      (buffer4[3] << 24);
    //printf("(29-32) Byte Rate: %u , Bit Rate:%u\n", header.byterate, header.byterate * 8);

    read = fread(buffer2, sizeof(buffer2), 1, ptr);
    // printf("%u %u \n", buffer2[0], buffer2[1]);

    header.block_align = buffer2[0] |
                         (buffer2[1] << 8);
    // printf("(33-34) Block Alignment: %u \n", header.block_align);

    read = fread(buffer2, sizeof(buffer2), 1, ptr);
    // printf("%u %u \n", buffer2[0], buffer2[1]);

    header.bits_per_sample = buffer2[0] |
                             (buffer2[1] << 8);
    // printf("(35-36) Bits per sample: %u \n", header.bits_per_sample);

    read = fread(header.data_chunk_header, sizeof(header.data_chunk_header), 1, ptr);
    // printf("(37-40) Data Marker: %s \n", header.data_chunk_header);

    read = fread(buffer4, sizeof(buffer4), 1, ptr);
    // printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);

    header.data_size = buffer4[0] |
                       (buffer4[1] << 8) |
                       (buffer4[2] << 16) |
                       (buffer4[3] << 24);
    // printf("(41-44) Size of data chunk: %u \n", header.data_size);

    // calculate no.of samples
    long num_samples = (8 * header.data_size) / (header.channels * header.bits_per_sample);
    // printf("Number of samples:%lu \n", num_samples);

    long size_of_each_sample = (header.channels * header.bits_per_sample) / 8;
    // printf("Size of each sample:%ld bytes\n", size_of_each_sample);
    // printf("Closing file..\n");

    *size = header.overall_size;

    *data_ = (int16_t *)malloc(sizeof(int16_t) * (*size));

    read = fread(*data_, sizeof(int16_t), *size, ptr);
    //printf("size %d\n", *size);
    fclose(ptr);
}
