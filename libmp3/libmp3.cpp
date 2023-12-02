#define PY_SSIZE_T_CLEAN

#include <Python.h>

#define MINIMP3_ONLY_MP3
#define MINIMP3_IMPLEMENTATION
#define MINIMP3_FLOAT_OUTPUT

#include <cstring>
#include "minimp3.h"
#include "minimp3_ex.h"
#include "iostream"

extern "C" {
struct mp3_info {
    int samples;
    int channels;
    int hz;
    int bitrate_kbps;
};

mp3_info mp3_probe_buffer(unsigned char *input_buffer, int input_size) {
    int err;
    mp3dec_ex_t dec;
    mp3_info info;

    err = mp3dec_ex_open_buf(&dec, input_buffer, input_size, MP3D_SEEK_TO_SAMPLE);

    if (err || !dec.info.channels || !dec.info.hz || !dec.info.bitrate_kbps) {
        info = {-1, -1, -1, -1};
    } else {
        info = {
                static_cast<int>(dec.samples / dec.info.channels),
                dec.info.channels,
                dec.info.hz,
                dec.info.bitrate_kbps,
        };
    }

    mp3dec_ex_close(&dec);
    return info;
}

mp3_info mp3_probe_file(const char *filename) {
    int err;
    mp3dec_ex_t dec;
    mp3_info info;

    err = mp3dec_ex_open(&dec, filename, MP3D_SEEK_TO_SAMPLE);

    if (err || !dec.info.channels || !dec.info.hz || !dec.info.bitrate_kbps) {
        info = {-1, -1, -1, -1};
    } else {
        info = {
                static_cast<int>(dec.samples / dec.info.channels),
                dec.info.channels,
                dec.info.hz,
                dec.info.bitrate_kbps,
        };
    }

    mp3dec_ex_close(&dec);
    return info;
}


int mp3_decode_buffer(unsigned char *input_buffer, int input_size,
                      float *output_buffer, int output_size,
                      long start, long length) {
    int err;
    size_t max_read;
    size_t read;
    mp3dec_ex_t dec;

    err = mp3dec_ex_open_buf(&dec, input_buffer, input_size, MP3D_SEEK_TO_SAMPLE | MP3D_DO_NOT_SCAN);

    if (err || !dec.info.channels || !dec.info.hz || !dec.info.bitrate_kbps) { return -100; }

    if (start) {
        err = mp3dec_ex_seek(&dec, start * dec.info.channels);
        if (err) { return -200; }
    }

    max_read = output_size;
    if (length) {
        if (length * dec.info.channels < max_read) {
            max_read = length * dec.info.channels;
        }
    }

    read = mp3dec_ex_read(&dec, output_buffer, max_read);

    if (read != max_read) {
        if (dec.last_error) { return dec.last_error; }
    }

    mp3dec_ex_close(&dec);
    return read;
}

int mp3_decode_file(const char *filename,
                    float *output_buffer, int output_size,
                    long start, long length) {
    int err;
    size_t max_read;
    size_t read;
    mp3dec_ex_t dec;

    err = mp3dec_ex_open(&dec, filename, MP3D_SEEK_TO_SAMPLE | MP3D_DO_NOT_SCAN);

    if (err || !dec.info.channels || !dec.info.hz || !dec.info.bitrate_kbps) { return -100; }

    if (start) {
        err = mp3dec_ex_seek(&dec, start * dec.info.channels);
        if (err) { return -200; }
    }

    max_read = output_size;
    if (length) {
        if (length * dec.info.channels < max_read) {
            max_read = length * dec.info.channels;
        }
    }

    read = mp3dec_ex_read(&dec, output_buffer, max_read);

    if (read != max_read) {
        if (dec.last_error) { return dec.last_error; }
    }

    mp3dec_ex_close(&dec);
    return read;
}


int mp3_decode_slow(unsigned char *input_buffer, int input_size, float *output_buffer) {
    mp3dec_t mp3d;
    mp3dec_file_info_t info;

    mp3dec_load_buf(&mp3d,
                    input_buffer,
                    input_size,
                    &info,
                    nullptr,
                    nullptr);

    // if (info.samples == 0) throw std::runtime_error("mp3: could not read any data");

    // copy the data to the output buffer
    std::memcpy(output_buffer, info.buffer, sizeof(float) * info.samples);
    free(info.buffer);
    return info.samples;
}


mp3dec_file_info_t mp3_decode_slow2(unsigned char *input_buffer, int input_size) {
    mp3dec_t mp3d;
    mp3dec_file_info_t info;

    mp3dec_load_buf(&mp3d,
                    input_buffer,
                    input_size,
                    &info,
                    nullptr,
                    nullptr);

    return info;
}

void free_mp3dec_file_info_t(mp3dec_file_info_t info) {
    free(info.buffer);
}

int unpackbits(unsigned char *src, int src_size, unsigned char *dst) {
    for (int byte = 0; byte < src_size; ++byte) {

        const int index = byte * 8;
        const int value = src[byte];

        for (int bit = 0; bit < 8; ++bit) {
            const uint8_t mask = 7 - bit;
            dst[index + bit] = ((value & (uint8_t{1} << mask)) >> mask);
        }
    }
    return 0;
}
// can be improved by using a lookup table: https://github.com/numpy/numpy/blob/main/numpy/core/src/multiarray/compiled_base.c#L1754


static PyMethodDef libmp3Methods[] = {
        {NULL, NULL, 0, NULL}
};


static struct PyModuleDef libmp3vmodule = {
        PyModuleDef_HEAD_INIT,
        "libmp3", /* name of module */
        "This is a dummy python extension, the real code is available through ctypes", /* module documentation, may be NULL */
        -1,  /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
        libmp3Methods
};

PyMODINIT_FUNC
PyInit__libmp3(void) {
    return PyModule_Create(&libmp3vmodule);
}

}