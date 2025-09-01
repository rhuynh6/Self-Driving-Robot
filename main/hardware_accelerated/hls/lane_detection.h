#pragma once
#include <hls_math.h>
#include <ap_fixed.h>
#include "ap_axi_sdata.h"
#include "hls_stream.h"

#define IMG_HEIGHT 180
#define IMG_WIDTH 320
#define MAX_COMPONENT_POINTS 5000
#define MAX_COMPONENTS 200
#define MIN_HEIGHT 30
#define MIN_AREA 500

typedef ap_uint<1> bin_t;
typedef ap_uint<8> pix_t;
typedef ap_uint<10> coord_t;
typedef float fixed_t;
typedef ap_axis<32, 2, 5, 8> axis_data;

struct Point {
    coord_t x;
    coord_t y;
};

struct ComponentStats {
    int x, y, width, height, area;
    fixed_t cx, cy;
};

void lane_detection(hls::stream<axis_data> &in, hls::stream<axis_data> &out);
