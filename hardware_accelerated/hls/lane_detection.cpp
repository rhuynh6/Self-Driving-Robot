#include "lane_detection.h"

static const fixed_t lower_white[3] = {fixed_t(0), fixed_t(210), fixed_t(0)};
static const fixed_t upper_white[3] = {fixed_t(255), fixed_t(255), fixed_t(255)};
static const fixed_t lower_yellow[3] = {fixed_t(18), fixed_t(0), fixed_t(100)};
static const fixed_t upper_yellow[3] = {fixed_t(30), fixed_t(220), fixed_t(255)};

/**
 * Translates RGB pixel to HLS.
 *
 * @param rgb The input RGB pixel.
 * @param hls The output HLS pixel.
 */
void rgb_to_hls(pix_t rgb[3], fixed_t hls[3]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=rgb complete dim=0
#pragma HLS ARRAY_PARTITION variable=hls complete dim=0
    fixed_t r = (fixed_t)rgb[0] / fixed_t(255.0);
    fixed_t g = (fixed_t)rgb[1] / fixed_t(255.0);
    fixed_t b = (fixed_t)rgb[2] / fixed_t(255.0);

    fixed_t max_val = r > g ? (r > b ? r : b) : (g > b ? g : b);
    fixed_t min_val = r < g ? (r < b ? r : b) : (g < b ? g : b);
    fixed_t delta = max_val - min_val;

    fixed_t L = (max_val + min_val) / fixed_t(2.0);
    fixed_t S = 0.0f;
    if (delta != fixed_t(0.0)) {
        S = L < fixed_t(0.5)
            ? delta / (max_val + min_val)
            : delta / (fixed_t(2.0) - max_val - min_val);
    }

    fixed_t H = 0.0;
    if (delta != fixed_t(0.0)) {
        if (max_val == r)
            H = (g - b) / delta;
        else if (max_val == g)
            H = (b - r) / delta + fixed_t(2.0);
        else
            H = (r - g) / delta + fixed_t(4.0);
        H *= fixed_t(60.0);
        if (H < 0) H += fixed_t(360.0);
    }

    hls[0] = H * fixed_t(255.0 / 360.0);
    hls[1] = L * fixed_t(255.0);
    hls[2] = S * fixed_t(255.0);
}

/**
 * Determines if an HLS pixel is within thresholding range.
 *
 * @param hls The input HLS pixel.
 * @param lower The lower bound of the range.
 * @param upper The upper bound of the range.
 * @return Boolean indicating if pixel is within provided range.
 */
bool in_range(fixed_t hls[3], const fixed_t lower[3], const fixed_t upper[3]) {
#pragma HLS INLINE
    return (hls[0] >= lower[0] && hls[0] <= upper[0]) &&
           (hls[1] >= lower[1] && hls[1] <= upper[1]) &&
           (hls[2] >= lower[2] && hls[2] <= upper[2]);
}

/**
 * Returns binary value if RGB pixel is within provided thresholding range.
 *
 * @param img The source image.
 * @param y The y coordinate of the pixel to be thresholded.
 * @param x The x coordinate of the pixel to be thresholded.
 * @return Binary value if pixel is within provided thresholding range.
 */
bin_t threshold_pixel(pix_t img[IMG_HEIGHT][IMG_WIDTH][3], int y, int x) {
#pragma HLS INLINE
    pix_t rgb[3] = {img[y][x][0], img[y][x][1], img[y][x][2]};
    fixed_t hls[3];
    rgb_to_hls(rgb, hls);
    return in_range(hls, lower_white, upper_white) || in_range(hls, lower_yellow, upper_yellow);
}

/**
 * Maps a source pixel to a destination using projective transformation.
 *
 * @param M The projective transform matrix.
 * @param x The source x coordinate.
 * @param y The source y coordinate.
 * @param outX The destination x coordinate.
 * @param outY The destination y coordinate.
 */
void perspective_transform(const fixed_t M[3][3], fixed_t x, fixed_t y, fixed_t& outX, fixed_t& outY) {
#pragma HLS INLINE
    fixed_t denom = M[2][0]*x + M[2][1]*y + M[2][2];
    if (hls::abs(denom) < fixed_t(1e-3)) denom = fixed_t(1e-3);
    outX = (M[0][0]*x + M[0][1]*y + M[0][2]) / denom;
    outY = (M[1][0]*x + M[1][1]*y + M[1][2]) / denom;
}

/**
 * Projective transform function to map an a source image to a destination image.
 *
 * @param input The source image.
 * @param warped The destination image.
 * @param M The projective transform matrix.
 */
void warp_perspective(pix_t input[IMG_HEIGHT][IMG_WIDTH][3], bin_t warped[IMG_HEIGHT][IMG_WIDTH], fixed_t M[3][3]) {
    fixed_t invM[3][3];
    fixed_t det = M[0][0]*(M[1][1]*M[2][2] - M[1][2]*M[2][1]) -
                  M[0][1]*(M[1][0]*M[2][2] - M[1][2]*M[2][0]) +
                  M[0][2]*(M[1][0]*M[2][1] - M[1][1]*M[2][0]);
    #pragma HLS ARRAY_PARTITION variable=invM complete dim=0
    #pragma HLS ARRAY_PARTITION variable=M complete dim=0

    // Calculate inverse of M for projective transform
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            #pragma HLS UNROLL
            invM[i][j] = (
                ((M[(j+1)%3][(i+1)%3] * M[(j+2)%3][(i+2)%3]) -
                (M[(j+1)%3][(i+2)%3] * M[(j+2)%3][(i+1)%3])) / det
            );

    for (int y = 0; y < IMG_HEIGHT; y++) {
        for (int x = 0; x < IMG_WIDTH; x++) {
            #pragma HLS PIPELINE II=1
            fixed_t srcX, srcY;
            perspective_transform(invM, x, y, srcX, srcY);

            int srcXint = static_cast<int>((srcX + fixed_t(0.5)));
            int srcYint = static_cast<int>((srcY + fixed_t(0.5)));
            
            // Map src pixels to destination in image
            if (srcXint >= 0 && srcXint < IMG_WIDTH && srcYint >= 0 && srcYint < IMG_HEIGHT) {
                warped[y][x] = threshold_pixel(input, srcYint, srcXint);
            } else {
                warped[y][x] = 0;
            }
        }
    }
}

/**
 * DSU algorithm used to find root parent for a point.
 *
 * @param label The index of the point.
 * @param parent Parent array.
 * @return The index of the parent of the set.
 */
int find_root(int label, int parent[MAX_COMPONENTS]) {
    while (label != parent[label]) {
        parent[label] = parent[parent[label]];  // Path compression
        label = parent[label];
    }
    return label;
}

/**
 * Takes a binary image and clusters touching pixels together.
 *
 * @param binary The binary image.
 * @param label_map DSU matrix to map points to a cluster.
 * @param stats Maintains data on clusters (Centroid, width, height, area).
 * @param num_labels Count of active points in binary image.
 */
void connected_components(bin_t binary[IMG_HEIGHT][IMG_WIDTH],
                          int label_map[IMG_HEIGHT][IMG_WIDTH],
                          ComponentStats stats[MAX_COMPONENTS],
                          int &num_labels) {
    int next_label = 1;
    int parent[MAX_COMPONENTS];
    int size[MAX_COMPONENTS];  // For union by size

    // Initialize union-find structures
    for (int i = 0; i < MAX_COMPONENTS; i++) {
        parent[i] = i;
        size[i] = 1;
    }

    // First pass: assign labels and track equivalences
    for (int y = 0; y < IMG_HEIGHT; y++) {
        for (int x = 0; x < IMG_WIDTH; x++) {
            if (binary[y][x] == 0) {
                label_map[y][x] = 0;
                continue;
            }

            int left = (x > 0) ? label_map[y][x - 1] : 0;
            int up   = (y > 0) ? label_map[y - 1][x] : 0;

            if (left == 0 && up == 0) {
                label_map[y][x] = next_label;
                parent[next_label] = next_label;
                size[next_label] = 1;
                next_label++;
            } else if (left > 0 && up == 0) {
                label_map[y][x] = left;
            } else if (up > 0 && left == 0) {
                label_map[y][x] = up;
            } else {
                int root_left = find_root(left, parent);
                int root_up   = find_root(up, parent);
                int min_root = (root_left < root_up) ? root_left : root_up;
                int max_root = (root_left > root_up) ? root_left : root_up;

                if (min_root != max_root) {
                    if (size[min_root] < size[max_root]) {
                        parent[min_root] = max_root;
                        size[max_root] += size[min_root];
                    } else {
                        parent[max_root] = min_root;
                        size[min_root] += size[max_root];
                    }
                }

                label_map[y][x] = min_root;
            }
        }
    }

    // Second pass: resolve all labels to root
    for (int y = 0; y < IMG_HEIGHT; y++) {
        for (int x = 0; x < IMG_WIDTH; x++) {
            int label = label_map[y][x];
            if (label > 0) {
                label_map[y][x] = find_root(label, parent);
            }
        }
    }

    // Initialize stats
    for (int i = 0; i < MAX_COMPONENTS; i++) {
        stats[i].x = IMG_WIDTH;
        stats[i].y = IMG_HEIGHT;
        stats[i].width = 0;
        stats[i].height = 0;
        stats[i].area = 0;
        stats[i].cx = 0;
        stats[i].cy = 0;
    }

    // Collect component stats using canonical labels
    for (int y = 0; y < IMG_HEIGHT; y++) {
        for (int x = 0; x < IMG_WIDTH; x++) {
            int label = label_map[y][x];
            if (label == 0 || label >= MAX_COMPONENTS) continue;

            ComponentStats &s = stats[label];
            s.area++;
            if (x < s.x) s.x = x;
            if (y < s.y) s.y = y;
            if (x > s.x + s.width)  s.width = x - s.x;
            if (y > s.y + s.height) s.height = y - s.y;
            s.cx += x;
            s.cy += y;
        }
    }

    // Finalize centroid calculation and count valid labels
    num_labels = 0;
    for (int i = 1; i < next_label; i++) {
        if (stats[i].area > 0) {
            stats[i].cx = stats[i].cx / stats[i].area;
            stats[i].cy = stats[i].cy / stats[i].area;
            num_labels++;
        }
    }
}

/**
 * Finds left and right lane lines from connected components.
 *
 * @param label_map DSU matrix to map points to a cluster.
 * @param left_pts Left points found.
 * @param right_pts Right points found.
 * @param stats Maintains data on clusters (Centroid, width, height, area).
 * @param left_count Count of points in left lane.
 * @param right_count Count of points in right lane.
 */
void get_lane_lines(
    int label_map[IMG_HEIGHT][IMG_WIDTH], 
    Point left_pts[MAX_COMPONENT_POINTS],
    Point right_pts[MAX_COMPONENT_POINTS],
    ComponentStats stats[MAX_COMPONENTS],
    int &left_count, int &right_count) {
    for (int y = 0; y < IMG_HEIGHT; y++) {
        for (int x = 0; x < IMG_WIDTH; x++) {
            #pragma HLS PIPELINE II=1
            int i = label_map[y][x];
            if (i >= MAX_COMPONENTS)
                continue;
            bool is_left = stats[i].cx < IMG_WIDTH / 2;
            if (stats[i].area >= MIN_AREA) {
                if (is_left && left_count < MAX_COMPONENT_POINTS) {
                    left_pts[left_count++] = {x, y};
                    label_map[y][x] = 255;
                } else if (!is_left && right_count < MAX_COMPONENT_POINTS) {
                    right_pts[right_count++] = {x, y};
                }
            }
        }
    }
}

void read_input_stream(
    hls::stream<axis_data> &in,
    axis_data &temp,
    pix_t img[IMG_HEIGHT][IMG_WIDTH][3]
) {
    for (int y = 0; y < IMG_HEIGHT; y++) {
        for (int x = 0; x < IMG_WIDTH; x++) {
            for (int c = 0; c < 3; c++) {
                #pragma HLS PIPELINE II=1
                temp = in.read();
                img[y][x][c] = temp.data;
            }
        }
    }
}

void write_output_stream(
    hls::stream<axis_data> &out,
    axis_data &temp,
    Point left_pts[MAX_COMPONENT_POINTS],
    Point right_pts[MAX_COMPONENT_POINTS],
    int left_count, int right_count
) {
    temp.data = left_count;
    temp.last = 0;
    out.write(temp);

    temp.data = right_count;
    temp.last = 0;
    out.write(temp);

    for (int side = 0; side < 2; side++) {
        for (int direction = 0; direction < 2; direction++) {
            for (int p = 0; p < MAX_COMPONENT_POINTS; p++) {
                #pragma HLS PIPELINE II=1
                if (side == 0) {
                    if (direction == 0) {
                        coord_t coord = left_pts[p].x;
                        temp.data = coord;
                    } else {
                        coord_t coord = left_pts[p].y;
                        temp.data = coord;
                    }
                } else {
                    if (direction == 0) {
                        coord_t coord = right_pts[p].x;
                        temp.data = coord;
                    } else {
                        coord_t coord = right_pts[p].y;
                        temp.data = coord;
                    }
                }

                if (side == 1 && direction == 1 && p == MAX_COMPONENT_POINTS - 1) {
                    temp.last = 1;
                } else {
                    temp.last = 0;
                }

                out.write(temp);
            }
        }
    }
}

void lane_detection(hls::stream<axis_data> &in, hls::stream<axis_data> &out) {
    #pragma HLS INTERFACE axis port=in
    #pragma HLS INTERFACE axis port=out
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    // Perspective transform matrix (homography)
    fixed_t M[3][3] = {
        {fixed_t(7.78135048e-01), fixed_t(1.37191854e-01), fixed_t(3.54983923e+01)},
        {fixed_t(-3.08780779e-16), fixed_t(4.93890675e+00), fixed_t(-5.55627010e+02)},
        {fixed_t(1.59919820e-18), fixed_t(8.57449089e-04), fixed_t(1)}
    };
    #pragma HLS ARRAY_PARTITION variable=M complete dim=0

    // Image buffers
    pix_t img[IMG_HEIGHT][IMG_WIDTH][3];
    #pragma HLS ARRAY_PARTITION variable=img complete dim=3

    bin_t warped[IMG_HEIGHT][IMG_WIDTH];
    axis_data temp;

    // Lane detection buffers
    int label_map[IMG_HEIGHT][IMG_WIDTH];
    ComponentStats stats[MAX_COMPONENTS];
    Point left_pts[MAX_COMPONENT_POINTS];
    Point right_pts[MAX_COMPONENT_POINTS];

    int num_labels = 0;
    int left_count = 0;
    int right_count = 0;

    // Processing pipeline
    read_input_stream(in, temp, img);
    warp_perspective(img, warped, M);
    connected_components(warped, label_map, stats, num_labels);
    get_lane_lines(label_map, left_pts, right_pts, stats, left_count, right_count);
    write_output_stream(out, temp, left_pts, right_pts, left_count, right_count);
}
