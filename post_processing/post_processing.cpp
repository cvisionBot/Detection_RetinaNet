#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <io.h>
#include <stdlib.h>
#include <cassert>

using namespace std;

#define ANCHOR_COUNT 9
#define ANCHOR_TOTAL 19206
#define INPUT 320
#define CLASS 20
#define CONF 0.3f
#define MAX_OBJ 10

void decode_bbox(float* raw_data, float* anchors, int max_loc_idx)
{
    float scale[] = { 0.1f, 0.2f };
    float pred_cx, pred_cy, pred_width, pred_height;
    float make_cx, make_cy, make_width, make_height;
    int anchor_cx, anchor_cy, anchor_width, anchor_height;

    anchor_cx = max_loc_idx + 0;
    anchor_cy = max_loc_idx + 1;
    anchor_width = max_loc_idx + 2;
    anchor_height = max_loc_idx + 3;

    make_width = anchors[anchor_width] - anchors[anchor_cx];
    make_height = anchors[anchor_height] - anchors[anchor_cy];
    make_cx = 0.5f * (anchors[anchor_width] + anchors[anchor_cx]);
    make_cy = 0.5f * (anchors[anchor_height] + anchors[anchor_cy]);
    pred_cx = make_cx + (raw_data[0] * scale[0]) * make_width;
    pred_cy = make_cy + (raw_data[1] * scale[0]) * make_height;
    pred_width = make_width * expf((raw_data[2] * scale[1]));
    pred_height = make_height * expf((raw_data[3] * scale[1]));

    raw_data[0] = pred_cx - 0.5f * pred_width;
    raw_data[1] = pred_cx + 0.5f * pred_width;
    raw_data[2] = pred_cy - 0.5f * pred_height;
    raw_data[3] = pred_cy + 0.5f * pred_height;


void FsNetPostProcessingFPN(float* reg_pred, float* cls_pred)
{
    int step;
    int idx;
    int cls_idx = 0;
    int loc_idx = 0;
    float max_conf = 0.f;
    int num_obj = 0;

    float anchor_cls[20] = { 0.f, };
    float pred_boxes[4] = { 0.f, };

    for (step = 0; step < CLASS; step++) {
        max_conf = 0.f;
        for (idx = (step * ANCHOR_TOTAL); idx < (step+1) * ANCHOR_TOTAL; idx++) {
            if (cls_pred[idx] > max_conf) {
                max_conf = cls_pred[idx];
                cls_idx = idx;
            }
        }
        if (cls_pred[cls_idx] < CONF)
            continue;
        if (num_obj >= MAX_OBJ)
            break;

        loc_idx = cls_idx - (step *  ANCHOR_TOTAL);
        pred_boxes[0] = reg_pred[loc_idx];
        pred_boxes[1] = reg_pred[loc_idx + (1 * ANCHOR_TOTAL)];
        pred_boxes[2] = reg_pred[loc_idx + (2 * ANCHOR_TOTAL)];
        pred_boxes[3] = reg_pred[loc_idx + (3 * ANCHOR_TOTAL)];
        loc_idx = (cls_idx - (step * ANCHOR_TOTAL)) * 4;
        decode_bbox(pred_boxes, retina_anchors, loc_idx);

        anchor_cls[step] = max_conf;
        printf("= = = = = class info = = = = = \n");
        printf("%d class conf : %f \n", step, anchor_cls[step]);
        printf("= = = = = = = = = = \n");
        printf("x1 : %f \n", pred_boxes[0] * INPUT);
        printf("x2 : %f \n", pred_boxes[1] * INPUT);
        printf("y1 : %f \n", pred_boxes[2] * INPUT);
        printf("y2 : %f \n", pred_boxes[3] * INPUT);
        printf("= = = = = = = = = = \n");
    }

}


int main(void) {
    float* reg_out, * cls_out;
    char reg_filename[200];
    char cls_filename[200];
    snprintf(reg_filename, 200, "./dump/regression_out.bin");
    snprintf(cls_filename, 200, "./dump/classification_out.bin");
    reg_out = (float*)calloc(19206 * 4, sizeof(float));
    cls_out = (float*)calloc(19206 * 20, sizeof(float));

    FILE* f_reg = fopen(reg_filename, "rb");
    FILE* f_cls = fopen(cls_filename, "rb");

    fread(cls_out, 19206 * 20, sizeof(float), f_cls);
    fread(reg_out, 19206 * 4, sizeof(float), f_reg);

    FsNetPostProcessingFPN(reg_out, cls_out);

    fclose(f_cls);
    fclose(f_reg);
    free(cls_out);
    free(reg_out);

	return 0;
}