#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>

#define ALOGD(fmt, ...) fprintf(stderr, "[NPU] " fmt, ##__VA_ARGS__)
#define ALOGW(fmt, ...) fprintf(stderr, "[NPU WRN] " fmt, ##__VA_ARGS__)

#include "vip_lite.h"

typedef struct Awnn_params {
    vip_buffer_create_params_t vip_param;
    vip_uint32_t elements;
    float        scale;       /* TF asymmetric quantization scale */
    int32_t      zero_point;  /* TF asymmetric quantization zero-point */
} Awnn_params_t;

typedef struct Awnn_Context {
    pthread_mutex_t mutex;
    vip_network     network;
    vip_uint32_t    input_count;
    vip_uint32_t    output_count;
    vip_buffer     *input_buffers;
    vip_buffer     *output_buffers;
    Awnn_params_t  *input_params;
    Awnn_params_t  *output_params;
    float         **user_output_buffers;
} Awnn_Context_t;

void awnn_init(void) {
    vip_status_e s = vip_init();
    fprintf(stderr, "[NPU] vip_init=%d\n", s);
}

void awnn_uninit(void) { vip_destroy(); }

Awnn_Context_t *awnn_create(const char *nbg) {
    vip_status_e s;
    vip_uint32_t i, d;
    Awnn_Context_t *ctx = calloc(1, sizeof(Awnn_Context_t));
    if (!ctx) return NULL;

    fprintf(stderr, "[NPU] loading %s\n", nbg);
    s = vip_create_network(nbg, 0, VIP_CREATE_NETWORK_FROM_FILE, &ctx->network);
    if (s != VIP_SUCCESS) { fprintf(stderr,"[NPU] create_network failed %d\n",s); free(ctx); return NULL; }

    s = vip_prepare_network(ctx->network);
    if (s != VIP_SUCCESS) { fprintf(stderr,"[NPU] prepare_network failed %d\n",s); vip_destroy_network(ctx->network); free(ctx); return NULL; }

    vip_query_network(ctx->network, VIP_NETWORK_PROP_INPUT_COUNT,  &ctx->input_count);
    vip_query_network(ctx->network, VIP_NETWORK_PROP_OUTPUT_COUNT, &ctx->output_count);
    fprintf(stderr,"[NPU] in=%u out=%u\n", ctx->input_count, ctx->output_count);

    ctx->input_buffers       = calloc(ctx->input_count,  sizeof(vip_buffer));
    ctx->input_params        = calloc(ctx->input_count,  sizeof(Awnn_params_t));
    ctx->output_buffers      = calloc(ctx->output_count, sizeof(vip_buffer));
    ctx->output_params       = calloc(ctx->output_count, sizeof(Awnn_params_t));
    ctx->user_output_buffers = calloc(ctx->output_count, sizeof(float*));

    for (i = 0; i < ctx->input_count; i++) {
        vip_query_input(ctx->network, i, VIP_BUFFER_PROP_DATA_FORMAT,       &ctx->input_params[i].vip_param.data_format);
        vip_query_input(ctx->network, i, VIP_BUFFER_PROP_NUM_OF_DIMENSION,  &ctx->input_params[i].vip_param.num_of_dims);
        vip_query_input(ctx->network, i, VIP_BUFFER_PROP_SIZES_OF_DIMENSION, ctx->input_params[i].vip_param.sizes);
        vip_query_input(ctx->network, i, VIP_BUFFER_PROP_QUANT_FORMAT,      &ctx->input_params[i].vip_param.quant_format);
        /* Read scale and zero-point using the dedicated properties */
        vip_query_input(ctx->network, i, VIP_BUFFER_PROP_TF_SCALE,     &ctx->input_params[i].scale);
        vip_query_input(ctx->network, i, VIP_BUFFER_PROP_TF_ZERO_POINT, &ctx->input_params[i].zero_point);

        ctx->input_params[i].elements = 1;
        for (d = 0; d < ctx->input_params[i].vip_param.num_of_dims; d++)
            ctx->input_params[i].elements *= ctx->input_params[i].vip_param.sizes[d];
        fprintf(stderr,"[NPU] input[%u] elems=%u fmt=%d scale=%f zp=%d\n", i,
            ctx->input_params[i].elements,
            ctx->input_params[i].vip_param.data_format,
            ctx->input_params[i].scale,
            ctx->input_params[i].zero_point);
        vip_create_buffer(&ctx->input_params[i].vip_param, sizeof(vip_buffer_create_params_t), &ctx->input_buffers[i]);
        vip_set_input(ctx->network, i, ctx->input_buffers[i]);
    }

    for (i = 0; i < ctx->output_count; i++) {
        vip_query_output(ctx->network, i, VIP_BUFFER_PROP_DATA_FORMAT,       &ctx->output_params[i].vip_param.data_format);
        vip_query_output(ctx->network, i, VIP_BUFFER_PROP_NUM_OF_DIMENSION,  &ctx->output_params[i].vip_param.num_of_dims);
        vip_query_output(ctx->network, i, VIP_BUFFER_PROP_SIZES_OF_DIMENSION, ctx->output_params[i].vip_param.sizes);
        vip_query_output(ctx->network, i, VIP_BUFFER_PROP_QUANT_FORMAT,      &ctx->output_params[i].vip_param.quant_format);
        /* Read scale and zero-point using the dedicated properties */
        vip_query_output(ctx->network, i, VIP_BUFFER_PROP_TF_SCALE,     &ctx->output_params[i].scale);
        vip_query_output(ctx->network, i, VIP_BUFFER_PROP_TF_ZERO_POINT, &ctx->output_params[i].zero_point);

        ctx->output_params[i].elements = 1;
        for (d = 0; d < ctx->output_params[i].vip_param.num_of_dims; d++)
            ctx->output_params[i].elements *= ctx->output_params[i].vip_param.sizes[d];
        fprintf(stderr,"[NPU] output[%u] elems=%u fmt=%d scale=%f zp=%d\n", i,
            ctx->output_params[i].elements,
            ctx->output_params[i].vip_param.data_format,
            ctx->output_params[i].scale,
            ctx->output_params[i].zero_point);
        vip_create_buffer(&ctx->output_params[i].vip_param, sizeof(vip_buffer_create_params_t), &ctx->output_buffers[i]);
        vip_set_output(ctx->network, i, ctx->output_buffers[i]);
        ctx->user_output_buffers[i] = calloc(ctx->output_params[i].elements, sizeof(float));
    }
    return ctx;
}

void awnn_set_input_buffers(Awnn_Context_t *ctx, void **input_data) {
    vip_uint32_t i;
    for (i = 0; i < ctx->input_count; i++) {
        void *ptr = vip_map_buffer(ctx->input_buffers[i]);
        if (!ptr) continue;
        memcpy(ptr, input_data[i], ctx->input_params[i].elements);
        vip_flush_buffer(ctx->input_buffers[i], VIP_BUFFER_OPER_TYPE_FLUSH);
        vip_unmap_buffer(ctx->input_buffers[i]);
    }
}

void awnn_run(Awnn_Context_t *ctx) {
    vip_status_e s = vip_run_network(ctx->network);
    if (s != VIP_SUCCESS) fprintf(stderr,"[NPU] run failed %d\n", s);
}

float **awnn_get_output_buffers(Awnn_Context_t *ctx) {
    vip_uint32_t i, n;
    for (i = 0; i < ctx->output_count; i++) {
        vip_flush_buffer(ctx->output_buffers[i], VIP_BUFFER_OPER_TYPE_INVALIDATE);
        void *ptr = vip_map_buffer(ctx->output_buffers[i]);
        if (!ptr) continue;

        vip_buffer_format_e fmt = ctx->output_params[i].vip_param.data_format;
        float  scale = ctx->output_params[i].scale;
        int32_t  zp  = ctx->output_params[i].zero_point;

        if (fmt == VIP_BUFFER_FORMAT_UINT8) {
            uint8_t *raw = (uint8_t *)ptr;
            if (scale == 0.0f) scale = 1.0f; /* safety: avoid all-zeros */
            for (n = 0; n < ctx->output_params[i].elements; n++)
                ctx->user_output_buffers[i][n] = ((float)raw[n] - (float)zp) * scale;
        } else if (fmt == VIP_BUFFER_FORMAT_INT8) {
            int8_t *raw = (int8_t *)ptr;
            if (scale == 0.0f) scale = 1.0f;
            for (n = 0; n < ctx->output_params[i].elements; n++)
                ctx->user_output_buffers[i][n] = ((float)raw[n] - (float)zp) * scale;
        } else if (fmt == VIP_BUFFER_FORMAT_FP16) {
            /* FP16 → FP32 conversion */
            uint16_t *raw = (uint16_t *)ptr;
            for (n = 0; n < ctx->output_params[i].elements; n++) {
                uint32_t flt;
                uint16_t h = raw[n];
                uint32_t sign = (h >> 15) & 1;
                uint32_t exp  = (h >> 10) & 0x1f;
                uint32_t mant = h & 0x3ff;
                if (exp == 0) {
                    flt = (sign << 31) | (mant << 13);
                } else if (exp == 31) {
                    flt = (sign << 31) | (0xff << 23) | (mant << 13);
                } else {
                    flt = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
                }
                memcpy(&ctx->user_output_buffers[i][n], &flt, sizeof(float));
            }
        } else {
            /* FP32 or anything else: direct memcpy */
            memcpy(ctx->user_output_buffers[i], ptr, ctx->output_params[i].elements * sizeof(float));
        }
        vip_unmap_buffer(ctx->output_buffers[i]);
    }
    return ctx->user_output_buffers;
}

uint32_t awnn_get_output_count(Awnn_Context_t *ctx)             { return ctx->output_count; }
uint32_t awnn_get_output_elements(Awnn_Context_t *ctx, int idx) { return ctx->output_params[idx].elements; }
uint32_t awnn_get_input_elements(Awnn_Context_t *ctx, int idx)  { return ctx->input_params[idx].elements; }

/* Expose scale/zp so Python can verify or override */
float    awnn_get_output_scale(Awnn_Context_t *ctx, int idx)    { return ctx->output_params[idx].scale; }
int32_t  awnn_get_output_zp(Awnn_Context_t *ctx, int idx)       { return ctx->output_params[idx].zero_point; }

void awnn_destroy(Awnn_Context_t *ctx) {
    if (!ctx) return;
    vip_uint32_t i;
    for (i = 0; i < ctx->input_count; i++)  vip_destroy_buffer(ctx->input_buffers[i]);
    for (i = 0; i < ctx->output_count; i++) {
        vip_destroy_buffer(ctx->output_buffers[i]);
        free(ctx->user_output_buffers[i]);
    }
    free(ctx->input_buffers);  free(ctx->output_buffers);
    free(ctx->input_params);   free(ctx->output_params);
    free(ctx->user_output_buffers);
    vip_destroy_network(ctx->network);
    free(ctx);
}
