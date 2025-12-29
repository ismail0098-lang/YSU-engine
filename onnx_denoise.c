#include "onnx_denoise.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "onnxruntime_c_api.h"   // third_party/onnxruntime/include

static int ysu_env_int(const char *name, int defv) {
    const char *s = getenv(name);
    if (!s || !s[0]) return defv;
    return atoi(s);
}

static const char* ysu_env_str(const char *name, const char *defv) {
    const char *s = getenv(name);
    return (s && s[0]) ? s : defv;
}

static inline float clamp01(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

static void pixels_to_nchw(const Vec3 *pixels, float *dst, int w, int h) {
    // dst: [1,3,h,w] in row-major contiguous
    // channel-first: R plane, then G, then B
    const int hw = w * h;
    float *r = dst + 0 * hw;
    float *g = dst + 1 * hw;
    float *b = dst + 2 * hw;

    for (int i = 0; i < hw; ++i) {
        // Assume pixels already linear 0..1. If yours are HDR, tonemap first.
        r[i] = clamp01(pixels[i].x);
        g[i] = clamp01(pixels[i].y);
        b[i] = clamp01(pixels[i].z);
    }
}

static void nchw_to_pixels(const float *src, Vec3 *pixels, int w, int h) {
    const int hw = w * h;
    const float *r = src + 0 * hw;
    const float *g = src + 1 * hw;
    const float *b = src + 2 * hw;

    for (int i = 0; i < hw; ++i) {
        pixels[i].x = clamp01(r[i]);
        pixels[i].y = clamp01(g[i]);
        pixels[i].z = clamp01(b[i]);
    }
}

void ysu_neural_denoise_maybe(Vec3 *pixels, int width, int height)
{
    if (!pixels || width <= 0 || height <= 0) return;

    int enabled = ysu_env_int("YSU_NEURAL_DENOISE", 0) ? 1 : 0;
    if (!enabled) return;

    const char *model_path = ysu_env_str("YSU_ONNX_MODEL", "");
    if (!model_path[0]) {
        printf("[ONNX] YSU_ONNX_MODEL not set -> skip.\n");
        return;
    }

    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!api) {
        printf("[ONNX] OrtGetApi failed.\n");
        return;
    }

    OrtEnv *env = NULL;
    OrtSessionOptions *so = NULL;
    OrtSession *sess = NULL;
    OrtMemoryInfo *mem = NULL;

    // 1) Env
    if (api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "YSU", &env) != NULL) goto fail;

    // 2) SessionOptions
    if (api->CreateSessionOptions(&so) != NULL) goto fail;

    // Önerim: render zaten MT -> ORT tek/az thread
    int intra = ysu_env_int("YSU_ONNX_INTRA", 1);
    int inter = ysu_env_int("YSU_ONNX_INTER", 1);
    api->SetIntraOpNumThreads(so, intra);
    api->SetInterOpNumThreads(so, inter);

    // 3) Session
    {
        OrtStatus *st = api->CreateSession(env, model_path, so, &sess);
        if (st) {
            const char *msg = api->GetErrorMessage(st);
            printf("[ONNX] CreateSession failed: %s\n", msg ? msg : "(null)");
            api->ReleaseStatus(st);
            goto fail;
        }
    }

    // 4) Alloc / MemoryInfo
    if (api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem) != NULL) goto fail;

    // 5) Prepare input tensor
    const int hw = width * height;
    size_t tensor_elems = (size_t)3 * (size_t)hw;
    size_t tensor_bytes = tensor_elems * sizeof(float);

    float *in_buf  = (float*)malloc(tensor_bytes);
    float *out_buf = (float*)malloc(tensor_bytes);
    if (!in_buf || !out_buf) goto fail;

    pixels_to_nchw(pixels, in_buf, width, height);

    int64_t dims[4] = {1, 3, (int64_t)height, (int64_t)width};
    OrtValue *input_tensor = NULL;
    OrtValue *output_tensor = NULL;

    {
        OrtStatus *st = api->CreateTensorWithDataAsOrtValue(
            mem, in_buf, tensor_bytes, dims, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor
        );
        if (st) {
            api->ReleaseStatus(st);
            goto fail2;
        }
    }

    // 6) Names (basit yaklaşım: modelin ilk input/output ismini al)
    OrtAllocator *allocator = NULL;
    api->GetAllocatorWithDefaultOptions(&allocator);

    char *in_name = NULL;
    char *out_name = NULL;

    api->SessionGetInputName(sess, 0, allocator, &in_name);
    api->SessionGetOutputName(sess, 0, allocator, &out_name);

    if (!in_name || !out_name) {
        printf("[ONNX] Could not query input/output names.\n");
        goto fail3;
    }

    const char* input_names[1] = { in_name };
    const char* output_names[1] = { out_name };

    // 7) Run
    {
        OrtRunOptions *ro = NULL;
        api->CreateRunOptions(&ro);

        OrtStatus *st = api->Run(sess, ro,
                                input_names, (const OrtValue* const*)&input_tensor, 1,
                                output_names, 1,
                                &output_tensor);

        api->ReleaseRunOptions(ro);

        if (st) {
            const char *msg = api->GetErrorMessage(st);
            printf("[ONNX] Run failed: %s\n", msg ? msg : "(null)");
            api->ReleaseStatus(st);
            goto fail3;
        }
    }

    // 8) Copy output back
    {
        float *out_data = NULL;
        api->GetTensorMutableData(output_tensor, (void**)&out_data);
        if (!out_data) goto fail3;

        // output_tensor out_data zaten modelin buffer'ı; direkt pixels’e yazabiliriz.
        nchw_to_pixels(out_data, pixels, width, height);
    }

    printf("[ONNX] denoise OK (model=%s, intra=%d inter=%d)\n", model_path, intra, inter);

fail3:
    if (in_name)  allocator->Free(allocator, in_name);
    if (out_name) allocator->Free(allocator, out_name);

    if (output_tensor) api->ReleaseValue(output_tensor);
    if (input_tensor)  api->ReleaseValue(input_tensor);

fail2:
    free(in_buf);
    free(out_buf);

fail:
    if (sess) api->ReleaseSession(sess);
    if (so)   api->ReleaseSessionOptions(so);
    if (mem)  api->ReleaseMemoryInfo(mem);
    if (env)  api->ReleaseEnv(env);
}
