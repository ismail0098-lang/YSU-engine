
// Large switch: 32 cases (should trigger BRX jump table)
extern "C" __global__ void __launch_bounds__(128)
cf_switch32(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    float v=in[i];
    int sel=(int)(v*32.0f)&31;
    switch(sel){
    case 0: v=v*1.01f;break; case 1: v=v*1.02f;break;
    case 2: v=v*1.03f;break; case 3: v=v*1.04f;break;
    case 4: v=v*1.05f;break; case 5: v=v*1.06f;break;
    case 6: v=v*1.07f;break; case 7: v=v*1.08f;break;
    case 8: v=v*1.09f;break; case 9: v=v*1.10f;break;
    case 10:v=v*1.11f;break; case 11:v=v*1.12f;break;
    case 12:v=v*1.13f;break; case 13:v=v*1.14f;break;
    case 14:v=v*1.15f;break; case 15:v=v*1.16f;break;
    case 16:v=v*1.17f;break; case 17:v=v*1.18f;break;
    case 18:v=v*1.19f;break; case 19:v=v*1.20f;break;
    case 20:v=v*1.21f;break; case 21:v=v*1.22f;break;
    case 22:v=v*1.23f;break; case 23:v=v*1.24f;break;
    case 24:v=v*1.25f;break; case 25:v=v*1.26f;break;
    case 26:v=v*1.27f;break; case 27:v=v*1.28f;break;
    case 28:v=v*1.29f;break; case 29:v=v*1.30f;break;
    case 30:v=v*1.31f;break; case 31:v=v*1.32f;break;
    }
    out[i]=v;
}
