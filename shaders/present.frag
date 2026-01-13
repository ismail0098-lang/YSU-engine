#version 450
layout(location=0) in vec2 vUV;
layout(location=0) out vec4 oColor;
layout(set=0, binding=0) uniform sampler2D uTex;
void main(){
    vec2 uv = clamp(vUV, 0.0, 1.0);
    oColor = texture(uTex, uv);
}
