// Shared Cook-Torrance PBR helper snippet for ShaderPreprocessor #include workflows.
#ifndef GLENGINE_PBR_BRDF_GLSL
#define GLENGINE_PBR_BRDF_GLSL

const float GLENGINE_PI = 3.14159265359;

float glengineDistributionGGX(float nDotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float denom = (nDotH * nDotH) * (a2 - 1.0) + 1.0;
    return a2 / max(GLENGINE_PI * denom * denom, 1e-7);
}

vec3 glengineFresnelSchlick(float cosTheta, vec3 f0) {
    float oneMinusCos = clamp(1.0 - cosTheta, 0.0, 1.0);
    float f = oneMinusCos * oneMinusCos;
    return f0 + (vec3(1.0) - f0) * (f * f * oneMinusCos);
}

#endif
