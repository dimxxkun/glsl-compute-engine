#version 460 core

layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 oFragColor;

const float PI = 3.14159265359;

struct PointLight {
    vec4 positionRadius;  // xyz = world position, w = influence radius
    vec4 colorIntensity;  // rgb = linear color, w = luminous intensity
};

layout(binding = 0) uniform sampler2D uGBufferAlbedoMetallic;
layout(binding = 1) uniform sampler2D uGBufferNormalRoughness;
layout(binding = 2) uniform sampler2D uGBufferPositionAO;

layout(std140, binding = 0) uniform LightingUBO {
    vec4  cameraPosExposure;     // xyz = camera world position, w = exposure
    vec4  dirLightDirIntensity;  // xyz = normalized direction toward directional light, w = intensity
    vec4  dirLightColorAmbient;  // rgb = directional light color, w = ambient intensity
    uvec4 tileInfo;              // x = tileCountX, y = tileCountY, z = maxLightsPerTile, w = tileSize (16)
} uLight;

layout(std430, binding = 0) readonly buffer PointLightBuffer {
    PointLight lights[];
};

layout(std430, binding = 1) readonly buffer TileLightIndexBuffer {
    uint tileLightIndices[];
};

layout(std430, binding = 2) readonly buffer TileLightCountBuffer {
    uint tileLightCounts[];
};

// -----------------------------------------------------------------------------
// Cook-Torrance Microfacet PBR BRDF
// -----------------------------------------------------------------------------
float distributionGGX(float nDotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float denom = (nDotH * nDotH) * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denom * denom, 1e-7);
}

float geometrySchlickGGX(float nDotX, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) * 0.125;
    return nDotX / max(nDotX * (1.0 - k) + k, 1e-7);
}

float geometrySmith(float nDotV, float nDotL, float roughness) {
    return geometrySchlickGGX(nDotV, roughness) * geometrySchlickGGX(nDotL, roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 f0) {
    float oneMinusCos = clamp(1.0 - cosTheta, 0.0, 1.0);
    float factor = oneMinusCos * oneMinusCos;
    factor = factor * factor * oneMinusCos; // pow(1 - cosTheta, 5)
    return f0 + (vec3(1.0) - f0) * factor;
}

vec3 evaluateCookTorrance(
    vec3 albedo,
    float metallic,
    float roughness,
    vec3 N,
    vec3 V,
    vec3 L,
    vec3 radiance
) {
    float nDotL = max(dot(N, L), 0.0);
    float nDotV = max(dot(N, V), 1e-4);
    if (nDotL <= 0.0) {
        return vec3(0.0);
    }

    vec3 H = normalize(V + L);
    float nDotH = max(dot(N, H), 0.0);
    float hDotV = max(dot(H, V), 0.0);

    vec3 f0 = mix(vec3(0.04), albedo, metallic);

    float ndf = distributionGGX(nDotH, roughness);
    float vis = geometrySmith(nDotV, nDotL, roughness);
    vec3  F   = fresnelSchlick(hDotV, f0);

    vec3 specular = (ndf * vis * F) / max(4.0 * nDotV * nDotL, 1e-5);

    // Energy-conserving Lambertian diffuse weighted by non-reflected, non-metallic fraction.
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diffuse = kD * (albedo / PI);

    return (diffuse + specular) * radiance * nDotL;
}

// Physically-based smooth-windowed inverse-square point light attenuation (Karis).
float pointLightAttenuation(float distanceToLight, float radius) {
    if (distanceToLight >= radius || radius <= 0.0) {
        return 0.0;
    }
    float ratio  = distanceToLight / radius;
    float ratio2 = ratio * ratio;
    float window = clamp(1.0 - ratio2 * ratio2, 0.0, 1.0);
    return (window * window) / (distanceToLight * distanceToLight + 1.0);
}

// Narkowicz ACES filmic tone mapping approximation.
vec3 acesFilmicToneMap(vec3 hdr) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((hdr * (a * hdr + b)) / (hdr * (c * hdr + d) + e), 0.0, 1.0);
}

void main() {
    vec4 albedoMetallic  = texture(uGBufferAlbedoMetallic, vUv);
    vec4 normalRoughness = texture(uGBufferNormalRoughness, vUv);
    vec4 positionAO      = texture(uGBufferPositionAO, vUv);

    vec3  albedo    = albedoMetallic.rgb;
    float metallic  = clamp(albedoMetallic.a, 0.0, 1.0);
    vec3  normalRaw = normalRoughness.xyz;
    float roughness = clamp(normalRoughness.a, 0.045, 1.0);
    vec3  worldPos  = positionAO.xyz;
    float ao        = clamp(positionAO.a, 0.0, 1.0);

    // Background pixels (empty G-Buffer normal) receive subtle sky gradient.
    if (dot(normalRaw, normalRaw) < 1e-4) {
        vec3 skyBottom = vec3(0.03, 0.04, 0.06);
        vec3 skyTop    = vec3(0.10, 0.16, 0.28);
        vec3 skyColor  = mix(skyBottom, skyTop, clamp(vUv.y, 0.0, 1.0));
        vec3 mappedSky = pow(acesFilmicToneMap(skyColor * uLight.cameraPosExposure.w), vec3(1.0 / 2.2));
        oFragColor = vec4(mappedSky, 1.0);
        return;
    }

    vec3 N = normalize(normalRaw);
    vec3 V = normalize(uLight.cameraPosExposure.xyz - worldPos);

    // 1. Directional key light
    vec3 dirL = normalize(uLight.dirLightDirIntensity.xyz);
    vec3 dirRadiance = uLight.dirLightColorAmbient.rgb * uLight.dirLightDirIntensity.w;
    vec3 lo = evaluateCookTorrance(albedo, metallic, roughness, N, V, dirL, dirRadiance);

    // 2. Tiled point light accumulation (reads tile list produced by tiled_light_cull.comp)
    uint tileSize = max(uLight.tileInfo.w, 1u);
    uvec2 tileCoord = uvec2(gl_FragCoord.xy) / tileSize;
    tileCoord = min(tileCoord, max(uLight.tileInfo.xy, uvec2(1u)) - uvec2(1u));

    uint tileIndex  = tileCoord.y * uLight.tileInfo.x + tileCoord.x;
    uint maxPerTile = uLight.tileInfo.z;
    uint lightCount = min(tileLightCounts[tileIndex], maxPerTile);
    uint baseOffset = tileIndex * maxPerTile;

    for (uint i = 0u; i < lightCount; ++i) {
        uint lightIdx = tileLightIndices[baseOffset + i];
        PointLight pl = lights[lightIdx];

        vec3 toLight = pl.positionRadius.xyz - worldPos;
        float dist = length(toLight);
        float atten = pointLightAttenuation(dist, pl.positionRadius.w);
        if (atten <= 0.0) {
            continue;
        }

        vec3 L = toLight / max(dist, 1e-5);
        vec3 radiance = pl.colorIntensity.rgb * (pl.colorIntensity.w * atten);
        lo += evaluateCookTorrance(albedo, metallic, roughness, N, V, L, radiance);
    }

    // 3. Hemispheric ambient term modulated by AO
    float hemi = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 ambientColor = mix(vec3(0.06, 0.05, 0.05), vec3(0.14, 0.18, 0.26), hemi);
    vec3 ambient = ambientColor * albedo * (uLight.dirLightColorAmbient.w * ao * (1.0 - 0.5 * metallic));

    vec3 hdrColor = (ambient + lo) * max(uLight.cameraPosExposure.w, 0.01);
    vec3 toneMapped = acesFilmicToneMap(hdrColor);
    vec3 srgbColor = pow(max(toneMapped, vec3(0.0)), vec3(1.0 / 2.2));

    oFragColor = vec4(srgbColor, 1.0);
}
