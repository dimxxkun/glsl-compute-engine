#version 460 core

layout(location = 0) in vec3  vWorldPos;
layout(location = 1) in vec3  vWorldNormal;
layout(location = 2) in vec4  vWorldTangent;
layout(location = 3) in vec2  vTexCoord;
layout(location = 4) in vec4  vBaseColor;
layout(location = 5) in vec4  vPbrFactors;
layout(location = 6) flat in uvec2 vMaterialMeta;

// Texture flag bitmask in vMaterialMeta.y
const uint FLAG_ALBEDO_MAP     = 1u << 0u;
const uint FLAG_NORMAL_MAP     = 1u << 1u;
const uint FLAG_METAL_ROUGH    = 1u << 2u;
const uint FLAG_OCCLUSION_MAP  = 1u << 3u;

layout(binding = 0) uniform sampler2D uAlbedoMap;
layout(binding = 1) uniform sampler2D uNormalMap;
layout(binding = 2) uniform sampler2D uMetallicRoughnessMap;
layout(binding = 3) uniform sampler2D uOcclusionMap;

// G-Buffer MRT outputs
layout(location = 0) out vec4 oAlbedoMetallic;   // rgb = linear albedo, a = metallic
layout(location = 1) out vec4 oNormalRoughness;  // rgb = unit world normal, a = perceptual roughness
layout(location = 2) out vec4 oPositionAO;       // rgb = world position, a = ambient occlusion

void main() {
    uint texFlags = vMaterialMeta.y;

    // 1. Albedo
    vec3 albedo = vBaseColor.rgb;
    if ((texFlags & FLAG_ALBEDO_MAP) != 0u) {
        vec4 texSample = texture(uAlbedoMap, vTexCoord);
        albedo *= texSample.rgb;
    }

    // 2. Metallic & Perceptual Roughness (glTF 2.0 channel convention: G = roughness, B = metallic)
    float metallic  = clamp(vPbrFactors.x, 0.0, 1.0);
    float roughness = clamp(vPbrFactors.y, 0.045, 1.0);
    if ((texFlags & FLAG_METAL_ROUGH) != 0u) {
        vec2 mr = texture(uMetallicRoughnessMap, vTexCoord).bg;
        metallic  = clamp(metallic * mr.x, 0.0, 1.0);
        roughness = clamp(roughness * mr.y, 0.045, 1.0);
    }

    // 3. Ambient Occlusion
    float ao = clamp(vPbrFactors.z, 0.0, 1.0);
    if ((texFlags & FLAG_OCCLUSION_MAP) != 0u) {
        ao *= texture(uOcclusionMap, vTexCoord).r;
    }

    // 4. World-space Normal (optional tangent-space normal map)
    vec3 N = normalize(vWorldNormal);
    if ((texFlags & FLAG_NORMAL_MAP) != 0u) {
        vec3 T = normalize(vWorldTangent.xyz);
        vec3 B = normalize(cross(N, T)) * vWorldTangent.w;
        mat3 TBN = mat3(T, B, N);
        vec3 tangentNormal = texture(uNormalMap, vTexCoord).xyz * 2.0 - 1.0;
        N = normalize(TBN * tangentNormal);
    }

    oAlbedoMetallic  = vec4(albedo, metallic);
    oNormalRoughness = vec4(N, roughness);
    oPositionAO      = vec4(vWorldPos, ao);
}
