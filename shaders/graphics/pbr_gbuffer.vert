#version 460 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec4 aTangent;   // xyz = tangent direction, w = bitangent handedness (+1/-1)
layout(location = 3) in vec2 aTexCoord;

struct InstanceData {
    mat4  modelMatrix;
    mat4  normalMatrix;      // Precomputed transpose(inverse(modelMatrix))
    vec4  baseColorFactor;   // rgb = linear albedo factor, a = alpha
    vec4  pbrFactors;        // x = metallic, y = roughness, z = ao, w = unused
    uvec4 materialMeta;      // x = materialId, y = textureFlags, z = 0, w = 0
};

layout(std140, binding = 0) uniform CameraUBO {
    mat4 viewProjMatrix;
    mat4 viewMatrix;
    vec4 cameraPos;
} uCamera;

layout(std430, binding = 0) readonly buffer VisibleInstanceBuffer {
    uint visibleInstanceIds[];
};

layout(std430, binding = 1) readonly buffer InstanceDataBuffer {
    InstanceData instances[];
};

layout(location = 0) out vec3  vWorldPos;
layout(location = 1) out vec3  vWorldNormal;
layout(location = 2) out vec4  vWorldTangent;
layout(location = 3) out vec2  vTexCoord;
layout(location = 4) out vec4  vBaseColor;
layout(location = 5) out vec4  vPbrFactors;
layout(location = 6) flat out uvec2 vMaterialMeta;

void main() {
    // Compacted visible instance lookup for glMultiDrawElementsIndirect.
    uint visibleSlot = uint(gl_BaseInstance + gl_InstanceID);
    uint instanceId  = visibleInstanceIds[visibleSlot];

    InstanceData inst = instances[instanceId];

    vec4 worldPos = inst.modelMatrix * vec4(aPosition, 1.0);
    mat3 normalMat = mat3(inst.normalMatrix);

    vec3 worldNormal  = normalize(normalMat * aNormal);
    vec3 worldTangent = normalize(mat3(inst.modelMatrix) * aTangent.xyz);
    // Re-orthogonalize tangent against normal via Gram-Schmidt.
    worldTangent = normalize(worldTangent - worldNormal * dot(worldNormal, worldTangent));

    vWorldPos     = worldPos.xyz;
    vWorldNormal  = worldNormal;
    vWorldTangent = vec4(worldTangent, aTangent.w);
    vTexCoord     = aTexCoord;
    vBaseColor    = inst.baseColorFactor;
    vPbrFactors   = inst.pbrFactors;
    vMaterialMeta = inst.materialMeta.xy;

    gl_Position = uCamera.viewProjMatrix * worldPos;
}
