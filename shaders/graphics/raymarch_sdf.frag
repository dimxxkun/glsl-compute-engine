#version 460 core

layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 oFragColor;

layout(std140, binding = 0) uniform RaymarchUBO {
    vec4 resolutionTime;       // xy = viewport size, z = elapsedTime, w = fovYRadians
    vec4 cameraPosMaxDist;     // xyz = camera world pos, w = maxRayDistance
    vec4 cameraTargetExposure; // xyz = look-at target, w = exposure
    vec4 keyLightDirIntensity; // xyz = light direction (toward light), w = light intensity
} uRay;

const int   MAX_MARCH_STEPS = 128;
const float MIN_HIT_DIST    = 0.0005;

// -----------------------------------------------------------------------------
// SDF Primitives & Smooth CSG Operators
// -----------------------------------------------------------------------------
float sdSphere(vec3 p, float radius) {
    return length(p) - radius;
}

float sdRoundBox(vec3 p, vec3 halfExtents, float cornerRadius) {
    vec3 q = abs(p) - halfExtents;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - cornerRadius;
}

float sdTorus(vec3 p, vec2 radii) {
    vec2 q = vec2(length(p.xz) - radii.x, p.y);
    return length(q) - radii.y;
}

float sdCylinder(vec3 p, float radius, float halfHeight) {
    vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(radius, halfHeight);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

// Polynomial smooth minimum (smooth CSG union) returning (distance, blendFactor).
vec2 sminCubic(float a, float b, float k) {
    float h = max(k - abs(a - b), 0.0) / max(k, 1e-5);
    float m = h * h * h * 0.5;
    float s = m * k * (1.0 / 3.0);
    return (a < b) ? vec2(a - s, m) : vec2(b - s, 1.0 - m);
}

float smaxPoly(float a, float b, float k) {
    float h = clamp(0.5 - 0.5 * (a + b) / max(k, 1e-5), 0.0, 1.0);
    return mix(a, -b, h) + k * h * (1.0 - h);
}

mat2 rot2D(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat2(c, -s, s, c);
}

// Scene SDF returning vec2(signedDistance, materialId)
vec2 mapScene(vec3 p) {
    float t = uRay.resolutionTime.z;

    // 1. Ground plane at y = -1.25
    float dGround = p.y + 1.25;
    vec2 res = vec2(dGround, 0.0);

    // 2. Central sculptural assembly: morphing round box + orbiting spheres + smooth torus union
    vec3 coreP = p - vec3(0.0, 0.15, 0.0);
    coreP.xz = rot2D(t * 0.45) * coreP.xz;

    float dBox = sdRoundBox(coreP, vec3(0.65, 0.65, 0.65), 0.12);
    float dCavity = sdSphere(coreP, 0.78 + 0.06 * sin(t * 1.7));
    float dCarvedCore = smaxPoly(dBox, dCavity, 0.14);

    vec3 torusP = p - vec3(0.0, 0.15, 0.0);
    torusP.xy = rot2D(t * 0.6) * torusP.xy;
    torusP.yz = rot2D(t * 0.35) * torusP.yz;
    float dTorus = sdTorus(torusP, vec2(1.15, 0.18));

    vec3 orbOffset = vec3(cos(t * 1.1) * 1.35, 0.35 * sin(t * 1.9), sin(t * 1.1) * 1.35);
    float dOrb = sdSphere(p - orbOffset, 0.34);

    vec2 coreUnion = sminCubic(dCarvedCore, dTorus, 0.28);
    vec2 sculpture = sminCubic(coreUnion.x, dOrb, 0.32);

    if (sculpture.x < res.x) {
        res = vec2(sculpture.x, 1.0 + 0.5 * sculpture.y);
    }

    // 3. Domain-repeated colonnade pillars outside the central clearing
    float cellSize = 4.5;
    vec2 cellId = round(p.xz / cellSize);
    bool isCenterClearing = (abs(cellId.x) < 1.5 && abs(cellId.y) < 1.5);
    if (!isCenterClearing) {
        vec3 repP = p;
        repP.xz = p.xz - cellSize * cellId;
        float dPillar = sdCylinder(repP - vec3(0.0, 0.35, 0.0), 0.28, 1.65);
        float dRing   = sdTorus(repP - vec3(0.0, 0.25 + 0.3 * sin(t + cellId.x + cellId.y), 0.0), vec2(0.42, 0.08));
        float dColumn = sminCubic(dPillar, dRing, 0.15).x;
        if (dColumn < res.x) {
            res = vec2(dColumn, 2.0);
        }
    }

    return res;
}

// 4-tap analytic tetrahedron gradient normal (avoids 6-tap finite difference overhead).
vec3 calcTetrahedronNormal(vec3 p) {
    const float eps = 0.0005;
    const vec2 k = vec2(1.0, -1.0);
    return normalize(
        k.xyy * mapScene(p + k.xyy * eps).x +
        k.yyx * mapScene(p + k.yyx * eps).x +
        k.yxy * mapScene(p + k.yxy * eps).x +
        k.xxx * mapScene(p + k.xxx * eps).x
    );
}

// Improved penumbra soft shadow raymarch.
float calcSoftShadow(vec3 ro, vec3 rd, float tMin, float tMax, float kPenumbra) {
    float shadow = 1.0;
    float prevDist = 1e20;
    float t = tMin;

    for (int i = 0; i < 48; ++i) {
        if (t >= tMax) {
            break;
        }
        float h = mapScene(ro + rd * t).x;
        if (h < 0.0004) {
            return 0.0;
        }
        float y = (h * h) / max(2.0 * prevDist, 1e-6);
        float d = sqrt(max(h * h - y * y, 0.0));
        shadow = min(shadow, kPenumbra * d / max(0.001, t - y));
        prevDist = h;
        t += clamp(h, 0.02, 0.45);
    }
    return clamp(shadow, 0.0, 1.0);
}

// Multi-tap SDF ambient occlusion along surface normal.
float calcAmbientOcclusion(vec3 pos, vec3 normal) {
    float occ = 0.0;
    float weight = 1.0;
    for (int i = 0; i < 5; ++i) {
        float h = 0.02 + 0.14 * float(i) / 4.0;
        float d = mapScene(pos + normal * h).x;
        occ += (h - d) * weight;
        weight *= 0.78;
    }
    return clamp(1.0 - 2.6 * occ, 0.0, 1.0);
}

vec3 acesFilmicToneMap(vec3 hdr) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((hdr * (a * hdr + b)) / (hdr * (c * hdr + d) + e), 0.0, 1.0);
}

void main() {
    vec2 resolution = max(uRay.resolutionTime.xy, vec2(1.0));
    float aspect = resolution.x / resolution.y;
    vec2 ndc = vUv * 2.0 - 1.0;

    float fovY = clamp(uRay.resolutionTime.w, 0.35, 2.2);
    float tanHalfFov = tan(0.5 * fovY);

    vec3 ro = uRay.cameraPosMaxDist.xyz;
    vec3 target = uRay.cameraTargetExposure.xyz;
    float maxDist = max(uRay.cameraPosMaxDist.w, 10.0);

    vec3 forward = normalize(target - ro);
    vec3 right   = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    vec3 up      = cross(right, forward);

    vec3 rd = normalize(forward + right * (ndc.x * aspect * tanHalfFov) + up * (ndc.y * tanHalfFov));

    // Sphere tracing loop
    float t = 0.0;
    float matId = -1.0;
    bool hit = false;

    for (int stepIdx = 0; stepIdx < MAX_MARCH_STEPS; ++stepIdx) {
        vec3 p = ro + rd * t;
        vec2 dMat = mapScene(p);
        float hitEps = max(MIN_HIT_DIST, 0.00045 * t);
        if (dMat.x < hitEps) {
            matId = dMat.y;
            hit = true;
            break;
        }
        t += dMat.x;
        if (t > maxDist) {
            break;
        }
    }

    vec3 skyColor = mix(vec3(0.05, 0.08, 0.14), vec3(0.18, 0.30, 0.52), clamp(rd.y * 0.5 + 0.5, 0.0, 1.0));
    vec3 color = skyColor;

    if (hit) {
        vec3 hitPos = ro + rd * t;
        vec3 N = calcTetrahedronNormal(hitPos);
        vec3 V = -rd;
        vec3 L = normalize(uRay.keyLightDirIntensity.xyz);
        vec3 H = normalize(V + L);

        // Material albedo selection
        vec3 albedo = vec3(0.55, 0.58, 0.62);
        float specPower = 48.0;
        if (matId < 0.5) {
            // Subtle architectural checkerboard floor
            float checker = mod(floor(hitPos.x * 1.2) + floor(hitPos.z * 1.2), 2.0);
            albedo = mix(vec3(0.14, 0.16, 0.19), vec3(0.26, 0.29, 0.33), checker);
            specPower = 24.0;
        } else if (matId < 1.8) {
            // Metallic-ceramic sculptural blend
            albedo = mix(vec3(0.82, 0.38, 0.16), vec3(0.18, 0.64, 0.78), clamp(matId - 1.0, 0.0, 1.0));
            specPower = 96.0;
        } else {
            // Colonnade stone pillars
            albedo = vec3(0.42, 0.45, 0.50);
            specPower = 36.0;
        }

        float nDotL = max(dot(N, L), 0.0);
        float nDotV = max(dot(N, V), 0.0);
        float nDotH = max(dot(N, H), 0.0);

        float shadow = (nDotL > 0.0) ? calcSoftShadow(hitPos + N * 0.004, L, 0.02, 20.0, 12.0) : 0.0;
        float ao     = calcAmbientOcclusion(hitPos, N);

        // Key light diffuse + Blinn-Phong specular
        vec3 keyColor = vec3(1.0, 0.94, 0.86) * max(uRay.keyLightDirIntensity.w, 0.1);
        vec3 diffuse  = albedo * (nDotL * shadow) * keyColor;
        vec3 specular = vec3(0.45) * pow(nDotH, specPower) * (nDotL * shadow) * keyColor;

        // Sky hemisphere fill + bounce light
        vec3 skyFill = albedo * vec3(0.16, 0.24, 0.38) * (0.5 + 0.5 * N.y) * ao;
        vec3 bounce  = albedo * vec3(0.12, 0.09, 0.07) * (0.5 - 0.5 * N.y) * ao;

        // Fresnel rim highlight
        float fresnel = pow(clamp(1.0 - nDotV, 0.0, 1.0), 4.0);
        vec3 rim = vec3(0.35, 0.58, 0.88) * (fresnel * ao * 0.65);

        color = diffuse + specular + skyFill + bounce + rim;

        // Exponential distance fog
        float fogFactor = 1.0 - exp(-0.0014 * t * t);
        color = mix(color, skyColor, clamp(fogFactor, 0.0, 1.0));
    }

    vec3 hdr = color * max(uRay.cameraTargetExposure.w, 0.1);
    vec3 mapped = acesFilmicToneMap(hdr);
    vec3 srgb = pow(max(mapped, vec3(0.0)), vec3(1.0 / 2.2));
    oFragColor = vec4(srgb, 1.0);
}
