#version 460 core

// Fullscreen triangle generated purely from gl_VertexID (0, 1, 2) — no VBO required.
layout(location = 0) out vec2 vUv;

void main() {
    vec2 uv = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    vUv = uv;
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
