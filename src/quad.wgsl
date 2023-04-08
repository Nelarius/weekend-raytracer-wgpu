@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vsMain(model: VertexInput) -> VertexOutput {
    return VertexOutput(
        uniforms.viewProjectionMat * uniforms.modelMat * vec4<f32>(model.position, 0.0, 1.0),
        model.texCoords
    );
}

struct Uniforms {
    viewProjectionMat: mat4x4<f32>,
    modelMat: mat4x4<f32>,
}

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) texCoords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clipPosition: vec4<f32>,
    @location(0) texCoords: vec2<f32>,
}

@fragment
fn fsMain(in: VertexOutput) -> @location(0) vec4<f32> {
    let c = vec3(255f, 0f, 255f);
    return vec4(c, 1f);
}
