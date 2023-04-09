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

@group(1) @binding(0) var<uniform> imageDimensions: vec2<u32>;
@group(1) @binding(1) var<storage, read_write> seedBuffer: array<u32>;

@fragment
fn fsMain(in: VertexOutput) -> @location(0) vec4<f32> {
    let u = in.texCoords.x;
    let v = in.texCoords.y;

    let x = u32(u * f32(imageDimensions.x));
    let y = u32(v * f32(imageDimensions.y));

    let idx = imageDimensions.x * y + x;
    var state = seedBuffer[idx];

    let rgb = vec3<f32>(
        rngNextFloat(&state),
        rngNextFloat(&state),
        rngNextFloat(&state)
    );

    seedBuffer[idx] = state;

    return vec4(rgb, 1f);
}

fn rngNextInt(state: ptr<function, u32>) {
    // PCG random number generator
    // Based on https://www.shadertoy.com/view/XlGcRh

    let oldState = *state + 747796405u + 2891336453u;
    let word = ((oldState >> ((oldState >> 28u) + 4u)) ^ oldState) * 277803737u;
    *state = (word >> 22u) ^ word;
}

fn rngNextFloat(state: ptr<function, u32>) -> f32 {
    rngNextInt(state);
    return f32(*state) / f32(0xffffffffu);
}
