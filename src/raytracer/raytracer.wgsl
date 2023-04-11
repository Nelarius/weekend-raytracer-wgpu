const PI = 3.1415927f;
const FRAC_1_PI = 0.31830987f;
const FRAC_PI_2 = 1.5707964f;

const T_MIN = 0.001f;
const T_MAX = 1000f;

@group(0) @binding(0)
var<uniform> vertex_uniforms: VertexUniforms;

@vertex
fn vsMain(model: VertexInput) -> VertexOutput {
    return VertexOutput(
        vertex_uniforms.viewProjectionMat * vertex_uniforms.modelMat * vec4<f32>(model.position, 0.0, 1.0),
        model.texCoords
    );
}

struct VertexUniforms {
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
@group(1) @binding(1) var<storage, read_write> rngStateBuffer: array<u32>;

@group(2) @binding(0) var<uniform> camera: Camera;
@group(2) @binding(1) var<storage, read> spheres: array<Sphere>;

@fragment
fn fsMain(in: VertexOutput) -> @location(0) vec4<f32> {
    let u = in.texCoords.x;
    let v = in.texCoords.y;

    let x = u32(u * f32(imageDimensions.x));
    let y = u32(v * f32(imageDimensions.y));

    let idx = imageDimensions.x * y + x;
    var rngState = rngStateBuffer[idx];

    let primaryRay = cameraMakeRay(camera, &rngState, u, 1f - v);
    let rgb = rayColor(primaryRay, &rngState);

    rngStateBuffer[idx] = rngState;

    return vec4(rgb, 1f);
}

fn rayColor(primaryRay: Ray, rngState: ptr<function, u32>) -> vec3<f32> {
    var ray = primaryRay;
    var color = vec3<f32>(0f, 0f, 0f);
    var closestIntersect = Intersection();

    var tClosest = T_MAX;
    var testIntersect = Intersection();
    var hit = false;
    for (var idx = 0u; idx < arrayLength(&spheres); idx = idx + 1u) {
        let sphere = spheres[idx];
        if rayIntersectSphere(ray, sphere, T_MIN, T_MAX, &testIntersect) {
            if testIntersect.t < tClosest {
                tClosest = testIntersect.t;
                closestIntersect = testIntersect;
                hit = true;
            }
        }
    }

    if hit {
        color = 0.5f * (vec3<f32>(1f, 1f, 1f) + closestIntersect.n);
    }

    return color;
}

struct Sphere {
    center: vec3<f32>,
    radius: f32,
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>
}

struct Intersection {
    p: vec3<f32>,
    n: vec3<f32>,
    t: f32,
}

fn rayIntersectSphere(ray: Ray, sphere: Sphere, tmin: f32, tmax: f32, hit: ptr<function, Intersection>) -> bool {
    let oc = ray.origin - sphere.center;
    let a = dot(ray.direction, ray.direction);
    let b = dot(oc, ray.direction);
    let c = dot(oc, oc) - sphere.radius * sphere.radius;
    let discriminant = b * b - a * c;

    if discriminant > 0f {
        var t = (-b - sqrt(b * b - a * c)) / a;
        if t < tmax && t > tmin {
            *hit = sphereIntersection(ray, sphere, t);
            return true;
        }

        t = (-b + sqrt(b * b - a * c)) / a;
        if t < tmax && t > tmin {
            *hit = sphereIntersection(ray, sphere, t);
            return true;
        }
    }

    return false;
}

fn sphereIntersection(ray: Ray, sphere: Sphere, t: f32) -> Intersection {
    let p = rayPointAtParameter(ray, t);
    let n = (1f / sphere.radius) * (p - sphere.center);

    return Intersection(p, n, t);
}

fn rayPointAtParameter(ray: Ray, t: f32) -> vec3<f32> {
    return ray.origin + t * ray.direction;
}

struct Camera {
    eye: vec3<f32>,
    horizontal: vec3<f32>,
    vertical: vec3<f32>,
    u: vec3<f32>,
    v: vec3<f32>,
    lensRadius: f32,
    lowerLeftCorner: vec3<f32>,
}

fn cameraMakeRay(camera: Camera, rngState: ptr<function, u32>, u: f32, v: f32) -> Ray {
    let randomPointInLens = camera.lensRadius * rngNextVec3InUnitDisk(rngState);
    let lensOffset = randomPointInLens.x * camera.u + randomPointInLens.y * camera.v;

    let origin = camera.eye + lensOffset;
    let direction = camera.lowerLeftCorner + u * camera.horizontal + v * camera.vertical - origin;

    return Ray(origin, direction);
}

fn rngNextVec3InUnitDisk(state: ptr<function, u32>) -> vec3<f32> {
    // Generate numbers uniformly in a disk:
    // https://stats.stackexchange.com/a/481559

    // r^2 is distributed as U(0, 1).
    let r = sqrt(rngNextFloat(state));
    let alpha = 2f * PI * rngNextFloat(state);

    let x = r * cos(alpha);
    let y = r * sin(alpha);

    return vec3(x, y, 0f);
}

fn rngNextVec3InUnitSphere(state: ptr<function, u32>) -> vec3<f32> {
    // probability density is uniformly distributed over r^3
    let r = pow(rngNextFloat(state), 0.33333f);
    let theta = PI * rngNextFloat(state);
    let phi = 2f * PI * rngNextFloat(state);

    let x = r * sin(theta) * cos(phi);
    let y = r * sin(theta) * sin(phi);
    let z = r * cos(theta);

    return vec3(x, y, z);
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
