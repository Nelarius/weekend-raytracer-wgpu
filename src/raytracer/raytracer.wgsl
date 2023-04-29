const pi = 3.1415927f;
const frac1Pi = 0.31830987f;
const fracPi2 = 1.5707964f;

const minT = 0.001f;
const maxT = 1000f;

@group(0) @binding(0) var<uniform> vertexUniforms: VertexUniforms;

@vertex
fn vsMain(model: VertexInput) -> VertexOutput {
    return VertexOutput(
        vertexUniforms.viewProjectionMat * vertexUniforms.modelMat * vec4<f32>(model.position, 0.0, 1.0),
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
@group(1) @binding(1) var<storage, read_write> imageBuffer: array<array<f32, 3>>;
@group(1) @binding(2) var<storage, read_write> rngStateBuffer: array<u32>;

@group(2) @binding(0) var<uniform> camera: Camera;
@group(2) @binding(1) var<uniform> samplingParams: SamplingParams;

@group(3) @binding(0) var<storage, read> spheres: array<Sphere>;
@group(3) @binding(1) var<storage, read> materials: array<Material>;

@fragment
fn fsMain(in: VertexOutput) -> @location(0) vec4<f32> {
    let u = in.texCoords.x;
    let v = in.texCoords.y;

    let x = u32(u * f32(imageDimensions.x));
    let y = u32(v * f32(imageDimensions.y));

    let idx = imageDimensions.x * y + x;

    var rngState = rngStateBuffer[idx];
    var pixel = imageBuffer[idx];    
    {
        if samplingParams.clearAccumulatedSamples == 1u {
            pixel = array<f32, 3>(0f, 0f, 0f);
        }

        let rgb = samplePixel(x, y, &rngState);

        pixel[0u] += rgb[0u];
        pixel[1u] += rgb[1u];
        pixel[2u] += rgb[2u];
    }
    imageBuffer[idx] = pixel;
    rngStateBuffer[idx] = rngState;

    let invN = 1f / f32(samplingParams.accumulatedSamplesPerPixel);
    return vec4(
        invN * pixel[0u],
        invN * pixel[1u],
        invN * pixel[2u],
        1f
    );
}

fn samplePixel(x: u32, y: u32, rngState: ptr<function, u32>) -> vec3<f32> {
    let invWidth = 1f / f32(imageDimensions.x);
    let invHeight = 1f / f32(imageDimensions.y);

    let numSamples = samplingParams.numSamplesPerPixel;
    var color = vec3(0f, 0f, 0f);
    for (var i = 0u; i < numSamples; i += 1u) {
        let u = (f32(x) + rngNextFloat(rngState)) * invWidth;
        let v = (f32(y) + rngNextFloat(rngState)) * invHeight;

        let primaryRay = cameraMakeRay(camera, rngState, u, 1f - v);
        color += rayColor(primaryRay, rngState);
    }

    return color;
}

fn rayColor(primaryRay: Ray, rngState: ptr<function, u32>) -> vec3<f32> {
    var ray = primaryRay;

    var color = vec3(0f, 0f, 0f);
    var throughput = vec3(1f, 1f, 1f);

    for (var bounce = 0u; bounce < samplingParams.numBounces; bounce += 1u) {
        var intersection = Intersection();
        var materialIdx = 0u;

        // Intersection test
        var closestT = maxT;

        for (var idx = 0u; idx < arrayLength(&spheres); idx = idx + 1u) {
            let sphere = spheres[idx];
            var testIntersect = Intersection();
            if rayIntersectSphere(ray, sphere, minT, closestT, &testIntersect) {
                closestT = testIntersect.t;
                intersection = testIntersect;
                materialIdx = sphere.materialIdx;
            }
        }

        // The ray missed. Output background color.
        if closestT == maxT {
            let unitDirection = normalize(ray.direction);
            let t = 0.5f * (unitDirection.y + 1f);
            color = (1f - t) * vec3(1f, 1f, 1f) + t * vec3(0.5f, 0.7f, 1f);
            break;
        }

        // Scatter the ray from the surface
        let material = materials[materialIdx];
        var scatter = scatterRay(ray, intersection, material, rngState);
        ray = scatter.ray;
        throughput *= scatter.albedo;
    }

    return throughput * color;
}

fn scatterRay(rayIn: Ray, hit: Intersection, material: Material, rngState: ptr<function, u32>) -> Scatter {
    switch material.id {
        case 0u: {
            return scatterLambertian(hit, material, rngState);
        }

        case 1u: {
            return scatterMetal(rayIn, hit, material, rngState);
        }

        case 2u: {
            return scatterDielectric(rayIn, hit, material, rngState);
        }

        default: {
            // An aggressive pink color to indicate an error
            return scatterLambertian(hit, Material(vec4(0.9921f, 0.24705f, 0.57254f, 1f), 0f, 0u), rngState);
        }
    }
}

fn scatterLambertian(hit: Intersection, material: Material, rngState: ptr<function, u32>) -> Scatter {
    let scatterDirection = hit.n + rngNextVec3InUnitSphere(rngState);
    let albedo = material.albedo_and_pad.rgb;
    return Scatter(Ray(hit.p, scatterDirection), albedo);
}

fn scatterMetal(rayIn: Ray, hit: Intersection, material: Material, rngState: ptr<function, u32>) -> Scatter {
    let fuzz = material.x;
    let scatterDirection = reflect(rayIn.direction, hit.n) + material.x * rngNextVec3InUnitSphere(rngState);
    let albedo = material.albedo_and_pad.rgb;
    return Scatter(Ray(hit.p, scatterDirection), albedo);
}

fn scatterDielectric(rayIn: Ray, hit: Intersection, material: Material, rngState: ptr<function, u32>) -> Scatter {
    let refractionIndex = material.x;

    var outwardNormal = vec3(0f, 0f, 0f);
    var niOverNt = 0f;
    var cosine = 0f;
    if dot(rayIn.direction, hit.n) > 0f {
        outwardNormal = -hit.n;
        niOverNt = refractionIndex;
        cosine = refractionIndex * dot(normalize(rayIn.direction), hit.n);
    } else {
        outwardNormal = hit.n;
        niOverNt = 1f / refractionIndex;
        cosine = dot(normalize(-rayIn.direction), hit.n);
    };

    var refractedDirection = vec3(0f, 0f, 0f);
    if refract(rayIn.direction, outwardNormal, niOverNt, &refractedDirection) {
        let reflectionProb = schlick(cosine, refractionIndex);
        var scatteredRay = refractedDirection;
        if rngNextFloat(rngState) < reflectionProb {
            reflect(rayIn.direction, hit.n);
        }

        return Scatter(Ray(hit.p, scatteredRay), vec3(1f, 1f, 1f));
    }

    let scatteredRay = reflect(rayIn.direction, hit.n);
    return Scatter(Ray(hit.p, scatteredRay), vec3(1f, 1f, 1f));
}

fn refract(v: vec3<f32>, n: vec3<f32>, niOverNt: f32, refractDirection: ptr<function, vec3<f32>>) -> bool {
    // ni * sin(i) = nt * sin(t)
    // sin(t) = sin(i) * (ni / nt)
    let uv = normalize(v);
    let dt = dot(uv, n);
    let discriminant = 1f - niOverNt * niOverNt * (1f - dt * dt);
    if discriminant > 0f {
        *refractDirection = niOverNt * (uv - dt * n) - sqrt(discriminant) * n;
        return true;
    }

    return false;
}

fn schlick(cosine: f32, refractionIndex: f32) -> f32 {
    var r0 = (1f - refractionIndex) / (1f + refractionIndex);
    r0 = r0 * r0;
    return r0 + pow((1f - r0) * (1f - cosine), 5f);
}

struct SamplingParams {
    numSamplesPerPixel: u32,
    numBounces: u32,
    accumulatedSamplesPerPixel: u32,
    clearAccumulatedSamples: u32,
}

struct Sphere {
    center_and_pad: vec4<f32>,
    radius: f32,
    materialIdx: u32,
}

struct Material {
    albedo_and_pad: vec4<f32>,
    x: f32,
    id: u32,
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>
}

struct Scatter {
    ray: Ray,
    albedo: vec3<f32>,
}

struct Intersection {
    p: vec3<f32>,
    n: vec3<f32>,
    t: f32,
}

fn rayIntersectSphere(ray: Ray, sphere: Sphere, tmin: f32, tmax: f32, hit: ptr<function, Intersection>) -> bool {
    let oc = ray.origin - sphere.center_and_pad.xyz;
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
    let n = (1f / sphere.radius) * (p - sphere.center_and_pad.xyz);

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
    let alpha = 2f * pi * rngNextFloat(state);

    let x = r * cos(alpha);
    let y = r * sin(alpha);

    return vec3(x, y, 0f);
}

fn rngNextVec3InUnitSphere(state: ptr<function, u32>) -> vec3<f32> {
    // probability density is uniformly distributed over r^3
    let r = pow(rngNextFloat(state), 0.33333f);
    let theta = pi * rngNextFloat(state);
    let phi = 2f * pi * rngNextFloat(state);

    let x = r * sin(theta) * cos(phi);
    let y = r * sin(theta) * sin(phi);
    let z = r * cos(theta);

    return vec3(x, y, z);
}

fn rngNextFloat(state: ptr<function, u32>) -> f32 {
    rngNextInt(state);
    return f32(*state) / f32(0xffffffffu);
}

fn rngNextInt(state: ptr<function, u32>) {
    // PCG random number generator
    // Based on https://www.shadertoy.com/view/XlGcRh

    let oldState = *state + 747796405u + 2891336453u;
    let word = ((oldState >> ((oldState >> 28u) + 4u)) ^ oldState) * 277803737u;
    *state = (word >> 22u) ^ word;
}
