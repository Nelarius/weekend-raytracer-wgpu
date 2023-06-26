# README

A WebGPU implementation of Peter Shirley's ray tracing books. An interactive renderer which allows you to play around with the camera and renderer settings in real-time.

## Run

```sh
$ cargo run --release
```

### Camera and render controls

* Translate the camera using WASD keys.
* Pan the camera by dragging the right mouse button.
* Adjust camera and rendering parameters using UI

## Read

More details about the WebGPU implementation is available in the following series of blog posts.

**[Weekend raytracing with wgpu, part 2](https://nelari.us/post/weekend_raytracing_with_wgpu_2/)**

![blog post part 2](https://nelari.us/img/weekend_raytracing_with_wgpu_2/featured_image.png)
* Adds texture support from "Ray Tracing: The Next Week" and a physically-based sky based on the Hosek-Wilkie model.
* Describes the global texture lookup system implemented in WGSL.

**[Weekend raytracing with wgpu, part 1](https://nelari.us/post/weekend_raytracing_with_wgpu_1/)**

![blog post part 1](https://nelari.us/img/weekend_raytracing_with_wgpu_1/featured_image.png)
* A straightforward fragment shader implementation of the "Ray Tracing In One Weekend" book.
* Notes on porting RNG, recursive rendering function and more to WGSL.

## Asset credits

assets/moon.jpeg
* NASA's Scientific Visualization Studio
* https://svs.gsfc.nasa.gov/4720

assets/earthmap.jpeg
* https://raytracing.github.io/images/earthmap.jpg

assets/sun.jpeg
* https://www.solarsystemscope.com/textures/
