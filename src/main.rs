#![deny(
    clippy::pedantic,
    rust_2018_idioms,
    future_incompatible,
    nonstandard_style
)]

pub extern crate nalgebra_glm as glm;

use fly_camera::FlyCameraController;
use raytracer::{
    Material, Raytracer, RenderParams, SamplingParams, Scene, SkyParams, Sphere, Texture,
};
use std::{collections::VecDeque, time::Instant};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

mod fly_camera;
mod raytracer;

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("weekend-raytracer-wgpu")
        .with_inner_size(winit::dpi::PhysicalSize::new(800, 600))
        .build(&event_loop)
        .unwrap();
    let mut context = pollster::block_on(GpuContext::new(&window));

    let viewport_size = {
        let viewport = window.inner_size();
        (viewport.width, viewport.height)
    };
    let max_viewport_resolution = window
        .available_monitors()
        .map(|monitor| -> u32 {
            let size = monitor.size();
            size.width * size.height
        })
        .max()
        .expect("There should be at least one monitor available");

    let scene = scene();
    let mut fly_camera_controller = FlyCameraController::default();

    let mut render_params = RenderParams {
        camera: fly_camera_controller.renderer_camera(),
        sky: SkyParams::default(),
        sampling: SamplingParams::default(),
        viewport_size,
    };
    let mut raytracer = Raytracer::new(
        &context.device,
        &context.surface_config,
        &scene,
        &render_params,
        max_viewport_resolution,
    )
    .expect("The default values should be selected correctly");

    let mut imgui = imgui::Context::create();
    let mut imgui_platform = imgui_winit_support::WinitPlatform::init(&mut imgui);
    imgui_platform.attach_window(
        imgui.io_mut(),
        &window,
        imgui_winit_support::HiDpiMode::Rounded,
    );
    imgui.set_ini_filename(Some(std::path::PathBuf::from("imgui.ini")));
    let hidpi_factor = window.scale_factor() as f32;
    let font_size = 13.0 * hidpi_factor;
    imgui.io_mut().font_global_scale = 1.0 / hidpi_factor;
    imgui
        .fonts()
        .add_font(&[imgui::FontSource::DefaultFontData {
            config: Some(imgui::FontConfig {
                size_pixels: font_size,
                ..Default::default()
            }),
        }]);
    let imgui_renderer_config = imgui_wgpu::RendererConfig {
        texture_format: context.surface_config.format,
        ..Default::default()
    };
    let mut imgui_renderer = imgui_wgpu::Renderer::new(
        &mut imgui,
        &context.device,
        &context.queue,
        imgui_renderer_config,
    );
    let mut last_cursor = None;

    let mut last_time = Instant::now();
    let mut fps_counter = FpsCounter::new();

    event_loop.run(move |event, _, _control_flow| {
        imgui_platform.handle_event(imgui.io_mut(), &window, &event);

        match event {
            Event::WindowEvent { event, .. } => {
                fly_camera_controller.handle_event(&event);
                match event {
                    WindowEvent::CloseRequested => {
                        *_control_flow = ControlFlow::Exit;
                    }

                    WindowEvent::Resized(physical_size) => {
                        if physical_size.width > 0 && physical_size.height > 0 {
                            render_params.viewport_size =
                                (physical_size.width, physical_size.height);
                            context.surface_config.width = physical_size.width;
                            context.surface_config.height = physical_size.height;
                            context
                                .surface
                                .configure(&context.device, &context.surface_config);
                        }
                    }

                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        if new_inner_size.width > 0 && new_inner_size.height > 0 {
                            render_params.viewport_size =
                                (new_inner_size.width, new_inner_size.height);
                            context.surface_config.width = new_inner_size.width;
                            context.surface_config.height = new_inner_size.height;
                            context
                                .surface
                                .configure(&context.device, &context.surface_config);
                        }
                    }

                    _ => {}
                }
            }

            Event::MainEventsCleared => {
                {
                    let dt = last_time.elapsed().as_secs_f32();
                    let now = Instant::now();

                    fps_counter.update(dt);
                    fly_camera_controller.after_events(render_params.viewport_size, 2.0 * dt);

                    imgui.io_mut().update_delta_time(now - last_time);

                    last_time = now;
                }

                {
                    imgui_platform
                        .prepare_frame(imgui.io_mut(), &window)
                        .expect("WinitPlatform::prepare_frame failed");
                    let ui = imgui.frame();
                    {
                        let window = ui.window("Parameters");
                        window
                            .size([300.0, 300.0], imgui::Condition::FirstUseEver)
                            .build(|| {
                                ui.text(format!(
                                    "FPS: {:.1}, render progress: {:.1} %",
                                    fps_counter.average_fps(),
                                    raytracer.progress() * 100.0
                                ));
                                ui.separator();

                                ui.text("Sampling parameters");

                                ui.text("num_samples_per_pixel");
                                ui.same_line();
                                ui.radio_button(
                                    "1",
                                    &mut render_params.sampling.num_samples_per_pixel,
                                    1_u32,
                                );
                                ui.same_line();
                                ui.radio_button(
                                    "2",
                                    &mut render_params.sampling.num_samples_per_pixel,
                                    2_u32,
                                );
                                ui.same_line();
                                ui.radio_button(
                                    "4",
                                    &mut render_params.sampling.num_samples_per_pixel,
                                    4_u32,
                                );

                                ui.text("max_samples_per_pixel");
                                ui.same_line();
                                ui.radio_button(
                                    "128",
                                    &mut render_params.sampling.max_samples_per_pixel,
                                    128_u32,
                                );
                                ui.same_line();
                                ui.radio_button(
                                    "256",
                                    &mut render_params.sampling.max_samples_per_pixel,
                                    256_u32,
                                );
                                ui.same_line();
                                ui.radio_button(
                                    "512",
                                    &mut render_params.sampling.max_samples_per_pixel,
                                    512_u32,
                                );

                                ui.slider(
                                    "num_bounces",
                                    4,
                                    10,
                                    &mut render_params.sampling.num_bounces,
                                );

                                ui.separator();

                                ui.text("Camera parameters");
                                ui.slider(
                                    "vfov",
                                    10.0,
                                    90.0,
                                    &mut fly_camera_controller.vfov_degrees,
                                );
                                ui.slider(
                                    "aperture",
                                    0.0,
                                    1.0,
                                    &mut fly_camera_controller.aperture,
                                );
                                ui.slider(
                                    "focus_distance",
                                    5.0,
                                    20.0,
                                    &mut fly_camera_controller.focus_distance,
                                );

                                ui.separator();

                                ui.text("Sky parameters");
                                ui.slider(
                                    "azimuth",
                                    0_f32,
                                    360_f32,
                                    &mut render_params.sky.azimuth_degrees,
                                );
                                ui.slider(
                                    "inclination",
                                    0_f32,
                                    90_f32,
                                    &mut render_params.sky.zenith_degrees,
                                );
                                ui.slider(
                                    "turbidity",
                                    1_f32,
                                    10_f32,
                                    &mut render_params.sky.turbidity,
                                );
                                ui.color_edit3("albedo", &mut render_params.sky.albedo);
                            });
                    }

                    if last_cursor != Some(ui.mouse_cursor()) {
                        last_cursor = Some(ui.mouse_cursor());
                        imgui_platform.prepare_render(&ui, &window);
                    }
                }

                render_params.camera = fly_camera_controller.renderer_camera();
                match raytracer.set_render_params(&context.queue, &render_params) {
                    Err(e) => {
                        eprintln!("Error setting render params: {e}")
                    }
                    _ => {}
                }

                window.request_redraw();
            }

            Event::RedrawRequested(window_id) if window_id == window.id() => {
                let frame = match context.surface.get_current_texture() {
                    Ok(frame) => frame,
                    Err(e) => {
                        eprintln!("Surface error: {:?}", e);
                        return;
                    }
                };

                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                let mut encoder = context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.012,
                                    g: 0.012,
                                    b: 0.012,
                                    a: 1.0,
                                }),
                                store: true,
                            },
                        })],
                        depth_stencil_attachment: None,
                        label: None,
                    });

                    raytracer.render_frame(&context.queue, &mut render_pass);

                    match imgui_renderer.render(
                        imgui.render(),
                        &context.queue,
                        &context.device,
                        &mut render_pass,
                    ) {
                        Err(e) => eprintln!("Imgui render error: {:?}", e),
                        _ => {}
                    }
                }

                context.queue.submit(Some(encoder.finish()));
                frame.present();
            }

            _ => {}
        }
    });
}

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface,
    surface_config: wgpu::SurfaceConfiguration,
}

impl GpuContext {
    async fn new(window: &Window) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = unsafe {
            instance
                .create_surface(window)
                .expect("Surface creation should succeed on desktop")
        };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .expect("Adapter should be available on desktop");
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits {
                        max_storage_buffer_binding_size: 512_u32 << 20,
                        ..Default::default()
                    },
                    label: None,
                },
                None,
            )
            .await
            .expect("Device should be available on desktop");

        let window_size = window.inner_size();
        // NOTE
        // You can check supported configs, such as formats, by calling Surface::capabilities
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: window_size.width,
            height: window_size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![wgpu::TextureFormat::Bgra8Unorm],
        };
        surface.configure(&device, &surface_config);

        Self {
            device,
            queue,
            surface,
            surface_config,
        }
    }
}

struct FpsCounter {
    frame_times: VecDeque<f32>,
}

impl FpsCounter {
    const MAX_FRAME_TIMES: usize = 8;

    pub fn new() -> Self {
        Self {
            frame_times: VecDeque::with_capacity(Self::MAX_FRAME_TIMES),
        }
    }

    pub fn update(&mut self, dt: f32) {
        self.frame_times.push_back(dt);
        if self.frame_times.len() > Self::MAX_FRAME_TIMES {
            self.frame_times.pop_front();
        }
    }

    pub fn average_fps(&self) -> f32 {
        let sum: f32 = self.frame_times.iter().sum();
        self.frame_times.len() as f32 / sum
    }
}

fn scene() -> Scene {
    let materials = vec![
        Material::Checkerboard {
            even: Texture::new_from_color(glm::vec3(0.5_f32, 0.7_f32, 0.8_f32)),
            odd: Texture::new_from_color(glm::vec3(0.9_f32, 0.9_f32, 0.9_f32)),
        },
        Material::Lambertian {
            albedo: Texture::new_from_image("assets/moon.jpeg")
                .expect("Hardcoded path should be valid"),
        },
        Material::Metal {
            albedo: Texture::new_from_color(glm::vec3(1_f32, 0.85_f32, 0.57_f32)),
            fuzz: 0.4_f32,
        },
        Material::Dielectric {
            refraction_index: 1.5_f32,
        },
        Material::Lambertian {
            albedo: Texture::new_from_image("assets/earthmap.jpeg")
                .expect("Hardcoded path should be valid"),
        },
        Material::LambertianEmissive {
            albedo: Texture::new_from_color(glm::vec3(0.0, 0.0, 0.0)),
            emit: Texture::new_from_scaled_image("assets/sun.jpeg", 50.0)
                .expect("Hardcoded path should be valid"),
        },
        Material::Lambertian {
            albedo: Texture::new_from_color(glm::vec3(0.3_f32, 0.9_f32, 0.9_f32)),
        },
        Material::LambertianEmissive {
            albedo: Texture::new_from_color(glm::vec3(1.0, 0.3, 0.3)),
            emit: Texture::new_from_color(glm::vec3(50.0_f32, 0.0_f32, 0.0_f32)),
        },
        Material::LambertianEmissive {
            albedo: Texture::new_from_color(glm::vec3(0.3, 1.0, 0.3)),
            emit: Texture::new_from_color(glm::vec3(0.0_f32, 50.0_f32, 0.0_f32)),
        },
        Material::LambertianEmissive {
            albedo: Texture::new_from_color(glm::vec3(0.3, 0.3, 1.0)),
            emit: Texture::new_from_color(glm::vec3(0.0, 0.0, 50.0)),
        },
    ];

    let spheres = vec![
        Sphere::new(glm::vec3(0.0, -500.0, -1.0), 500.0, 0_u32),
        // left row
        Sphere::new(glm::vec3(-5.0, 1.0, -4.0), 1.0, 7_u32),
        Sphere::new(glm::vec3(0.0, 1.0, -4.0), 1.0, 8_u32),
        Sphere::new(glm::vec3(5.0, 1.0, -4.0), 1.0, 9_u32),
        // middle row
        Sphere::new(glm::vec3(-5.0, 1.0, 0.0), 1.0, 2_u32),
        Sphere::new(glm::vec3(0.0, 1.0, 0.0), 1.0, 3_u32),
        Sphere::new(glm::vec3(5.0, 1.0, 0.0), 1.0, 6_u32),
        // right row
        Sphere::new(glm::vec3(-5.0, 0.8, 4.0), 0.8, 1_u32),
        Sphere::new(glm::vec3(0.0, 1.2, 4.0), 1.2, 4_u32),
        Sphere::new(glm::vec3(5.0, 2.0, 4.0), 2.0, 5_u32),
    ];

    Scene { spheres, materials }
}
