#![deny(
    clippy::pedantic,
    rust_2018_idioms,
    future_incompatible,
    nonstandard_style
)]

pub extern crate nalgebra_glm as glm;

use raytracer::{Angle, Camera, Raytracer, RenderParams, Scene, Sphere};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

mod raytracer;

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

fn scene_and_camera() -> (Scene, Camera) {
    let spheres = vec![
        Sphere {
            center: glm::vec3(0.0, -500.0, -1.0),
            radius: 500.0,
        },
        Sphere {
            center: glm::vec3(0.0, 1.0, 0.0),
            radius: 1.0,
        },
        Sphere {
            center: glm::vec3(-5.0, 1.0, 0.0),
            radius: 1.0,
        },
        Sphere {
            center: glm::vec3(5.0, 1.0, 0.0),
            radius: 1.0,
        },
    ];

    let camera = {
        let look_from = glm::vec3(-10.0, 2.0, -4.0);
        let look_at = glm::vec3(0.0, 1.0, 0.0);
        let up = glm::vec3(0.0, 1.0, 0.0);
        let vfov = Angle::degrees(30_f32);
        let aperture = 1.0_f32;
        let focus_distance = glm::magnitude(&(look_at - look_from));

        Camera {
            eye_pos: look_from,
            eye_dir: look_at - look_from,
            up,
            vfov,
            aperture,
            focus_distance,
        }
    };

    (Scene { spheres }, camera)
}

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

    let (scene, camera) = scene_and_camera();
    let mut render_params = RenderParams {
        camera,
        viewport_size,
    };

    let mut raytracer = Raytracer::new(
        &context.device,
        &context.surface_config,
        scene,
        render_params,
        max_viewport_resolution,
    );

    event_loop.run(move |event, _, _control_flow| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => {
                *_control_flow = ControlFlow::Exit;
            }

            WindowEvent::Resized(physical_size) => {
                render_params.viewport_size = (physical_size.width, physical_size.height);
                context.surface_config.width = physical_size.width;
                context.surface_config.height = physical_size.height;
                context
                    .surface
                    .configure(&context.device, &context.surface_config);
            }

            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                render_params.viewport_size = (new_inner_size.width, new_inner_size.height);
                context.surface_config.width = new_inner_size.width;
                context.surface_config.height = new_inner_size.height;
                context
                    .surface
                    .configure(&context.device, &context.surface_config);
            }

            _ => {}
        },

        Event::MainEventsCleared => {
            raytracer.set_render_params(&context.queue, render_params);

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

                raytracer.render_frame(&mut render_pass);
            }

            context.queue.submit(Some(encoder.finish()));
            frame.present();
        }

        _ => {}
    });
}
