#![deny(
    clippy::pedantic,
    rust_2018_idioms,
    future_incompatible,
    nonstandard_style
)]

pub extern crate nalgebra_glm as glm;

use fly_camera::{FlyCamera, FlyCameraControlState};
use raytracer::{Raytracer, RenderParams, Scene, Sphere};
use std::time::Instant;
use winit::{
    event::{ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

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
    let mut fly_camera_control_state = FlyCameraControlState::default();
    let mut fly_camera = FlyCamera::default();

    let mut render_params = RenderParams {
        camera: fly_camera.renderer_camera(),
        viewport_size,
    };
    let mut raytracer = Raytracer::new(
        &context.device,
        &context.surface_config,
        scene,
        render_params,
        max_viewport_resolution,
    );

    let mut last_time = Instant::now();

    event_loop.run(move |event, _, _control_flow| match event {
        Event::NewEvents(_) => {
            fly_camera_control_state.begin_frame();
        }

        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => {
                *_control_flow = ControlFlow::Exit;
            }

            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                let is_pressed = state == ElementState::Pressed;
                match keycode {
                    VirtualKeyCode::W => {
                        fly_camera_control_state.forward_pressed = is_pressed;
                    }
                    VirtualKeyCode::S => {
                        fly_camera_control_state.backward_pressed = is_pressed;
                    }
                    VirtualKeyCode::A => {
                        fly_camera_control_state.left_pressed = is_pressed;
                    }
                    VirtualKeyCode::D => {
                        fly_camera_control_state.right_pressed = is_pressed;
                    }
                    VirtualKeyCode::Q => {
                        fly_camera_control_state.down_pressed = is_pressed;
                    }
                    VirtualKeyCode::E => {
                        fly_camera_control_state.up_pressed = is_pressed;
                    }
                    _ => {}
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let position = (position.x as f32, position.y as f32);
                if let Some(previous_position) = fly_camera_control_state.previous_mouse_pos {
                    fly_camera_control_state.mouse_delta = (
                        position.0 - previous_position.0,
                        previous_position.1 - position.1,
                    );
                }
                fly_camera_control_state.previous_mouse_pos = Some(position);
            }

            WindowEvent::MouseInput { button, state, .. } => match button {
                MouseButton::Right => {
                    fly_camera_control_state.look_pressed = state == ElementState::Pressed;
                }
                _ => (),
            },

            WindowEvent::Resized(physical_size) => {
                if physical_size.width > 0 && physical_size.height > 0 {
                    render_params.viewport_size = (physical_size.width, physical_size.height);
                    context.surface_config.width = physical_size.width;
                    context.surface_config.height = physical_size.height;
                    context
                        .surface
                        .configure(&context.device, &context.surface_config);
                }
            }

            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                if new_inner_size.width > 0 && new_inner_size.height > 0 {
                    render_params.viewport_size = (new_inner_size.width, new_inner_size.height);
                    context.surface_config.width = new_inner_size.width;
                    context.surface_config.height = new_inner_size.height;
                    context
                        .surface
                        .configure(&context.device, &context.surface_config);
                }
            }

            _ => {}
        },

        Event::MainEventsCleared => {
            {
                let dt = last_time.elapsed().as_secs_f32();
                last_time = Instant::now();

                fly_camera.translate(fly_camera_control_state.translation(2.0 * dt));
                fly_camera.rotate(fly_camera_control_state.rotation(0.5 * dt));
            }

            render_params.camera = fly_camera.renderer_camera();
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

mod fly_camera {
    use crate::raytracer::{Angle, Camera};

    pub struct FlyCamera {
        pub position: glm::Vec3,
        pub yaw: Angle,
        pub pitch: Angle,
        pub vfov_degrees: f32,
        pub aperture: f32,
        pub focus_distance: f32,
    }

    impl Default for FlyCamera {
        fn default() -> Self {
            let look_from = glm::vec3(-10.0, 2.0, -4.0);
            let look_at = glm::vec3(0.0, 1.0, 0.0);
            let focus_distance = glm::magnitude(&(look_at - look_from));

            Self {
                position: look_from,
                yaw: Angle::degrees(15_f32),
                pitch: Angle::degrees(-10_f32),
                vfov_degrees: 30.0,
                aperture: 1.0,
                focus_distance,
            }
        }
    }

    impl FlyCamera {
        pub fn translate(&mut self, delta: glm::Vec3) {
            let orientation = self.axes();
            self.position += orientation.right * delta.x
                + orientation.up * delta.y
                + orientation.forward * delta.z;
        }

        pub fn rotate(&mut self, deltas: EulerAngles) {
            self.yaw = self.yaw + deltas.yaw;
            self.pitch = self.pitch + deltas.pitch;
        }

        pub fn renderer_camera(&self) -> Camera {
            let orientation = self.axes();
            Camera {
                eye_pos: self.position,
                eye_dir: orientation.forward,
                up: orientation.up,
                vfov: Angle::degrees(self.vfov_degrees),
                aperture: self.aperture,
                focus_distance: self.focus_distance,
            }
        }

        fn axes(&self) -> Orientation {
            let forward = glm::normalize(&glm::vec3(
                self.yaw.as_radians().cos() * self.pitch.as_radians().cos(),
                self.pitch.as_radians().sin(),
                self.yaw.as_radians().sin() * self.pitch.as_radians().cos(),
            ));
            let up = glm::vec3(0.0, 1.0, 0.0);
            let right = glm::cross(&forward, &up);
            Orientation { forward, right, up }
        }
    }

    pub struct FlyCameraControlState {
        pub forward_pressed: bool,
        pub backward_pressed: bool,
        pub left_pressed: bool,
        pub right_pressed: bool,
        pub up_pressed: bool,
        pub down_pressed: bool,
        pub look_pressed: bool,
        pub previous_mouse_pos: Option<(f32, f32)>,
        pub mouse_delta: (f32, f32),
    }

    impl Default for FlyCameraControlState {
        fn default() -> Self {
            Self {
                forward_pressed: false,
                backward_pressed: false,
                left_pressed: false,
                right_pressed: false,
                up_pressed: false,
                down_pressed: false,
                look_pressed: false,
                previous_mouse_pos: None,
                mouse_delta: (0.0, 0.0),
            }
        }
    }

    impl FlyCameraControlState {
        pub fn translation(&self, s: f32) -> glm::Vec3 {
            let v = |b| if b { 1_f32 } else { 0_f32 };
            glm::vec3(
                s * (v(self.right_pressed) - v(self.left_pressed)),
                s * (v(self.up_pressed) - v(self.down_pressed)),
                s * (v(self.forward_pressed) - v(self.backward_pressed)),
            )
        }

        pub fn rotation(&self, s: f32) -> EulerAngles {
            if self.look_pressed {
                EulerAngles {
                    yaw: Angle::radians(-s * 0.5 * self.mouse_delta.0),
                    pitch: Angle::radians(-s * 0.5 * self.mouse_delta.1),
                }
            } else {
                EulerAngles {
                    yaw: Angle::radians(0.0),
                    pitch: Angle::radians(0.0),
                }
            }
        }

        pub fn begin_frame(&mut self) {
            self.mouse_delta = (0.0, 0.0);
        }
    }

    struct Orientation {
        forward: glm::Vec3,
        right: glm::Vec3,
        up: glm::Vec3,
    }

    pub struct EulerAngles {
        yaw: Angle,
        pitch: Angle,
    }
}

fn scene() -> Scene {
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

    Scene { spheres }
}
