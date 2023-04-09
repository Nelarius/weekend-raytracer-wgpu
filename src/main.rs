#![deny(
    clippy::pedantic,
    rust_2018_idioms,
    future_incompatible,
    nonstandard_style
)]

pub extern crate nalgebra_glm as glm;

use image_buffers::ImageBuffers;
use renderer::Renderer;
use winit::{
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

mod gpu_buffer;
mod image_buffers;

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

mod renderer {
    use super::gpu_buffer::UniformBuffer;
    use super::image_buffers::ImageBuffers;
    use wgpu::util::DeviceExt;

    pub struct Renderer {
        uniform_buffer: UniformBuffer,
        uniform_bind_group: wgpu::BindGroup,
        image_bind_group: wgpu::BindGroup,
        vertex_buffer: wgpu::Buffer,
        pipeline: wgpu::RenderPipeline,
    }

    impl Renderer {
        pub fn new(
            device: &wgpu::Device,
            surface_config: &wgpu::SurfaceConfiguration,
            image_buffers: &ImageBuffers,
            viewport_size: (u32, u32),
        ) -> Self {
            let uniforms = VertexUniforms {
                view_projection_matrix: view_projection_matrix(viewport_size),
                model_matrix: glm::identity(),
            };
            let uniform_buffer = UniformBuffer::new_from_bytes(
                device,
                bytemuck::bytes_of(&uniforms),
                Some("uniforms"),
            );
            let uniform_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[uniform_buffer.layout(0, wgpu::ShaderStages::VERTEX)],
                    label: Some("uniforms layout"),
                });
            let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &uniform_bind_group_layout,
                entries: &[uniform_buffer.binding(0)],
                label: Some("uniforms bind group"),
            });

            let image_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        image_buffers
                            .image_dimensions_buffer
                            .layout(0, wgpu::ShaderStages::FRAGMENT),
                        image_buffers.rng_seed_buffer.layout(
                            1,
                            wgpu::ShaderStages::FRAGMENT,
                            false,
                        ),
                    ],
                    label: Some("image layout"),
                });
            let image_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &image_bind_group_layout,
                entries: &[
                    image_buffers.image_dimensions_buffer.binding(0),
                    image_buffers.rng_seed_buffer.binding(1),
                ],
                label: Some("image bind group"),
            });

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                source: wgpu::ShaderSource::Wgsl(include_str!("quad.wgsl").into()),
                label: Some("quad.wgsl"),
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&uniform_bind_group_layout, &image_bind_group_layout],
                push_constant_ranges: &[],
                label: Some("quad.wgsl layout"),
            });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vsMain",
                    buffers: &[Vertex::desc()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fsMain",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_config.format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent::REPLACE,
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    cull_mode: Some(wgpu::Face::Back),
                    // Requires Features::DEPTH_CLAMPING
                    conservative: false,
                    unclipped_depth: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                label: Some("quad.wgsl pipeline"),
                // If the pipeline will be used with a multiview render pass, this
                // indicates how many array layers the attachments will have.
                multiview: None,
            });

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                contents: bytemuck::cast_slice(VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
                label: Some("VertexInput buffer"),
            });

            Self {
                uniform_buffer,
                uniform_bind_group,
                image_bind_group,
                pipeline,
                vertex_buffer,
            }
        }

        pub fn run<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.set_bind_group(1, &self.image_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

            let num_vertices = VERTICES.len() as u32;
            render_pass.draw(0..num_vertices, 0..1);
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct VertexUniforms {
        view_projection_matrix: glm::Mat4,
        model_matrix: glm::Mat4,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct Vertex {
        position: [f32; 2],
        tex_coords: [f32; 2],
    }

    impl Vertex {
        fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    // @location(0)
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    },
                    // @location(1)
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: std::mem::size_of::<[f32; 2]>() as u64,
                        shader_location: 1,
                    },
                ],
            }
        }
    }

    fn view_projection_matrix(viewport_size: (u32, u32)) -> glm::Mat4 {
        let vw = viewport_size.0 as f32;
        let vh = viewport_size.1 as f32;

        let sw = 0.5_f32;
        let sh = 0.5_f32 * vh / vw;

        // Our ortho camera is just centered at (0, 0)

        let left = -sw;
        let right = sw;
        let bottom = -sh;
        let top = sh;

        // DirectX, Metal, wgpu share the same left-handed coordinate system
        // for their normalized device coordinates:
        // https://github.com/gfx-rs/gfx/tree/master/src/backend/dx12
        glm::ortho_lh_zo(left, right, bottom, top, -1_f32, 1_f32)
    }

    const VERTICES: &[Vertex] = &[
        Vertex {
            position: [-0.5, 0.5],
            tex_coords: [0.0, 0.0],
        },
        Vertex {
            position: [-0.5, -0.5],
            tex_coords: [0.0, 1.0],
        },
        Vertex {
            position: [0.5, -0.5],
            tex_coords: [1.0, 1.0],
        },
        Vertex {
            position: [-0.5, 0.5],
            tex_coords: [0.0, 0.0],
        },
        Vertex {
            position: [0.5, -0.5],
            tex_coords: [1.0, 1.0],
        },
        Vertex {
            position: [0.5, 0.5],
            tex_coords: [1.0, 0.0],
        },
    ];
}

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("weekend-raytracer-wgpu")
        .build(&event_loop)
        .unwrap();
    let mut context = pollster::block_on(GpuContext::new(&window));

    let current_viewport_size = {
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
    let image_buffers = ImageBuffers::new(
        &context.device,
        current_viewport_size,
        max_viewport_resolution,
    );

    let renderer = Renderer::new(
        &context.device,
        &context.surface_config,
        &image_buffers,
        current_viewport_size,
    );

    event_loop.run(move |event, _, _control_flow| {
        *_control_flow = ControlFlow::Wait;

        match event {
            winit::event::Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::CloseRequested => {
                    *_control_flow = ControlFlow::Exit;
                }
                _ => {}
            },
            winit::event::Event::RedrawRequested(window_id) if window_id == window.id() => {
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

                    renderer.run(&mut render_pass);
                }

                context.queue.submit(Some(encoder.finish()));
                frame.present();
            }
            winit::event::Event::MainEventsCleared => {
                window.request_redraw();
            }
            _ => {}
        }
    });
}
