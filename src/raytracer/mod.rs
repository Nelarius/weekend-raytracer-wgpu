use gpu_buffer::{StorageBuffer, UniformBuffer};
use wgpu::util::DeviceExt;

mod gpu_buffer;

pub struct Raytracer {
    vertex_uniform_buffer: UniformBuffer,
    vertex_uniform_bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    image_bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
}

impl Raytracer {
    pub fn new(
        device: &wgpu::Device,
        surface_config: &wgpu::SurfaceConfiguration,
        viewport_size: (u32, u32),
        max_viewport_resolution: u32,
    ) -> Self {
        let uniforms = VertexUniforms {
            view_projection_matrix: view_projection_matrix(viewport_size),
            model_matrix: glm::identity(),
        };
        let vertex_uniform_buffer =
            UniformBuffer::new_from_bytes(device, bytemuck::bytes_of(&uniforms), Some("uniforms"));
        let vertex_uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[vertex_uniform_buffer.layout(0, wgpu::ShaderStages::VERTEX)],
                label: Some("uniforms layout"),
            });
        let vertex_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &vertex_uniform_bind_group_layout,
            entries: &[vertex_uniform_buffer.binding(0)],
            label: Some("uniforms bind group"),
        });

        let image_dimensions_buffer = {
            let image_dimensions = [viewport_size.0, viewport_size.1];
            UniformBuffer::new_from_bytes(
                device,
                bytemuck::bytes_of(&image_dimensions),
                Some("image dimensions buffer"),
            )
        };

        let rng_state_buffer = {
            let seed_buffer: Vec<u32> = (0..max_viewport_resolution).collect();
            StorageBuffer::new_from_bytes(
                device,
                bytemuck::cast_slice(seed_buffer.as_slice()),
                Some("rng seed buffer"),
            )
        };

        let image_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    image_dimensions_buffer.layout(0, wgpu::ShaderStages::FRAGMENT),
                    rng_state_buffer.layout(1, wgpu::ShaderStages::FRAGMENT, false),
                ],
                label: Some("image layout"),
            });
        let image_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &image_bind_group_layout,
            entries: &[
                image_dimensions_buffer.binding(0),
                rng_state_buffer.binding(1),
            ],
            label: Some("image bind group"),
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            source: wgpu::ShaderSource::Wgsl(include_str!("raytracer.wgsl").into()),
            label: Some("quad.wgsl"),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&vertex_uniform_bind_group_layout, &image_bind_group_layout],
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
            vertex_uniform_buffer,
            vertex_uniform_bind_group,
            image_bind_group,
            vertex_buffer,
            pipeline,
        }
    }

    pub fn render_frame<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.vertex_uniform_bind_group, &[]);
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
