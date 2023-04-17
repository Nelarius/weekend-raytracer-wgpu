pub use angle::Angle;
use gpu_buffer::{StorageBuffer, UniformBuffer};
use wgpu::util::DeviceExt;

mod angle;
mod gpu_buffer;

pub struct Raytracer {
    vertex_uniform_bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    image_dimensions_buffer: UniformBuffer,
    image_bind_group: wgpu::BindGroup,
    camera_buffer: UniformBuffer,
    scene_bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    latest_render_params: RenderParams,
}

impl Raytracer {
    pub fn new(
        device: &wgpu::Device,
        surface_config: &wgpu::SurfaceConfiguration,
        scene: &Scene,
        render_params: &RenderParams,
        max_viewport_resolution: u32,
    ) -> Self {
        let uniforms = VertexUniforms {
            view_projection_matrix: unit_quad_projection_matrix(),
            model_matrix: glm::identity(),
        };
        let vertex_uniform_buffer = UniformBuffer::new_from_bytes(
            device,
            bytemuck::bytes_of(&uniforms),
            0_u32,
            Some("uniforms"),
        );
        let vertex_uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[vertex_uniform_buffer.layout(wgpu::ShaderStages::VERTEX)],
                label: Some("uniforms layout"),
            });
        let vertex_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &vertex_uniform_bind_group_layout,
            entries: &[vertex_uniform_buffer.binding()],
            label: Some("uniforms bind group"),
        });

        let image_dimensions_buffer = {
            let image_dimensions = [render_params.viewport_size.0, render_params.viewport_size.1];
            UniformBuffer::new_from_bytes(
                device,
                bytemuck::bytes_of(&image_dimensions),
                0_u32,
                Some("image dimensions buffer"),
            )
        };

        let rng_state_buffer = {
            let seed_buffer: Vec<u32> = (0..max_viewport_resolution).collect();
            StorageBuffer::new_from_bytes(
                device,
                bytemuck::cast_slice(seed_buffer.as_slice()),
                1_u32,
                Some("rng seed buffer"),
            )
        };

        let camera_buffer = {
            let camera = GpuCamera::new(&render_params.camera, render_params.viewport_size);

            UniformBuffer::new_from_bytes(
                device,
                bytemuck::bytes_of(&camera),
                0_u32,
                Some("camera buffer"),
            )
        };

        let (scene_bind_group_layout, scene_bind_group) = {
            let sphere_buffer = StorageBuffer::new_from_bytes(
                device,
                bytemuck::cast_slice(scene.spheres.as_slice()),
                1_u32,
                Some("scene buffer"),
            );

            let materials = scene
                .materials
                .iter()
                .map(|material| match material {
                    Material::Lambertian { albedo } => GpuMaterial::lambertian(*albedo),
                    Material::Metal { albedo, fuzz } => GpuMaterial::metal(*albedo, *fuzz),
                    Material::Dielectric { refraction_index } => {
                        GpuMaterial::dielectric(*refraction_index)
                    }
                })
                .collect::<Vec<_>>();
            let material_buffer = StorageBuffer::new_from_bytes(
                device,
                bytemuck::cast_slice(materials.as_slice()),
                2_u32,
                Some("material buffer"),
            );

            let scene_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        camera_buffer.layout(wgpu::ShaderStages::FRAGMENT),
                        sphere_buffer.layout(wgpu::ShaderStages::FRAGMENT, true),
                        material_buffer.layout(wgpu::ShaderStages::FRAGMENT, true),
                    ],
                    label: Some("scene layout"),
                });
            let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &scene_bind_group_layout,
                entries: &[
                    camera_buffer.binding(),
                    sphere_buffer.binding(),
                    material_buffer.binding(),
                ],
                label: Some("scene bind group"),
            });

            (scene_bind_group_layout, scene_bind_group)
        };

        let image_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    image_dimensions_buffer.layout(wgpu::ShaderStages::FRAGMENT),
                    rng_state_buffer.layout(wgpu::ShaderStages::FRAGMENT, false),
                ],
                label: Some("image layout"),
            });
        let image_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &image_bind_group_layout,
            entries: &[
                image_dimensions_buffer.binding(),
                rng_state_buffer.binding(),
            ],
            label: Some("image bind group"),
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            source: wgpu::ShaderSource::Wgsl(include_str!("raytracer.wgsl").into()),
            label: Some("quad.wgsl"),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[
                &vertex_uniform_bind_group_layout,
                &image_bind_group_layout,
                &scene_bind_group_layout,
            ],
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
            vertex_uniform_bind_group,
            image_dimensions_buffer,
            image_bind_group,
            camera_buffer,
            scene_bind_group,
            vertex_buffer,
            pipeline,
            latest_render_params: *render_params,
        }
    }

    pub fn render_frame<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.vertex_uniform_bind_group, &[]);
        render_pass.set_bind_group(1, &self.image_bind_group, &[]);
        render_pass.set_bind_group(2, &self.scene_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

        let num_vertices = VERTICES.len() as u32;
        render_pass.draw(0..num_vertices, 0..1);
    }

    pub fn set_render_params(&mut self, queue: &wgpu::Queue, render_params: RenderParams) {
        if render_params == self.latest_render_params {
            return;
        }

        self.latest_render_params = render_params;

        {
            let image_dimensions = [render_params.viewport_size.0, render_params.viewport_size.1];
            queue.write_buffer(
                &self.image_dimensions_buffer.handle(),
                0,
                bytemuck::bytes_of(&image_dimensions),
            );
        }
        {
            let camera = GpuCamera::new(&render_params.camera, render_params.viewport_size);
            queue.write_buffer(&self.camera_buffer.handle(), 0, bytemuck::bytes_of(&camera));
        }
    }
}

pub struct Scene {
    pub spheres: Vec<Sphere>,
    pub materials: Vec<Material>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Sphere {
    center: glm::Vec3,
    radius: f32,
    material_idx: u32,
    _padding: [u32; 3],
}

impl Sphere {
    pub fn new(center: glm::Vec3, radius: f32, material_idx: u32) -> Self {
        Self {
            center,
            radius,
            material_idx,
            _padding: [0_u32; 3],
        }
    }

    pub fn center(&self) -> glm::Vec3 {
        self.center
    }

    pub fn radius(&self) -> f32 {
        self.radius
    }
}

pub enum Material {
    Lambertian { albedo: glm::Vec3 },
    Metal { albedo: glm::Vec3, fuzz: f32 },
    Dielectric { refraction_index: f32 },
}

#[derive(Clone, Copy, PartialEq)]
pub struct RenderParams {
    pub camera: Camera,
    pub viewport_size: (u32, u32),
}

#[derive(Clone, Copy, PartialEq)]
pub struct Camera {
    pub eye_pos: glm::Vec3,
    pub eye_dir: glm::Vec3,
    pub up: glm::Vec3,
    pub vfov: Angle,
    pub aperture: f32,
    pub focus_distance: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuCamera {
    eye: glm::Vec3,
    _padding1: f32,
    horizontal: glm::Vec3,
    _padding2: f32,
    vertical: glm::Vec3,
    _padding3: f32,
    u: glm::Vec3,
    _padding4: f32,
    v: glm::Vec3,
    lens_radius: f32,
    lower_left_corner: glm::Vec3,
    _padding5: f32,
}

impl GpuCamera {
    pub fn new(camera: &Camera, viewport_size: (u32, u32)) -> Self {
        let lens_radius = 0.5_f32 * camera.aperture;
        let aspect = viewport_size.0 as f32 / viewport_size.1 as f32;
        let theta = camera.vfov.as_radians();
        let half_height = camera.focus_distance * (0.5_f32 * theta).tan();
        let half_width = aspect * half_height;

        let w = glm::normalize(&camera.eye_dir);
        let v = glm::normalize(&camera.up);
        let u = glm::cross(&w, &v);

        let lower_left_corner =
            camera.eye_pos + camera.focus_distance * w - half_width * u - half_height * v;
        let horizontal = 2_f32 * half_width * u;
        let vertical = 2_f32 * half_height * v;

        Self {
            eye: camera.eye_pos,
            _padding1: 0_f32,
            horizontal,
            _padding2: 0_f32,
            vertical,
            _padding3: 0_f32,
            u,
            _padding4: 0_f32,
            v,
            lens_radius,
            lower_left_corner,
            _padding5: 0_f32,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuMaterial {
    albedo: glm::Vec3,
    x: f32,
    id: u32,
    _padding: [u32; 3],
}

impl GpuMaterial {
    pub fn lambertian(albedo: glm::Vec3) -> Self {
        Self {
            id: 0,
            albedo,
            x: 0_f32,
            _padding: [0_u32; 3],
        }
    }

    pub fn metal(albedo: glm::Vec3, fuzz: f32) -> Self {
        Self {
            id: 1,
            albedo,
            x: fuzz,
            _padding: [0_u32; 3],
        }
    }

    pub fn dielectric(refraction_index: f32) -> Self {
        Self {
            id: 2,
            albedo: glm::vec3(0_f32, 0_f32, 0_f32),
            x: refraction_index,
            _padding: [0_u32; 3],
        }
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

fn unit_quad_projection_matrix() -> glm::Mat4 {
    let sw = 0.5_f32;
    let sh = 0.5_f32;

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
