pub use angle::Angle;
use gpu_buffer::{StorageBuffer, UniformBuffer};
use wgpu::util::DeviceExt;

use thiserror::Error;

mod angle;
mod gpu_buffer;

pub struct Raytracer {
    vertex_uniform_bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    image_dimensions_buffer: UniformBuffer,
    image_bind_group: wgpu::BindGroup,
    camera_buffer: UniformBuffer,
    sampling_parameter_buffer: UniformBuffer,
    parameter_bind_group: wgpu::BindGroup,
    scene_bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    latest_render_params: RenderParams,
    render_progress: RenderProgress,
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

        let image_buffer = {
            let buffer = vec![[0_f32; 3]; max_viewport_resolution as usize];
            StorageBuffer::new_from_bytes(
                device,
                bytemuck::cast_slice(buffer.as_slice()),
                1_u32,
                Some("image buffer"),
            )
        };

        let rng_state_buffer = {
            let seed_buffer: Vec<u32> = (0..max_viewport_resolution).collect();
            StorageBuffer::new_from_bytes(
                device,
                bytemuck::cast_slice(seed_buffer.as_slice()),
                2_u32,
                Some("rng seed buffer"),
            )
        };

        let image_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    image_dimensions_buffer.layout(wgpu::ShaderStages::FRAGMENT),
                    image_buffer.layout(wgpu::ShaderStages::FRAGMENT, false),
                    rng_state_buffer.layout(wgpu::ShaderStages::FRAGMENT, false),
                ],
                label: Some("image layout"),
            });
        let image_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &image_bind_group_layout,
            entries: &[
                image_dimensions_buffer.binding(),
                image_buffer.binding(),
                rng_state_buffer.binding(),
            ],
            label: Some("image bind group"),
        });

        let camera_buffer = {
            let camera = GpuCamera::new(&render_params.camera, render_params.viewport_size);

            UniformBuffer::new_from_bytes(
                device,
                bytemuck::bytes_of(&camera),
                0_u32,
                Some("camera buffer"),
            )
        };

        let sampling_parameter_buffer = {
            // TODO: these should be inlcuded in the render params
            let sampling_params = GpuSamplingParams {
                num_samples_per_pixel: 2_u32,
                num_bounces: 6_u32,
                accumulated_samples_per_pixel: 0_u32,
                clear_accumulated_samples: 0_u32,
            };

            UniformBuffer::new_from_bytes(
                device,
                bytemuck::bytes_of(&sampling_params),
                1_u32,
                Some("sampling parameter buffer"),
            )
        };

        let parameter_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    camera_buffer.layout(wgpu::ShaderStages::FRAGMENT),
                    sampling_parameter_buffer.layout(wgpu::ShaderStages::FRAGMENT),
                ],
                label: Some("parameter layout"),
            });

        let parameter_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &parameter_bind_group_layout,
            entries: &[camera_buffer.binding(), sampling_parameter_buffer.binding()],
            label: Some("parameter bind group"),
        });

        let (scene_bind_group_layout, scene_bind_group) = {
            let sphere_buffer = StorageBuffer::new_from_bytes(
                device,
                bytemuck::cast_slice(scene.spheres.as_slice()),
                0_u32,
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
                1_u32,
                Some("material buffer"),
            );

            let scene_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        sphere_buffer.layout(wgpu::ShaderStages::FRAGMENT, true),
                        material_buffer.layout(wgpu::ShaderStages::FRAGMENT, true),
                    ],
                    label: Some("scene layout"),
                });
            let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &scene_bind_group_layout,
                entries: &[sphere_buffer.binding(), material_buffer.binding()],
                label: Some("scene bind group"),
            });

            (scene_bind_group_layout, scene_bind_group)
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            source: wgpu::ShaderSource::Wgsl(include_str!("raytracer.wgsl").into()),
            label: Some("quad.wgsl"),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[
                &vertex_uniform_bind_group_layout,
                &image_bind_group_layout,
                &parameter_bind_group_layout,
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

        let render_progress = RenderProgress::new();

        Self {
            vertex_uniform_bind_group,
            image_dimensions_buffer,
            image_bind_group,
            camera_buffer,
            sampling_parameter_buffer,
            parameter_bind_group,
            scene_bind_group,
            vertex_buffer,
            pipeline,
            latest_render_params: *render_params,
            render_progress,
        }
    }

    pub fn render_frame<'a>(
        &'a mut self,
        queue: &wgpu::Queue,
        render_pass: &mut wgpu::RenderPass<'a>,
    ) {
        {
            let gpu_sampling_params = self
                .render_progress
                .next_frame(&self.latest_render_params.sampling);

            queue.write_buffer(
                &self.sampling_parameter_buffer.handle(),
                0,
                bytemuck::cast_slice(&[gpu_sampling_params]),
            );
        }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.vertex_uniform_bind_group, &[]);
        render_pass.set_bind_group(1, &self.image_bind_group, &[]);
        render_pass.set_bind_group(2, &self.parameter_bind_group, &[]);
        render_pass.set_bind_group(3, &self.scene_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

        let num_vertices = VERTICES.len() as u32;
        render_pass.draw(0..num_vertices, 0..1);
    }

    pub fn set_render_params(
        &mut self,
        queue: &wgpu::Queue,
        render_params: RenderParams,
    ) -> Result<(), RenderParamsValidationError> {
        if render_params == self.latest_render_params {
            return Ok(());
        }

        if render_params.sampling.max_samples_per_pixel
            % render_params.sampling.num_samples_per_pixel
            != 0
        {
            return Err(RenderParamsValidationError::MaxSampleCountNotMultiple(
                render_params.sampling.max_samples_per_pixel,
                render_params.sampling.num_samples_per_pixel,
            ));
        }

        if render_params.viewport_size.0 == 0_u32 || render_params.viewport_size.1 == 0_u32 {
            return Err(RenderParamsValidationError::ViewportSize(
                render_params.viewport_size.0,
                render_params.viewport_size.1,
            ));
        }

        if !(Angle::degrees(0.0)..=Angle::degrees(90.0)).contains(&render_params.camera.vfov) {
            return Err(RenderParamsValidationError::VfovOutOfRange(
                render_params.camera.vfov.as_degrees(),
            ));
        }

        if !(0.0..=1.0).contains(&render_params.camera.aperture) {
            return Err(RenderParamsValidationError::ApertureOutOfRange(
                render_params.camera.aperture,
            ));
        }

        if render_params.camera.focus_distance < 0.0 {
            return Err(RenderParamsValidationError::FocusDistanceOutOfRange(
                render_params.camera.focus_distance,
            ));
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

        self.render_progress.reset();

        Ok(())
    }

    pub fn progress(&self) -> f32 {
        self.render_progress.accumulated_samples() as f32
            / self.latest_render_params.sampling.max_samples_per_pixel as f32
    }
}

#[derive(Error, Debug)]
pub enum RenderParamsValidationError {
    #[error("max_samples_per_pixel ({0}) is not a multiple of num_samples_per_pixel ({1})")]
    MaxSampleCountNotMultiple(u32, u32),
    #[error("viewport_size elements cannot be zero: ({0}, {1})")]
    ViewportSize(u32, u32),
    #[error("vfov must be between 0..=90 degrees")]
    VfovOutOfRange(f32),
    #[error("aperture must be between 0..=1")]
    ApertureOutOfRange(f32),
    #[error("focus_distance must be greater than zero")]
    FocusDistanceOutOfRange(f32),
}

pub struct Scene {
    pub spheres: Vec<Sphere>,
    pub materials: Vec<Material>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Sphere {
    // NOTE: naga memory alignment issue, see discussion at
    // https://github.com/gfx-rs/naga/issues/2000
    // It's safer to just use Vec4 instead of Vec3.
    center: glm::Vec4,  // 0 byte offset
    radius: f32,        // 16 byte offset
    material_idx: u32,  // 20 byte offset
    _padding: [u32; 2], // 24 byte offset, 8 bytes size
}

impl Sphere {
    pub fn new(center: glm::Vec3, radius: f32, material_idx: u32) -> Self {
        Self {
            center: glm::vec3_to_vec4(&center),
            radius,
            material_idx,
            _padding: [0_u32; 2],
        }
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
    pub sampling: SamplingParams,
    pub viewport_size: (u32, u32),
}

#[derive(Clone, Copy, PartialEq)]
pub struct Camera {
    pub eye_pos: glm::Vec3,
    pub eye_dir: glm::Vec3,
    pub up: glm::Vec3,
    /// Angle must be between 0..=90 degrees.
    pub vfov: Angle,
    /// Aperture must be between 0..=1.
    pub aperture: f32,
    /// Focus distance must be a positive number.
    pub focus_distance: f32,
}

#[derive(Clone, Copy, PartialEq)]
pub struct SamplingParams {
    pub max_samples_per_pixel: u32,
    pub num_samples_per_pixel: u32,
    pub num_bounces: u32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_samples_per_pixel: 128_u32,
            num_samples_per_pixel: 2_u32,
            num_bounces: 8_u32,
        }
    }
}

struct RenderProgress {
    accumulated_samples_per_pixel: u32,
}

impl RenderProgress {
    pub fn new() -> Self {
        Self {
            accumulated_samples_per_pixel: 0_u32,
        }
    }

    pub fn next_frame(&mut self, sampling_params: &SamplingParams) -> GpuSamplingParams {
        let current_accumulated_samples = self.accumulated_samples_per_pixel;
        let next_accumulated_samples =
            sampling_params.num_samples_per_pixel + current_accumulated_samples;

        // Initial state: no samples have been accumulated yet. This is the first frame
        // after a reset. The image buffer's previous samples should be cleared by
        // setting clear_accumulated_samples to 1_u32.
        if current_accumulated_samples == 0_u32 {
            self.accumulated_samples_per_pixel = next_accumulated_samples;
            GpuSamplingParams {
                num_samples_per_pixel: sampling_params.num_samples_per_pixel,
                num_bounces: sampling_params.num_bounces,
                accumulated_samples_per_pixel: next_accumulated_samples,
                clear_accumulated_samples: 1_u32,
            }
        }
        // Progressive render: accumulating samples in the image buffer over multiple
        // frames.
        else if next_accumulated_samples <= sampling_params.max_samples_per_pixel {
            self.accumulated_samples_per_pixel = next_accumulated_samples;
            GpuSamplingParams {
                num_samples_per_pixel: sampling_params.num_samples_per_pixel,
                num_bounces: sampling_params.num_bounces,
                accumulated_samples_per_pixel: next_accumulated_samples,
                clear_accumulated_samples: 0_u32,
            }
        }
        // Completed render: we have accumulated max_samples_per_pixel samples. Stop rendering
        // by setting num_samples_per_pixel to zero.
        else {
            GpuSamplingParams {
                num_samples_per_pixel: 0_u32,
                num_bounces: sampling_params.num_bounces,
                accumulated_samples_per_pixel: current_accumulated_samples,
                clear_accumulated_samples: 0_u32,
            }
        }
    }

    pub fn reset(&mut self) {
        self.accumulated_samples_per_pixel = 0_u32;
    }

    pub fn accumulated_samples(&self) -> u32 {
        self.accumulated_samples_per_pixel
    }
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
    albedo: glm::Vec4,   // 0 bytes offset
    x: f32,              // 16 bytes offset
    id: u32,             // 20 bytes offset
    _padding2: [u32; 2], // 24 bytes offset, 8 bytes size
}

impl GpuMaterial {
    pub fn lambertian(albedo: glm::Vec3) -> Self {
        Self {
            albedo: glm::vec3_to_vec4(&albedo),
            x: 0_f32,
            id: 0,
            _padding2: [0_u32; 2],
        }
    }

    pub fn metal(albedo: glm::Vec3, fuzz: f32) -> Self {
        Self {
            albedo: glm::vec3_to_vec4(&albedo),
            x: fuzz,
            id: 1,
            _padding2: [0_u32; 2],
        }
    }

    pub fn dielectric(refraction_index: f32) -> Self {
        Self {
            albedo: glm::vec4(0_f32, 0_f32, 0_f32, 0_f32),
            x: refraction_index,
            id: 2,
            _padding2: [0_u32; 2],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSamplingParams {
    num_samples_per_pixel: u32,
    num_bounces: u32,
    accumulated_samples_per_pixel: u32,
    clear_accumulated_samples: u32,
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
