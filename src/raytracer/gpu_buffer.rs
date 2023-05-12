use wgpu::util::DeviceExt;

pub struct UniformBuffer {
    handle: wgpu::Buffer,
    binding_idx: u32,
}

impl UniformBuffer {
    pub fn new(
        device: &wgpu::Device,
        buffer_size: wgpu::BufferAddress,
        binding_idx: u32,
        label: Option<&str>,
    ) -> Self {
        let handle = device.create_buffer(&wgpu::BufferDescriptor {
            size: buffer_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
            label,
        });

        Self {
            handle,
            binding_idx,
        }
    }

    pub fn new_from_bytes(
        device: &wgpu::Device,
        bytes: &[u8],
        binding_idx: u32,
        label: Option<&str>,
    ) -> Self {
        let handle = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            contents: bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            label,
        });

        Self {
            handle,
            binding_idx,
        }
    }

    pub fn handle(&self) -> &wgpu::Buffer {
        &self.handle
    }

    pub fn layout(&self, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding: self.binding_idx,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    pub fn binding(&self) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding: self.binding_idx,
            resource: self.handle.as_entire_binding(),
        }
    }
}

pub struct StorageBuffer {
    handle: wgpu::Buffer,
    binding_idx: u32,
}

impl StorageBuffer {
    pub fn new_from_bytes(
        device: &wgpu::Device,
        bytes: &[u8],
        binding_idx: u32,
        label: Option<&str>,
    ) -> Self {
        let handle = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            label,
        });

        Self {
            handle,
            binding_idx,
        }
    }

    pub fn handle(&self) -> &wgpu::Buffer {
        &self.handle
    }

    pub fn layout(
        &self,
        visibility: wgpu::ShaderStages,
        read_only: bool,
    ) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding: self.binding_idx,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    pub fn binding(&self) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding: self.binding_idx,
            resource: self.handle.as_entire_binding(),
        }
    }
}
