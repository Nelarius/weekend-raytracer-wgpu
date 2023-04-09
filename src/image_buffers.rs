use super::gpu_buffer::{StorageBuffer, UniformBuffer};

pub struct ImageBuffers {
    pub image_dimensions_buffer: UniformBuffer,
    pub rng_seed_buffer: StorageBuffer,
}

impl ImageBuffers {
    pub fn new(device: &wgpu::Device, image_dimensions: (u32, u32), max_buffer_size: u32) -> Self {
        let rng_seed_buffer = {
            let seed_buffer: Vec<u32> = (0..max_buffer_size).collect();

            StorageBuffer::new_from_bytes(
                device,
                bytemuck::cast_slice(seed_buffer.as_slice()),
                Some("rng seed buffer"),
            )
        };

        let image_dimensions_buffer = {
            let image_dimensions = [image_dimensions.0, image_dimensions.1];

            UniformBuffer::new_from_bytes(
                device,
                bytemuck::bytes_of(&image_dimensions),
                Some("image dimensions buffer"),
            )
        };

        Self {
            image_dimensions_buffer,
            rng_seed_buffer,
        }
    }
}
