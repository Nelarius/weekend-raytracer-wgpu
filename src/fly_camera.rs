use winit::event::{ElementState, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent};

use crate::raytracer::{Angle, Camera, Scene, Sphere};

pub struct FlyCameraController {
    pub position: glm::Vec3,
    pub yaw: Angle,
    pub pitch: Angle,
    pub vfov_degrees: f32,
    pub aperture: f32,
    pub focus_distance: f32,

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

impl Default for FlyCameraController {
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

impl FlyCameraController {
    pub fn renderer_camera(&self) -> Camera {
        let orientation = camera_orientation(self);
        Camera {
            eye_pos: self.position,
            eye_dir: orientation.forward,
            up: orientation.up,
            vfov: Angle::degrees(self.vfov_degrees),
            aperture: self.aperture,
            focus_distance: self.focus_distance,
        }
    }

    pub fn begin_frame(&mut self) {
        self.mouse_delta = (0.0, 0.0);
    }

    pub fn handle_event(&mut self, event: &WindowEvent<'_>) {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match keycode {
                    VirtualKeyCode::W => {
                        self.forward_pressed = is_pressed;
                    }
                    VirtualKeyCode::S => {
                        self.backward_pressed = is_pressed;
                    }
                    VirtualKeyCode::A => {
                        self.left_pressed = is_pressed;
                    }
                    VirtualKeyCode::D => {
                        self.right_pressed = is_pressed;
                    }
                    VirtualKeyCode::Q => {
                        self.down_pressed = is_pressed;
                    }
                    VirtualKeyCode::E => {
                        self.up_pressed = is_pressed;
                    }
                    _ => {}
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let position = (position.x as f32, position.y as f32);
                if let Some(previous_position) = self.previous_mouse_pos {
                    self.mouse_delta = (
                        position.0 - previous_position.0,
                        previous_position.1 - position.1,
                    );
                }
                self.previous_mouse_pos = Some(position);
            }

            WindowEvent::MouseInput { button, state, .. } => match button {
                MouseButton::Right => {
                    self.look_pressed = *state == ElementState::Pressed;
                }
                _ => {}
            },

            _ => {}
        }
    }

    pub fn update(
        &mut self,
        scene: &Scene,
        viewport_size: (u32, u32),
        translation_scale: f32,
        rotation_scale: f32,
    ) {
        {
            let v = |b| if b { 1_f32 } else { 0_f32 };
            let translation = glm::vec3(
                translation_scale * (v(self.right_pressed) - v(self.left_pressed)),
                translation_scale * (v(self.up_pressed) - v(self.down_pressed)),
                translation_scale * (v(self.forward_pressed) - v(self.backward_pressed)),
            );

            let orientation = camera_orientation(self);
            self.position += orientation.right * translation.x
                + orientation.up * translation.y
                + orientation.forward * translation.z;
        }

        {
            if self.look_pressed {
                if let Some(prev_mouse_pos) = self.previous_mouse_pos {
                    let ray = generate_camera_ray(self, prev_mouse_pos, viewport_size);
                    let radius = if let Some(p) = ray_intersect_scene(ray, scene) {
                        glm::distance(&self.position, &p)
                    } else {
                        10_f32
                    };

                    // From arc length: L = theta * r
                    let t = -rotation_scale / radius;
                    let yaw_delta = Angle::radians(t * self.mouse_delta.0);
                    let pitch_delta = Angle::radians(t * self.mouse_delta.1);

                    self.yaw = self.yaw + yaw_delta;
                    self.pitch = self.pitch + pitch_delta;
                }
            }
        }
    }
}

struct Orientation {
    forward: glm::Vec3,
    right: glm::Vec3,
    up: glm::Vec3,
}

#[derive(Copy, Clone)]
struct Ray {
    origin: glm::Vec3,
    direction: glm::Vec3,
}

impl Ray {
    pub fn point_at_t(&self, t: f32) -> glm::Vec3 {
        self.origin + self.direction * t
    }
}

fn generate_camera_ray(
    camera: &FlyCameraController,
    mouse_pos: (f32, f32),
    viewport_size: (u32, u32),
) -> Ray {
    let aspect_ratio = viewport_size.0 as f32 / viewport_size.1 as f32;
    let half_height =
        camera.focus_distance * (0.5 * Angle::degrees(camera.vfov_degrees).as_radians()).tan();
    let half_width = aspect_ratio * half_height;

    let x = mouse_pos.0 / (viewport_size.0 as f32);
    let y = mouse_pos.1 / (viewport_size.1 as f32);

    let origin = camera.position;
    let orientation = camera_orientation(camera);

    // Decomposition:

    // let top_left = camera.position
    //     + camera.focus_distance * orientation.forward
    //     - half_width * orientation.right;
    //     + half_height * orientation.up

    // let point_on_plane = top_left
    //     + 2_f32 * x * half_width * orientation.right
    //     - 2_f32 * y * half_height * orientation.up;

    let point_on_plane = camera.position
        + camera.focus_distance * orientation.forward
        + (2_f32 * x - 1_f32) * half_width * orientation.right
        + (1_f32 - 2_f32 * y) * half_height * orientation.up;

    Ray {
        origin,
        direction: glm::normalize(&(point_on_plane - camera.position)),
    }
}

fn camera_orientation(camera: &FlyCameraController) -> Orientation {
    let forward = glm::normalize(&glm::vec3(
        camera.yaw.as_radians().cos() * camera.pitch.as_radians().cos(),
        camera.pitch.as_radians().sin(),
        camera.yaw.as_radians().sin() * camera.pitch.as_radians().cos(),
    ));
    let up = glm::vec3(0.0, 1.0, 0.0);
    let right = glm::cross(&forward, &up);
    Orientation { forward, right, up }
}

fn ray_intersect_scene(ray: Ray, scene: &Scene) -> Option<glm::Vec3> {
    let mut closest_t = f32::INFINITY;
    let mut closest_intersection = None;

    for sphere in &scene.spheres {
        if let Some(intersection) = ray_intersect_sphere(ray, sphere, 0.001, closest_t) {
            closest_t = glm::distance(&ray.origin, &intersection);
            closest_intersection = Some(intersection);
        }
    }

    closest_intersection
}

fn ray_intersect_sphere(ray: Ray, sphere: &Sphere, tmin: f32, tmax: f32) -> Option<glm::Vec3> {
    let oc = ray.origin - sphere.center();
    let a = glm::dot(&ray.direction, &ray.direction);
    let b = glm::dot(&oc, &ray.direction);
    let c = glm::dot(&oc, &oc) - sphere.radius() * sphere.radius();
    let discriminant = b * b - a * c;

    if discriminant > 0_f32 {
        let mut t = (-b - (b * b - a * c).sqrt()) / a;
        if t < tmax && t > tmin {
            return Some(ray.point_at_t(t));
        }

        t = (-b + (b * b - a * c).sqrt()) / a;
        if t < tmax && t > tmin {
            return Some(ray.point_at_t(t));
        }
    }

    None
}
