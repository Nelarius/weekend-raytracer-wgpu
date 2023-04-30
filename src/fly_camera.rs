use winit::event::{ElementState, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent};

use crate::raytracer::{Angle, Camera};

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
    pub mouse_pos: (f32, f32),
}

impl Default for FlyCameraController {
    fn default() -> Self {
        let look_from = glm::vec3(-10.0, 2.0, -4.0);
        let look_at = glm::vec3(0.0, 1.0, 0.0);
        let focus_distance = glm::magnitude(&(look_at - look_from));

        Self {
            position: look_from,
            yaw: Angle::degrees(25_f32),
            pitch: Angle::degrees(-10_f32),
            vfov_degrees: 30.0,
            aperture: 0.8,
            focus_distance,
            forward_pressed: false,
            backward_pressed: false,
            left_pressed: false,
            right_pressed: false,
            up_pressed: false,
            down_pressed: false,
            look_pressed: false,
            previous_mouse_pos: None,
            mouse_pos: (0.0, 0.0),
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
                self.mouse_pos = (position.x as f32, position.y as f32);
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

    pub fn after_events(&mut self, viewport_size: (u32, u32), translation_scale: f32) {
        if self.look_pressed {
            if let Some(prev_mouse_pos) = self.previous_mouse_pos {
                let orientation = camera_orientation(self);
                let c1 = orientation.right;
                let c2 = orientation.forward;
                let c3 = glm::normalize(&glm::cross(&c1, &c2));
                let from_local = glm::mat3(c1.x, c2.x, c3.x, c1.y, c2.y, c3.y, c1.z, c2.z, c3.z);
                let to_local = glm::inverse(&from_local);

                // Perform cartesian to spherical coordinate conversion in camera-local space,
                // where z points straight into the screen. That way there is no need to worry
                // about which quadrant of the sphere we are in for the conversion.
                let current_dir =
                    to_local * generate_camera_ray_dir(self, self.mouse_pos, viewport_size);
                let previous_dir =
                    to_local * generate_camera_ray_dir(self, prev_mouse_pos, viewport_size);

                let x1 = current_dir.x;
                let y1 = current_dir.y;
                let z1 = current_dir.z;

                let x2 = previous_dir.x;
                let y2 = previous_dir.y;
                let z2 = previous_dir.z;

                let p1 = z1.acos();
                let p2 = z2.acos();

                let a1 = y1.signum() * (x1 / (x1 * x1 + y1 * y1).sqrt()).acos();
                let a2 = y2.signum() * (x2 / (x2 * x2 + y2 * y2).sqrt()).acos();

                self.yaw = self.yaw + Angle::radians(a1 - a2);
                self.pitch = (self.pitch + Angle::radians(p1 - p2))
                    .clamp(Angle::degrees(-89.0), Angle::degrees(89.0));
            }
        }

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

        self.previous_mouse_pos = Some(self.mouse_pos);
    }
}

fn generate_camera_ray_dir(
    camera: &FlyCameraController,
    mouse_pos: (f32, f32),
    viewport_size: (u32, u32),
) -> glm::Vec3 {
    let aspect_ratio = viewport_size.0 as f32 / viewport_size.1 as f32;
    let half_height =
        camera.focus_distance * (0.5 * Angle::degrees(camera.vfov_degrees).as_radians()).tan();
    let half_width = aspect_ratio * half_height;

    let x = mouse_pos.0 / (viewport_size.0 as f32);
    let y = mouse_pos.1 / (viewport_size.1 as f32);

    let orientation = camera_orientation(camera);

    let point_on_plane = camera.position
        + camera.focus_distance * orientation.forward
        + (2_f32 * x - 1_f32) * half_width * orientation.right
        + (1_f32 - 2_f32 * y) * half_height * orientation.up;

    glm::normalize(&(point_on_plane - camera.position))
}

struct Orientation {
    forward: glm::Vec3,
    right: glm::Vec3,
    up: glm::Vec3,
}

fn camera_orientation(camera: &FlyCameraController) -> Orientation {
    let forward = glm::normalize(&glm::vec3(
        camera.yaw.as_radians().cos() * camera.pitch.as_radians().cos(),
        camera.pitch.as_radians().sin(),
        camera.yaw.as_radians().sin() * camera.pitch.as_radians().cos(),
    ));
    let world_up = glm::vec3(0.0, 1.0, 0.0);
    let right = glm::cross(&forward, &world_up);
    let up = glm::cross(&right, &forward);
    Orientation { forward, right, up }
}
