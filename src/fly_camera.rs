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

    pub fn update(&mut self, translation_scale: f32, rotation_scale: f32) {
        {
            let v = |b| if b { 1_f32 } else { 0_f32 };
            let translation = glm::vec3(
                translation_scale * (v(self.right_pressed) - v(self.left_pressed)),
                translation_scale * (v(self.up_pressed) - v(self.down_pressed)),
                translation_scale * (v(self.forward_pressed) - v(self.backward_pressed)),
            );

            let orientation = self.axes();
            self.position += orientation.right * translation.x
                + orientation.up * translation.y
                + orientation.forward * translation.z;
        }

        {
            let rotation = if self.look_pressed {
                (
                    Angle::radians(-rotation_scale * self.mouse_delta.0),
                    Angle::radians(-rotation_scale * self.mouse_delta.1),
                )
            } else {
                (Angle::radians(0.0), Angle::radians(0.0))
            };
            self.yaw = self.yaw + rotation.0;
            self.pitch = self.pitch + rotation.1;
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

struct Orientation {
    forward: glm::Vec3,
    right: glm::Vec3,
    up: glm::Vec3,
}
