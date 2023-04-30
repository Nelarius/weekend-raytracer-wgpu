#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Angle {
    radians: f32,
}

impl Angle {
    #[allow(dead_code)]
    pub fn degrees(degrees: f32) -> Self {
        Self {
            radians: degrees * std::f32::consts::PI / 180_f32,
        }
    }

    #[allow(dead_code)]
    pub fn radians(radians: f32) -> Self {
        Self { radians }
    }

    #[allow(dead_code)]
    pub fn as_degrees(&self) -> f32 {
        self.radians * 180_f32 / std::f32::consts::PI
    }

    #[allow(dead_code)]
    pub fn as_radians(&self) -> f32 {
        self.radians
    }

    pub fn clamp(mut self, min: Self, max: Self) -> Self {
        // NOTE: f32 has it's own implementation of clamp, which does not rely on Ord being implemented.
        // f32 does not have total order, and thus we can't rely on #[derive(Ord)] here.
        if self.radians < min.radians {
            self.radians = min.radians;
        }
        if self.radians > max.radians {
            self.radians = max.radians;
        }
        self
    }
}

impl std::ops::Add for Angle {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            radians: self.radians + rhs.radians,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_angle_to_radians() {
        let angle = Angle::degrees(90_f32);
        assert_eq!(angle.as_radians(), std::f32::consts::FRAC_PI_2);
    }

    #[test]
    fn test_angle_to_degrees() {
        let angle = Angle::radians(std::f32::consts::FRAC_PI_2);
        assert_eq!(angle.as_degrees(), 90_f32);
    }

    #[test]
    fn test_angle_add() {
        let lhs = Angle::degrees(90_f32);
        let rhs = Angle::degrees(90_f32);
        let result = lhs + rhs;
        assert_eq!(result.as_degrees(), 180_f32);
    }

    #[test]
    fn test_angle_clamp_max() {
        let angle = Angle::degrees(90_f32);
        let min = Angle::degrees(0_f32);
        let max = Angle::degrees(45_f32);
        let result = angle.clamp(min, max);
        assert_eq!(result.as_degrees(), 45_f32);
    }

    #[test]
    fn test_angle_clamp_min() {
        let angle = Angle::degrees(-90_f32);
        let min = Angle::degrees(0_f32);
        let max = Angle::degrees(45_f32);
        let result = angle.clamp(min, max);
        assert_eq!(result.as_degrees(), 0_f32);
    }
}
