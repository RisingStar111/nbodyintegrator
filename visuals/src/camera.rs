use integrator::maths;
use std::f64::consts::PI;

use crate::maths::V3;

pub struct OrthographicOrbitalCamera {
    // to simplify controls and rotations, since there's not really a reason to pan the camera
    // (or even axially rotate it)
    // will use just pos as (latitude, longitude, radius) and define the camera as always pointing at the center
    // (can add 'panning'/focusing easily if wanted) (panning slightly more annoying and less useful)
    pub pos: V3,
    pub view_scale: f64, // could split into vertical/horizontal
    pub mouse_sensitivity: f64,
    pub scroll_sensitivity: f64,
}

impl OrthographicOrbitalCamera {
    pub fn new(pos: V3, view_scale: f64, mouse_sensitivity: f64, scroll_sensitivity: f64) -> Self {
        OrthographicOrbitalCamera { 
            pos, 
            view_scale, 
            mouse_sensitivity, 
            scroll_sensitivity }
    }
    fn pointing(&self) -> V3 {
        V3 {
            x: self.pos.x.cos() * self.pos.y.sin(), 
            y: self.pos.x.sin() * self.pos.y.sin(), 
            z: -self.pos.y.cos()
        } // could store this in the struct ~~and even not separate calls by keeping a dirtiness flag~~
    }

    pub fn world_to_screen(&self, world_pos: &V3) -> V3 {
        // rotate back around z
        let (newx, tempy) = maths::rotate_xy(world_pos.x, world_pos.y, -self.pos.x);
        let (newy, newz) = maths::rotate_xy(tempy, -world_pos.z, -self.pos.y);
        V3 {x: newx * 0.5 / self.view_scale + 0.5, y: newz * 0.5 / self.view_scale + 0.5, z: newy + self.pos.z} // xy 0-1, z is depth
    }

    pub fn rotate(&mut self, x: f64, y: f64) {
        self.pos.x += x;
        self.pos.y += y;
        // lock vertical and loop horizontal rotation
        self.pos.x %= 2.*PI;
        self.pos.y = f64::max(-PI/2., f64::min(self.pos.y, PI/2.));
    }
    pub fn mouse_rotate(&mut self, x: f64, y: f64) {
        self.rotate(x * self.mouse_sensitivity, y * self.mouse_sensitivity);
    }

    pub fn zoom(&mut self, amount: f64) {
        self.view_scale *= amount;
    }
    pub fn scroll_zoom(&mut self, amount: f64) { // bit scuffed
        self.zoom(1. - amount * self.scroll_sensitivity);
    }
}