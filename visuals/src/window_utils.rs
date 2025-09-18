use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::platform::modifier_supplement::KeyEventExtModifierSupplement;
use std::num::NonZeroU32;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use winit::event::{ElementState, Event, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{self, Window};
use softbuffer::Buffer;

pub struct SubWindow {
    left: isize,
    top: isize,
    pub width: isize,
    height: isize,
}

impl SubWindow {
    pub fn new(left: isize, top: isize, width: isize, height: isize) -> Self {
        SubWindow {left, top, width, height}
    }

    pub fn right(&self) -> isize {
        self.left + self.width
    }
    pub fn bottom(&self) -> isize {
        self.top + self.height
    }
    pub fn set_right(&mut self, right: isize) {
        self.width = right - self.left
    }
    pub fn set_bottom(&mut self, bottom: isize) {
        self.height = bottom - self.top
    }
    pub fn set_rect(&mut self, left: isize, top: isize, right: isize, bottom: isize) {
        self.left = left;
        self.top = top;
        self.set_right(right);
        self.set_bottom(bottom);
    }

    pub fn contains_pixel(&self, x: isize, y: isize) -> bool {
        let mut h = self.left <= x && x < self.right();
        let mut v = self.top <= y && y < self.bottom();
        if self.left > self.right() { h = self.left >= x && x > self.right() }
        if self.top > self.bottom() { v = self.top >= x && x > self.bottom() }
        h & v
    }
    pub fn from_uv(&self, x: f64, y: f64) -> (isize, isize) {
        (
            (self.width as f64 * x) as isize + self.left,
            (self.height as f64 * y) as isize + self.top
        )
    }
}

pub struct Button<F> where F: Fn() { // idea is to have buttons below each board to eg copy board to editor/play/start solver/redo/undo/etc
    window: SubWindow,
    callback: F
}

impl<F> Button<F> where F: Fn() { // not sure how useful this is
    fn click(&self) {
        (self.callback)();
    }
}