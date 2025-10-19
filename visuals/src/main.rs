use integrator::maths::rotate_xy_accurate;
use integrator::traits::{Solver, System};
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::platform::modifier_supplement::KeyEventExtModifierSupplement;
use std::f64::consts::PI;
use std::num::NonZeroU32;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::{f64, thread};
use std::time::{Duration, Instant};
use winit::event::{DeviceEvent, ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{self, Window};
use softbuffer::Buffer;

mod winit_app;
mod window_utils;
use integrator::{maths, solvers, systems};
mod camera;
mod graph;

trait Renderable {
    fn render(&self, camera: &camera::OrthographicOrbitalCamera, window: &window_utils::SubWindow, buffer: &mut Buffer<Rc<Window>, Rc<Window>>, window_width: usize, colour: u32);
}

impl Renderable for systems::ParticleSystem {
    fn render(&self, camera: &camera::OrthographicOrbitalCamera, window: &window_utils::SubWindow, buffer: &mut Buffer<Rc<Window>, Rc<Window>>, window_width: usize, colour: u32) {
        for particle in &self.particles {
            let screen_pos = camera.world_to_screen(&particle.position);
            if screen_pos.x < 0. || screen_pos.y < 0. {continue} // no longer needed but probably faster
            // convert to pixel
            let buffer_pos = window.from_uv(screen_pos.x, screen_pos.y);
            if window.contains_pixel(buffer_pos.0, buffer_pos.1) {
                // render pixel
                let i = buffer_pos.0 + buffer_pos.1 * window_width as isize;
                if i >= 0 && i < buffer.len() as isize {
                    buffer[i as usize] = colour;
                }
            }
        }
    }
}
impl Renderable for systems::Wedge {
    fn render(&self, camera: &camera::OrthographicOrbitalCamera, window: &window_utils::SubWindow, buffer: &mut Buffer<Rc<Window>, Rc<Window>>, window_width: usize, colour: u32) {
        self.base.render(camera, window, buffer, window_width, colour);
        for particle in &self.base.particles {
            for k in 0..self.ghost_wedges {
                let pos = [rotate_xy_accurate(particle.position.x, particle.position.y, self.angle*(k as f64+1.)).0, rotate_xy_accurate(particle.position.x, particle.position.y, self.angle*(k as f64+1.)).1, particle.position.z];
                let screen_pos = camera.world_to_screen(&pos.into());
                if screen_pos.x < 0. || screen_pos.y < 0. {continue} // no longer needed but probably faster
                // convert to pixel
                let buffer_pos = window.from_uv(screen_pos.x, screen_pos.y);
                if window.contains_pixel(buffer_pos.0, buffer_pos.1) {
                    // render pixel
                    let i = buffer_pos.0 + buffer_pos.1 * window_width as isize;
                    if i >= 0 && i < buffer.len() as isize {
                        buffer[i as usize] = 0x333333;
                    }
                }
                let pos = [rotate_xy_accurate(particle.position.x, particle.position.y, -self.angle*(k as f64+1.)).0, rotate_xy_accurate(particle.position.x, particle.position.y, -self.angle*(k as f64+1.)).1, particle.position.z];
                let screen_pos = camera.world_to_screen(&pos.into());
                if screen_pos.x < 0. || screen_pos.y < 0. {continue} // no longer needed but probably faster
                // convert to pixel
                let buffer_pos = window.from_uv(screen_pos.x, screen_pos.y);
                if window.contains_pixel(buffer_pos.0, buffer_pos.1) {
                    // render pixel
                    let i = buffer_pos.0 + buffer_pos.1 * window_width as isize;
                    if i >= 0 && i < buffer.len() as isize {
                        buffer[i as usize] = 0x333333;
                    }
                }
            }
        }
    }
}

impl Renderable for graph::PointGraph {
    fn render(&self, _: &camera::OrthographicOrbitalCamera, window: &window_utils::SubWindow, buffer: &mut Buffer<Rc<Window>, Rc<Window>>, window_width: usize, colour: u32) {
        for uv in self.series_fill_screen() {
            let buffer_pos = window.from_uv(uv.0, uv.1);
            if window.contains_pixel(buffer_pos.0, buffer_pos.1) {
                // render pixel
                let i = buffer_pos.0 + buffer_pos.1 * window_width as isize;
                if i >= 0 && i < buffer.len() as isize {
                    buffer[i as usize] = colour;
                }
            }
        }
    }
}

fn main() {
    let mut simulation_window = window_utils::SubWindow::new(0, 0, 100, 100);
    let mut simulation_window2 = window_utils::SubWindow::new(0, 0, 100, 100);

    // let solver = Arc::new(Mutex::new(solvers::ias::ParticleIAS::default()));
    let solver = Arc::new(Mutex::new(solvers::iaswedge::ParticleIAS::default()));
    // let solver = Arc::new(Mutex::new(solvers::euclideansheet::ParticleEuclidean::default()));
    let mut lock = solver.lock().unwrap();
    lock.system.set_wedge_num(0.5, 1);
    // lock.system.add_particle(integrator::particle::Particle::new(1., [1.,1.,4.], [0.,0.,0.]));
    // lock.system.add_particle(integrator::particle::Particle::new(2., [2.,2.,4.], [0.,0.,0.]));
    // lock.system.add_particle(integrator::particle::Particle::new(3., [3.,3.,4.], [0.,0.,0.]));
    // lock.system.add_particle(integrator::particle::Particle::new(4., [4.,4.,4.], [0.,0.,0.]));
    lock.system.add_particle(integrator::particle::Particle::new(1., [0.,0.,0.], [0.,0.,0.]));
    // lock.system.add_particle(integrator::particle::Particle::new(0.01, [1.,0.,0.], [0.,1.,0.]));
    // lock.system.add_particle(integrator::particle::Particle::new(0.0, [maths::rotate_xy_accurate(1., 0., 2.*PI/3.).0, maths::rotate_xy_accurate(1., 0., 2.*PI/3.).1, 0.], [maths::rotate_xy_accurate(0., 1., 2.*PI/3.).0, maths::rotate_xy_accurate(0., 1., 2.*PI/3.).1, 0.]));
    // lock.system.add_particle(integrator::particle::Particle::new(0.0, [maths::rotate_xy_accurate(1., 0., -2.*PI/3.).0, maths::rotate_xy_accurate(1., 0., -2.*PI/3.).1, 0.], [maths::rotate_xy_accurate(0., 1., -2.*PI/3.).0, maths::rotate_xy_accurate(0., 1., -2.*PI/3.).1, 0.]));
    // lock.system.add_particle(integrator::particle::Particle::new(0.01, [2.,0.,0.], [0.,1.,0.]));
    // lock.system.add_particle(integrator::particle::Particle::new(0.01, [1.5,0.,0.], [0.,1.,0.]));
    let p = lock.system.base.particles[0].clone();
    let g = lock.system.base.constants.G;
    // lock.system.add_particle(integrator::particle::Particle::new_from_orbit(0.1, &p, g, 1., 0., 0., 0., 0.));
    for i in 1..100 {
        lock.system.add_particle(integrator::particle::Particle::new_from_orbit(1e-6, &p, g, rand::random_range(0.8e3..1.2e3), rand::random_range(0.0..0.01), rand::random_range(0.0..0.01), rand::random_range(0.0..(2.*PI)), rand::random_range(0.0..(2.*PI))));
    }
    // lock.system.add_particle(integrator::particle::Particle::new_from_orbit(0.1, &p, g, rand::random_range(0.8e3..1.2e3), rand::random_range(0.0..0.01), rand::random_range(0.0..0.01), rand::random_range(0.0..(2.*PI)), rand::random_range(0.0..(2.*PI))));
    lock.system.set_acceleration_compensated();
    // lock.system.to_com();
    lock.system.stabilise_wedge();
    // lock.acceleration_calculation = solvers::ias::AccelerationCalculationMode::Simd;
    // lock.rayon_threads = 4;
    let e0 = lock.system.energy();
    println!("{:?}", e0);
    // lock.stepn(4);
    // println!("{:?}", lock.system.base);
    // lock.stepn(100000);
    // lock.set_delta_t(2.*std::f64::consts::PI / 1_0_000.);
    drop(lock);
    // let solver2 = Arc::new(Mutex::new(solvers::iasv1::ParticleIAS::default()));
    // let mut lock = solver2.lock().unwrap();
    // lock.system.add_particle(integrator::particle::Particle::new(1., [0.,0.,0.], [0.,0.,0.]));
    // lock.system.add_particle(integrator::particle::Particle::new(0.01, [1.,0.,0.], [0.,1.,0.]));
    // // lock.system.add_particle(integrator::particle::Particle::new(0.01, [2.,0.,0.], [0.,1.,0.]));
    // // lock.system.add_particle(integrator::particle::Particle::new(0.01, [1.5,0.,0.], [0.,1.,0.]));
    // // for i in 1..20 {
    // //     lock.system.add_particle(integrator::particle::Particle::new(0.01, [1./i as f64, 1., 1.], [0., 0.2, 0.]));
    // // }
    // lock.system.to_com();
    // lock.stepn(100000);
    // drop(lock);

    let mut camera = camera::OrthographicOrbitalCamera::new(
        maths::V3 {x: 0., y: 0., z: 5.}, 
        5.,
        0.001,
        0.1,
    );
    let mut left_mouse_state = ElementState::Released;
    let mut test_graph = graph::PointGraph {values: vec![], max_x: None, max_y: None, min_x: None, min_y: None};
    let mult = 1.01;
    let mut next_render = 1.;

    let refresh_rate = Duration::from_secs_f64(1./2.);
    let event_loop = EventLoop::new().unwrap();
    let mut modifiers = ModifiersState::empty();

    let mut app = winit_app::WinitAppBuilder::with_init(|elwt| {
        let window = {
            let window = elwt.create_window(Window::default_attributes().with_inner_size(PhysicalSize::new(500,500)));
            Rc::new(window.unwrap())
        };
        let context = softbuffer::Context::new(window.clone()).unwrap();
        let surface = softbuffer::Surface::new(&context, window.clone()).unwrap();

        (window, surface)
    }).with_event_handler(|state, event, elwt| {
        let (window, surface) = state;
        // elwt.set_control_flow(ControlFlow::Wait); // would ideally do this and have simulation thread send redraw requests
        elwt.set_control_flow(ControlFlow::wait_duration(refresh_rate)); // doesn't appear to do anything

        match event {
            // Keyboard input
            Event::AboutToWait => {
                // if simulationWindowsimulationController.lock().unwrap().redraw {
                //     simulationWindowsimulationController.lock().unwrap().redraw = false;
                    window.request_redraw();
                // }
            }
            Event::WindowEvent { event: WindowEvent::ModifiersChanged(new), window_id } if window_id == window.id() => {
                modifiers = new.state();
            }
            Event::WindowEvent { event: WindowEvent::KeyboardInput { event, .. }, window_id } if window_id == window.id() => {
                if event.state == ElementState::Pressed && !event.repeat {
                    // pause/unpause simulation
                    // if event.key_without_modifiers().as_ref() == Key::Character("p") {
                    //     let alreadyPaused = simulationWindowsimulationController.lock().unwrap().pause; // deadlocks if one-lined (makes sense but thought maybe the compiler would fix it)
                    //     simulationWindowsimulationController.lock().unwrap().pause = !alreadyPaused;
                    //     println!("Paused: {:?}", !alreadyPaused);
                    // }

                    // // (pause and) iterate simulation
                    // if event.key_without_modifiers().as_ref() == Key::Character("[") {
                    //     simulationWindowsimulationController.lock().unwrap().pause = true;
                    //     simulationWindowsimulationController.lock().unwrap().iterate= true;
                    // }
                }
            },
            // Mouse button on window
            Event::WindowEvent { event: WindowEvent::MouseInput { button, state, .. }, .. } => {
                match button {
                    MouseButton::Left => left_mouse_state = state,
                    _ => ()
                }
            },
            // Mouse button anywhere
            Event::DeviceEvent { event: DeviceEvent::Button { button, .. }, .. } => {
                if button == 0 { // left assumedly at least most of the time
                    // always release if not on window
                    left_mouse_state = ElementState::Released;
                }
            },
            // Mouse move
            Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } => {
                if left_mouse_state.is_pressed() {
                    camera.mouse_rotate(delta.0, delta.1);
                }
            },
            // Scroll wheel
            Event::DeviceEvent { event: DeviceEvent::MouseWheel { delta }, .. } => {
                match delta {
                    MouseScrollDelta::LineDelta(.., y) => {
                        camera.scroll_zoom(y as f64);
                    }
                    _ => {}
                }
            },
            // Redraw
            Event::WindowEvent { window_id, event: WindowEvent::RedrawRequested } if window_id == window.id() => {
                let (width, height) = {
                    let size = window.inner_size();
                    (size.width, size.height)
                };
                surface
                    .resize(
                        NonZeroU32::new(width).unwrap(),
                        NonZeroU32::new(height).unwrap(),
                    )
                    .unwrap();
                let mut buffer = surface.buffer_mut().unwrap();
                buffer[..].fill(0); // clear screen
                
                // simulationWindow
                simulation_window.set_rect(0, 0, width as isize, height as isize);
                // simulation_window2.set_rect(width as isize, height as isize, 0, 0); // sadge can't do inverted windows
                if solver.lock().unwrap().steps as f64 > next_render {
                    test_graph.values.push(((solver.lock().unwrap().system.energy()/e0 - 1.).abs(), 0.));
                    next_render *= mult;
                    println!("{:?}", (solver.lock().unwrap().system.energy()/e0 - 1.).abs());
                    println!("{:?}", solver.lock().unwrap().counters);
                    println!("{:?}", solver.lock().unwrap().delta_t);
                    // println!("{:?}", solver.lock().unwrap().time);
                }
                // test_graph.values.push(((solver2.lock().unwrap().system.energy()/e0 - 1.).abs()+3e-15, 0.));
                // println!("{:?}", (solver.lock().unwrap().system.energy()/e0 - 1.).abs());
                // println!("{:?}", (solver2.lock().unwrap().system.energy()/e0 - 1.).abs());
                test_graph.render(&camera, &simulation_window, &mut buffer, width as usize, 0x00FF00);
                solver.lock().unwrap().system.render(&camera, &simulation_window, &mut buffer, width as usize, 0x0000FF);
                // solver2.lock().unwrap().system.render(&camera, &simulation_window, &mut buffer, width as usize, 0xFF0000);
                // temp step solver a bit
                solver.lock().unwrap().step();
                // solver.lock().unwrap().step_acc_acc();
                // solver.lock().unwrap().system.to_com();
                // solver2.lock().unwrap().system.to_com(); // chaotic systems be like (to_com is causing discrepancies)
                // solver2.lock().unwrap().step();
                // render to screen
                buffer.present().unwrap();
            },
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                window_id,
            } if window_id == window.id() => {
                elwt.exit();
            },
            Event::UserEvent(()) => {
                window.request_redraw(); // never called for some reason
            },
            _ => {}
        }
    });

    event_loop.run_app(&mut app).unwrap();
}