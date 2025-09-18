#![feature(portable_simd)]
#![feature(f128)]
use std::{simd::{f64x4, StdFloat}, time::Instant};
mod maths;
use integrator::{particle::Particle, solvers, traits::{Solver, System}};
use maths::*;

struct Part {
    pos: f64,
    vel: f64,
    acc: f64,
}


#[inline(never)]
fn add_sqrt(parts: &[Part]) -> Vec<Part> {
    parts.iter().map(|p| Part {pos: p.pos + p.pos.sqrt(), vel: p.vel + p.vel.sqrt(), acc: p.acc + p.acc.sqrt()}).collect()
}
#[inline(never)]
fn add_sqrt2(pos: &mut [f64x4], vel: &mut [f64x4], acc: &mut [f64x4]) {
    // assert_eq!(pos.len(), vel.len());
    // assert_eq!(pos.len(), acc.len());
    for i in 0..pos.len() {
        pos[i] += pos[i].sqrt();
        vel[i] += vel[i].sqrt();
        acc[i] += acc[i].sqrt();
    }
}

const N: usize = 100_000;
const I: usize = 1_000;
const R: f64 = 1e9;

fn main() {
    // init
    // let mut parts = vec![];
    // for _ in 0..N {
    //     parts.push(Part {pos: rand::random_range(0.0..R), vel: rand::random_range(0.0..R), acc: rand::random_range(0.0..R)});
    // }
    // let mut pos = vec![];
    // let mut vel = vec![];
    // let mut acc = vec![];
    // for _ in 0..N/4 {
    //     pos.push(f64x4::from_array([rand::random_range(0.0..R),rand::random_range(0.0..R),rand::random_range(0.0..R),rand::random_range(0.0..R)]));
    //     vel.push(f64x4::from_array([rand::random_range(0.0..R),rand::random_range(0.0..R),rand::random_range(0.0..R),rand::random_range(0.0..R)]));
    //     acc.push(f64x4::from_array([rand::random_range(0.0..R),rand::random_range(0.0..R),rand::random_range(0.0..R),rand::random_range(0.0..R)]));
    // }

    // // warmup
    // for _ in 0..10 {
    //     add_sqrt(&mut parts);
    // }

    // let now = Instant::now();
    // for _ in 0..I {
    //     parts = add_sqrt(&mut parts);
    // }
    // println!("{} particles over {} iterations took {:?}", N, I, now.elapsed());
    // let now = Instant::now();
    // for _ in 0..I {
    //     add_sqrt2(&mut pos, &mut vel, &mut acc);
    // }
    // println!("{} particles over {} iterations took {:?}", N, I, now.elapsed());

    // let (mut x, mut y) = (1., 0.);
    // println!("{}, {}", x, y);
    // let n = 1224625;
    // let now = Instant::now();
    // for _ in 0..n/4 {
    //     (x, y) = rotate_xy(x, y, 2.*std::f64::consts::PI / (n as f64));
    // }
    // println!("{}, {}", x, y);
    // println!("{}", x*x + y*y);
    // println!("{:?}", now.elapsed());
    // let (mut x, mut y) = (1., 0.);
    // println!("{}, {}", x, y);
    // let now = Instant::now();
    // for _ in 0..n/4 {
    //     (x, y) = rotate_xy_accurate(x, y, 2.*std::f64::consts::PI / (n as f64));
    // }
    // println!("{}, {}", x, y);
    // println!("{}", x*x + y*y);
    // println!("{:?}", now.elapsed());

    // testing if precomputed divides are actually worse (hopefully, testing is hard)
    // well precompute does often give a different answer, including not 1.0 for x/x
    // precompute 'adds' an intermediate roundoff = bad, however if constants take more than 1 operation (before the divide), precompute those in higher precision (and still divide)
    // tho biased roundoffs from precompute pose issues and are odd to track
    // possibly also could make a 'trial' version where the divides are precomputed out where possible (tho, all of this is O(N) so shouldn't matter much for speed rip)
    // let n = 100000;
    // let mut l = vec![];
    // let mut dd = vec![];
    // let mut d = vec![];
    // for _ in 0..n {
    //     l.push((rand::random_range(0.0..10.) as f64).powf(rand::random_range(1.0..50.)));
    // }
    // for i in 0..n {
    //     dd.push((1./(l[i] as f128) + 2.) as f64);
    //     d.push(1./(l[i]) + 2.);
    // }
    // for i in 0..n {
    //     for j in 0..n {
    //         assert_eq!((l[i] - l[j]) * (l[0] * -d[i]), (l[j] - l[i]) * (l[0] * d[i]), "{}, {}, at {}, {}", l[i] / l[j], l[i] * d[j], l[i], l[j]);
    //     }
    // }

    // basic timings
    let steps = 1000;

    // let pos: Vec<[f64; 3]> = (0..100).map(|n| [rand::random_range(0.8..1.2), 0., 0.]).collect();
    // let vel: Vec<[f64; 3]> = (0..100).map(|n| [0., rand::random_range(0.8..1.2), 0.]).collect();
    let pos: Vec<[f64; 3]> = (0..100).map(|n| [n as f64 + 3., 0., 0.]).collect();
    let vel: Vec<[f64; 3]> = (0..100).map(|n| [0., n as f64 + 0.2, 0.2]).collect();
    let mut sim = solvers::ias::ParticleIAS::default();
    // sim.enable_gpu();
    sim.acceleration_calculation = solvers::ias::AccelerationCalculationMode::Simd;
    sim.system.add_particle(Particle::new(1., [0., 0., 0.], [0., 0., 0.]));
    sim.system.add_particle(Particle::new(0.01, [1., 0., 0.], [0., 1., 0.]));
    sim.system.add_particle(Particle::new(1., [2., 0., 0.], [0., 0.2, 0.2]));
    for i in 0..pos.len() {
        sim.system.add_particle(Particle::new(0.01, pos[i], vel[i]));
    }
    sim.system.to_com();
    let e0 = sim.system.energy();

    let start = Instant::now();
    sim.stepn(steps);
    println!("{:?}", start.elapsed());
    println!("{:?}", sim.counters);
    println!("{:?}", sim.system.particles[0]);
    println!("{:?}", sim.time);
    println!("{:?}", sim.delta_t);
    println!("{:?}", sim.system.energy());
    println!("{:?}", e0);

    let mut sim = solvers::ias::ParticleIAS::default();
    sim.system.add_particle(Particle::new(1., [0., 0., 0.], [0., 0., 0.]));
    sim.system.add_particle(Particle::new(0.01, [1., 0., 0.], [0., 1., 0.]));
    sim.system.add_particle(Particle::new(1., [2., 0., 0.], [0., 0.2, 0.2]));
    for i in 0..pos.len() {
        sim.system.add_particle(Particle::new(0.01, pos[i], vel[i]));
    }
    sim.system.to_com();
    let e0 = sim.system.energy();

    let start = Instant::now();
    sim.stepn(steps);
    println!("{:?}", start.elapsed());
    println!("{:?}", sim.counters);
    println!("{:?}", sim.system.particles[0]);
    println!("{:?}", sim.time);
    println!("{:?}", sim.delta_t); // dt is 1/3rd of rebounds... // dt starts capped min that's not good
    println!("{:?}", sim.system.energy());
    println!("{:?}", e0);
    // v1 to v2 causes a difference, tho still not matching with rebound
}

#[cfg(test)]
mod tests {
    use super::*;
    // possibly use nightly bench tests
}