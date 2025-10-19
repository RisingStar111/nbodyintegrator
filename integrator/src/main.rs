#![feature(portable_simd)]
use std::{arch::x86_64::_SIDD_UNIT_MASK, time::Instant};

use crate::{archiving::bufferwriter_from_file, particle::Particle, solvers::euclidean::ParticleEuclidean, traits::*};

#[macro_use]
mod traits;
mod particle;
mod solvers;
mod systems;
mod maths;
mod archiving;

enum ArchivingTimestep {
    Inactive,
    Linear,
    Logarithmic
}
/// Handling for archiving.
pub struct Manager {
    // logging: bool,
    // archiving: ArchivingTimestep,
    // target_archive_num: usize,
    // target_time: f64,
    // current_archive_num: usize,
}

impl Manager {
    // fn new(archiving: ArchivingTimestep) -> Self {
    //     Manager { logging: false, archiving, target_archive_num: 0, target_time: 0., current_archive_num: 0}
    // }

    /// Integrates from current timestep as though it were 0. Ignores previous archive end time (use integrate_resume in those cases)
    fn integrate<T>(solver: &mut T, archiving: ArchivingTimestep, target_archives: usize, clear_old: bool, archive_path: &str, duration: f64) where T: Solver {
        let mut current_archives = 0;
        let mut last_timestep = 0.;

        // load file, clearing it if desired
        let writer;
        if clear_old {
            writer = bufferwriter_from_file(archive_path, false);
        } else {
            // for resuming variant // current_archives = Manager::count_archives(&mut bufferreader_from_file(archive_path).unwrap());
            writer = bufferwriter_from_file(archive_path, true);
        }

        // calculate next archive time
        let next_timestep = match archiving {
            ArchivingTimestep::Inactive => duration,
            ArchivingTimestep::Linear => duration * (current_archives + 1) as f64 / target_archives as f64,
            ArchivingTimestep::Logarithmic => {duration.powf((current_archives + 1) as f64 / target_archives as f64)}
        };

        // step solver
        solver.stept(next_timestep - last_timestep);
        last_timestep = next_timestep;

        // archive
        solver.write_to(&mut writer.unwrap());
        current_archives += 1;
    }

    pub fn count_archives(reader: &mut std::io::BufReader<std::fs::File>) -> usize {
        let mut archive_num = 0;
        while let Some((archivable, _)) = archiving::read_component(reader) {
            match archivable {
                archiving::Archivable::ParticleSystem => archive_num += 1,
            }
        }
        archive_num
    }
}

// varpro crate for fitting probably

fn main() {
    let mut two_body_sim = ParticleEuclidean::default();
    // let mut two_body_sim = ArrayEuclidean::default();
    two_body_sim.system.add_particle(Particle::new(1., [0.,0.,0.], [0.,0.,0.]));
    
    for i in 2..1000 {
        
        two_body_sim.system.add_particle(Particle::new(0., [i as f64,0.,0.], [0.,1.,0.]));
    }
    two_body_sim.system.add_particle(Particle::new(0., [1.,0.,0.], [0.,1.,0.]));
    two_body_sim.system.to_com();
    two_body_sim.set_delta_t(2.*std::f64::consts::PI / 1__000.);
    // println!("{:?}", two_body_sim.system);
    let start = Instant::now();
    println!("{:?}", two_body_sim.system.particles[1]);
    two_body_sim.stepn(1__000);
    println!("{:?}", start.elapsed());
    println!("{:?}", two_body_sim.system.particles[999]);
}
