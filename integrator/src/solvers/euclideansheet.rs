use crate::{traits::*, maths::V3, systems};

pub struct ParticleEuclidean {
    pub system: systems::Sheet,
    // other settings/etc
    time: f64,
    delta_t: f64,
}

impl ParticleEuclidean {
    pub fn default() -> Self {
        ParticleEuclidean { 
            system: systems::Sheet::default(),
            time: 0., 
            delta_t: 1. 
        }
    }

    pub fn set_delta_t(&mut self, dt: f64) {
        self.delta_t = dt;
    }
    // testing setups (they all behave differently cuz chaos but don't currently have a way to 'say' which is better)
    pub fn step_acc(&mut self) {
        self.system.set_acceleration();
        // drift
        for particle in &mut self.system.base.particles {
            particle.velocity = particle.velocity + particle.acceleration * self.delta_t;
            particle.position = particle.position + particle.velocity * self.delta_t;
        }
    }
    pub fn step_acc_acc(&mut self) {
        self.system.set_acceleration_compensated();
        // drift
        for particle in &mut self.system.base.particles {
            particle.velocity = particle.velocity + particle.acceleration * self.delta_t;
            particle.position = particle.position + particle.velocity * self.delta_t;
        }
    }
}

impl Solver for ParticleEuclidean {
    fn step(&mut self) {
        // // kick
        // for i in 0..self.system.num_particles()-1 {
        //     for j in i+1..self.system.num_particles() {
        //         let delta = self.system.base.particles[j].position - self.system.base.particles[i].position;
        //         let sq_mag = delta.sq_mag();
        //         let a_mag = self.delta_t * self.system.constants.G / (sq_mag * sq_mag.sqrt());
        //         self.system.base.particles[i].velocity = self.system.base.particles[i].velocity + (delta * (a_mag * self.system.base.particles[j].mass));
        //         self.system.base.particles[j].velocity = self.system.base.particles[j].velocity + (delta * (a_mag * -self.system.base.particles[i].mass));   
        //     }
        // }
        // // drift
        // for particle in &mut self.system.base.particles {
        //     particle.position = particle.position + particle.velocity * self.delta_t;
        // }
        // kick
        for i in 0..self.system.num_particles()-1 {
            let mut ex = 0.;
            let mut ey = 0.;
            let mut ez = 0.; // i forgor if this was better or not
            for j in i+1..self.system.num_particles() {
                let delta = self.system.base.particles[j].position - self.system.base.particles[i].position;
                let sq_mag = delta.sq_mag();
                let a_mag = self.delta_t * self.system.base.constants.G / (sq_mag * sq_mag.sqrt());
                let g = crate::maths::fast_2sum(self.system.base.particles[i].velocity.x, (delta * (a_mag * self.system.base.particles[j].mass)).x + ex);
                self.system.base.particles[i].velocity.x = g.0;
                ex = g.1;
                // if i == 1 && ex != 0. {println!("{:?}", ex)}
                let g = crate::maths::fast_2sum(self.system.base.particles[i].velocity.y, (delta * (a_mag * self.system.base.particles[j].mass)).y + ey);
                self.system.base.particles[i].velocity.y = g.0;
                ey = g.1;
                let g = crate::maths::fast_2sum(self.system.base.particles[i].velocity.z, (delta * (a_mag * self.system.base.particles[j].mass)).z + ez);
                self.system.base.particles[i].velocity.z = g.0;
                ez = g.1;
                self.system.base.particles[j].velocity = self.system.base.particles[j].velocity + (delta * (a_mag * -self.system.base.particles[i].mass));   
            }
        }
        // drift
        for particle in &mut self.system.base.particles {
            particle.position = particle.position + particle.velocity * self.delta_t;
        }
    }
    fn stepn(&mut self, n: usize) {
        (0..n).for_each(|_| self.step());
    }
    fn stept(&mut self, t: f64) {
        self.stepn(t as usize);
    }
    impl_step_untilt!();
    fn write_to(&self, writer: &mut std::io::BufWriter<dyn std::io::Write>) {
        unimplemented!()
        // self.system.write_to(writer);
    }
}

pub struct ArrayEuclidean {
    pub system: systems::ArraySystem,
    // other settings/etc
    time: f64,
    delta_t: f64,
}

impl ArrayEuclidean {
    pub fn default() -> Self {
        ArrayEuclidean { 
            system: systems::ArraySystem::default(),
            time: 0., 
            delta_t: 1. 
        }
    }

    pub fn set_delta_t(&mut self, dt: f64) {
        self.delta_t = dt;
    }
}

impl Solver for ArrayEuclidean {
    fn step(&mut self) {
        // kick
        for i in 0..self.system.num_particles()-1 {
            for j in i+1..self.system.num_particles() {
                let delta = self.system.positions[j] - self.system.positions[i];
                let sq_mag = delta.sq_mag();
                let a_mag = self.delta_t * self.system.constants.G / (sq_mag * sq_mag.sqrt());
                self.system.velocities[i] = self.system.velocities[i] + (delta * (a_mag * self.system.masses[j]));
                self.system.velocities[j] = self.system.velocities[j] + (delta * (a_mag * -self.system.masses[i]));
            }
        }
        // drift
        for (position, velocity) in self.system.positions.iter_mut().zip(self.system.velocities.iter()) {
            *position = *position + *velocity * self.delta_t;
        }
    }
    fn stepn(&mut self, n: usize) {
        (0..n).for_each(|_| self.step());
    }
    fn stept(&mut self, t: f64) {
        self.stepn(t as usize);
    }
    impl_step_untilt!();
    fn write_to(&self, writer: &mut std::io::BufWriter<dyn std::io::Write>) {
        unimplemented!()
    }
}