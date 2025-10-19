use std::f64::consts::PI;

use crate::{maths::{rotate_xy_accurate, V3}, particle::Particle, traits::System};

/// SI values for various physical constants
pub trait Constants { // in SI
    /// Astronomical unit. Distance from Earth to the Sun.
    const AU: f64 = 1.495978707e11;
    /// Gravitational constant.
    const G: f64 = 6.6743e-11;
}

/// Stores base values for dimensions, and constants using those units.
#[derive(Debug)]
pub struct SystemConstants {
    pub G: f64,
    pub units_mass: f64,
    pub units_time: f64,
    pub units_length: f64,
}

impl SystemConstants {
    fn default() -> Self {
        SystemConstants { G: 1., units_mass: 1., units_time: 1., units_length: 1. }
    }
    pub fn to_le_bytes(&self) -> Vec<u8> {
        [self.G.to_le_bytes(), self.units_mass.to_le_bytes(), self.units_time.to_le_bytes(), self.units_length.to_le_bytes()].concat()
    }
}

/// Basic system using an array of particles, as opposed to ArraySystem, which uses arrays of particle properties.
#[derive(Debug)]
pub struct ParticleSystem {
    pub particles: Vec<Particle>,
    pub constants: SystemConstants
}

impl ParticleSystem {
    pub fn default() -> Self {
        ParticleSystem { particles: vec![], constants: SystemConstants::default() }
    }
}

impl System for ParticleSystem {
    fn add_particle(&mut self, particle: Particle) {
        self.particles.push(particle);
    }
    fn get_particle_ref(&self, index: usize) -> &Particle {
        &self.particles[index]
    }
    fn num_particles(&self) -> usize {
        self.particles.len()
    }
    fn remove_particle(&mut self, index: usize) -> Particle {
        self.particles.remove(index)
    }
    fn set_particle(&mut self, index: usize, new: Particle) {
        self.particles[index] = new;
    }
    fn total_mass(&self) -> f64 { // compensated (is different as expected)
        crate::maths::compensated_sum(self.particles.iter().map(|p| &p.mass).into_iter())
    }
    // fn to_com(&mut self) {
    //     let total_mass = self.total_mass();
    //     let com = self.particles.iter().fold(V3::zero(), |acc, e| 
    //         acc + e.position*e.mass
    //     ) / total_mass;
    //     let cov = self.particles.iter().fold(V3::zero(), |acc, e| 
    //         acc + e.velocity*e.mass
    //     ) / total_mass;
    //     for particle in self.particles.iter_mut() {
    //         particle.position = particle.position - com;
    //         particle.velocity = particle.velocity - cov;
    //     }
    // }
    /// Move the System to its center-of-mass frame.
    /// 
    /// This system uses compensated summation, diverging from REBOUND.
    fn to_com(&mut self) { // compensated (more stable again who could have guessed) // ~~rebound appears to take the worst possible approach in terms of error~~
        let total_mass = self.total_mass();
        let com = self.particles.iter().fold((V3::zero(), V3::zero()), |(acc, err), e| 
            {let x = e.position*e.mass - err;
            let t = acc + x;
            (t, (t - acc) - x)}
        ).0 / total_mass;
        let cov = self.particles.iter().fold((V3::zero(), V3::zero()), |(acc, err), e| 
            {let x = e.velocity*e.mass - err;
            let t = acc + x;
            (t, (t - acc) - x)}
        ).0 / total_mass;
        for particle in self.particles.iter_mut() {
            particle.position = particle.position - com;
            particle.velocity = particle.velocity - cov;
        }
    }
    fn set_acceleration(&mut self) {
        for i in 0..self.num_particles() {
            self.particles[i].acceleration = V3::zero();
        }
        for i in 0..self.num_particles()-1 { // more precise than directly adjusting the vel anyway, tho slower
            for j in i+1..self.num_particles() {
                let delta = self.particles[j].position - self.particles[i].position;
                let sq_mag = delta.sq_mag();
                let a_mag = self.constants.G / (sq_mag * sq_mag.sqrt());
                self.particles[i].acceleration = self.particles[i].acceleration + delta * (a_mag * self.particles[j].mass);
                self.particles[j].acceleration = self.particles[j].acceleration + delta * (a_mag * -self.particles[i].mass);
            }
        }
    }
    fn set_acceleration_compensated(&mut self) -> Vec<crate::maths::V3> {
        let num_particles = self.num_particles();
        let mut error = vec![V3::zero(); num_particles]; // inb4 v3 isn't getting 0 cost abstracted
        for i in 0..num_particles {
            self.particles[i].acceleration = V3::zero();
        }
        for i in 0..num_particles-1 { // more precise than directly adjusting the vel anyway, tho slower
            let mut pia = self.particles[i].acceleration;
            let pip = self.particles[i].position;
            let npim = -self.particles[i].mass;
            let mut pie = error[i];
            for j in i+1..num_particles {
                let delta = self.particles[j].position - pip; // a-b == -(b-a) which is good to know, so no extra error from that ~~(or the test compiled it out)~~ - the full expression also seems consistent regardless of where the - is, tho spooky (not that it would cause 'more' error, just 'different' error, if anything)
                let sq_mag = delta.sq_mag();
                let a_mag = self.constants.G / (sq_mag * sq_mag.sqrt());
                let id = delta * (a_mag * self.particles[j].mass) - pie;
                let jd = delta * (a_mag * npim) - error[j];
                let it = pia + id;
                let jt = self.particles[j].acceleration + jd;
                pie = (it - pia) - id;
                error[j] = (jt - self.particles[j].acceleration) - jd;
                pia = it;
                self.particles[j].acceleration = jt;
            }
            self.particles[i].acceleration = pia;
            error[i] = pie;
        }
        error
    }
    fn to_euclidean(&mut self) {
        unimplemented!();
    }
    fn energy(&self) -> f64 {
        let mut energy = 0.;
        // potential energy
        for i in 0..self.num_particles()-1 {
            for j in (i+1)..self.num_particles() {
                let delta = self.particles[j].position - self.particles[i].position;
                let sq_mag = delta.sq_mag();
                energy += self.constants.G * self.particles[i].mass * self.particles[j].mass / sq_mag.sqrt();
            }
        }
        energy *= -1.; // don't double count potentials and make attractive // i don't think there's anything wrong with this? (not the *0.5 later)
        // kinetic energy
        for i in 0..self.num_particles() {
            energy += self.particles[i].velocity.sq_mag() * self.particles[i].mass * 0.5;
        }
        energy
    }
    // todo orbital parameter initialisations
}

/// Basic system using arrays of particle properties, as opposed to ParticleSystem, which uses an array of particles.
/// 
/// Not very supported, due to the loss in ergonomics and lack of benefit for most computations.
#[derive(Debug)]
pub struct ArraySystem { // semi-tested but unused as conversions aren't costly (and borrow checker gets in they way of nice access from this end)
    pub masses: Vec<f64>,
    pub positions: Vec<V3>,
    pub velocities: Vec<V3>,
    pub constants: SystemConstants
}

impl ArraySystem {
    pub fn default() -> Self {
        ArraySystem { masses: vec![], positions: vec![], velocities: vec![], constants: SystemConstants::default() }
    }
}

impl System for ArraySystem {
    fn add_particle(&mut self, particle: Particle) {
        self.masses.push(particle.mass);
        self.positions.push(particle.position);
        self.velocities.push(particle.velocity);
    }
    fn get_particle_ref(&self, _: usize) -> &Particle {
        unimplemented!("can't get a reference to a temporary value")
        // &Particle::new(self.masses[index], self.positions[index], self.velocities[index])
    }
    fn num_particles(&self) -> usize {
        self.masses.len()
    }
    fn remove_particle(&mut self, index: usize) -> Particle {
        Particle::new(
            self.masses.remove(index),
            self.positions.remove(index),
            self.velocities.remove(index),
        )
    }
    fn set_particle(&mut self, index: usize, new: Particle) {
        self.masses[index] = new.mass;
        self.positions[index] = new.position;
        self.velocities[index] = new.velocity;
    }
    fn total_mass(&self) -> f64 {
        self.masses.iter().sum()
    }
    fn to_com(&mut self){
        let total_mass = self.total_mass();
        let mut com = V3::zero();
        let mut cov = V3::zero();
        for ((mass, position), velocity) in self.masses.iter().zip(self.positions.iter()).zip(self.velocities.iter()) {
            com = com + *position**mass;
            cov = cov + *velocity**mass;
        }
        com = com / total_mass;
        cov = cov / total_mass;
        for position in self.positions.iter_mut() {
            *position = *position - com;
        }
        for velocity in self.velocities.iter_mut() {
            *velocity = *velocity - cov;
        }
    }
    fn set_acceleration(&mut self) {
        unimplemented!()
    }
    fn set_acceleration_compensated(&mut self) -> Vec<crate::maths::V3> {
        unimplemented!()
    }
    fn to_euclidean(&mut self) {
        unimplemented!()
    }
    fn energy(&self) -> f64 {
        unimplemented!()
    }
}

/// System that directly reduces the domain to a wedge/segment centered around the primary (0th) particle.
///
/// Rotation (and thus assumed system orientation) is around the Z axis
#[derive(Debug)]
pub struct Wedge {
    pub base: ParticleSystem,
    pub angle: f64,
    pub ghost_wedges: usize, // how many either side, so 2x ghosts sections simulated
}

impl Wedge {
    pub fn default() -> Self {
        Wedge { base: ParticleSystem::default(), angle: 2.*PI/3., ghost_wedges: 1}
    }

    pub fn set_wedge_num(&mut self, angle: f64, ghost: usize) {
        self.angle = angle;
        self.ghost_wedges = ghost;
    }

    fn acceleration_from_source(&self, particle: V3, source: V3, source_mass: f64) -> V3 {
        let delta = source - particle;
        let sq_mag = delta.sq_mag();
        let a_mag = self.base.constants.G / (sq_mag * sq_mag.sqrt());
        delta * (a_mag * source_mass)
    }

    pub fn lock_to_wedge(&mut self) {
        let zeropos = self.base.particles[0].position;
        for p in &mut self.base.particles {
            p.position = p.position - zeropos;
            let (newx, newy) = rotate_xy_accurate(p.position.x, p.position.y, -self.angle);
            if newy >= 0. {
                p.position = [newx, newy, p.position.z].into();
                let (newvx, newvy) = rotate_xy_accurate(p.velocity.x, p.velocity.y, -self.angle);
                p.velocity = [newvx, newvy, p.velocity.z].into();
            }
            else if p.position.y < 0. {
                let (newx, newy) = rotate_xy_accurate(p.position.x, p.position.y, self.angle);
                p.position = [newx, newy, p.position.z].into();
                let (newvx, newvy) = rotate_xy_accurate(p.velocity.x, p.velocity.y, self.angle);
                p.velocity = [newvx, newvy, p.velocity.z].into();
            }
            p.position = p.position + zeropos;
        }
    }

    pub fn stabilise_wedge(&mut self) {
        for _ in 0..((2.*PI/self.angle).ceil() as usize) {
            self.lock_to_wedge();
        }
    }
}

impl System for Wedge {
    fn add_particle(&mut self, particle: Particle) {
        self.base.add_particle(particle);
    }
    fn get_particle_ref(&self, index: usize) -> &Particle {
        self.base.get_particle_ref(index)
    }
    fn num_particles(&self) -> usize {
        self.base.num_particles()
    }
    fn remove_particle(&mut self, index: usize) -> Particle {
        self.base.remove_particle(index)
    }
    fn set_particle(&mut self, index: usize, new: Particle) {
        self.base.set_particle(index, new);
    }
    fn total_mass(&self) -> f64 {
        self.base.total_mass()
    }
    fn to_com(&mut self) {
        self.base.to_com();
    }
    fn set_acceleration(&mut self) {
        unimplemented!()
        // rebound for their sheet does an interesting trick of having the force felt by ghost x from real y == force felt by real y from ghost x on the opposite box, and particularly, doing n^2/2 iterations, now to also avoid doublecounting (for compensated, this compounds error due to rotation so prolly don't use there)
        // not used in rebound however, for the rectangular boxes, (for non-error strict integration) can work out r_g^2 to a ghost particle with r_r^2+(ghost box offset vector)dot(ghost box offset vector + 2*real particle pos vector)
        // which is 3 * (addmul + mul) + add as well as not regoing over position memory (but do have to reget box offsets?)
        // gpt suggests r^2 + 2*p.g + g.g instead as 2g and g.g can be stored in a lookup
        // in place of 3*(add+sub)+2*addmul+mul ? // chatgpt aided analysis suggests this would be equal/slower ignoring mem costs, because you still need the delta to ghost
        // can reduce the add+sub in above by looping over ghosts on the inner since you already have r2-r1
        // could take out G (tho for compensated, messes with the second order error) // would this even be better since G is the target of a divide?
    }
    fn set_acceleration_compensated(&mut self) -> Vec<crate::maths::V3> {
        let mut error = vec![V3::zero(); self.num_particles()];
        for i in 0..self.base.num_particles() {
            self.base.particles[i].acceleration = V3::zero();
        }
        // calculate all ghost positions
        let mut particles = vec![];
        for i in 0..self.base.num_particles() {
            // real
            particles.push(self.base.particles[i].position);

            for k in 1..self.ghost_wedges+1 { // errors in rotation leads to (slight) inhomogeneity hmm (aka should this also be compensated)
                // anticlockwise
                let oldpos = self.base.particles[i].position;
                self.base.particles[i].position = self.base.particles[i].position - self.base.particles[0].position;
                particles.push(self.base.particles[i].position.rotated_accurate(k as f64 * self.angle));
                // clockwise
                particles.push(self.base.particles[i].position.rotated_accurate(-(k as f64) * self.angle));
                self.base.particles[i].position = oldpos;
            }
        }
        for i in 0..self.base.num_particles() { // iter all particles to get self ghosts gravity (inner loop is empty on last one)
            let primary_real_index = i * (2*self.ghost_wedges+1);
            // calculate force from own ghosts on original particle
            if i != 0 { // maybe not if the central body
                for k in 1..2*self.ghost_wedges+1 {
                    let id = self.acceleration_from_source(particles[primary_real_index], particles[primary_real_index + k], self.base.particles[i].mass) - error[i];
                    let it = self.base.particles[i].acceleration + id;
                    error[i] = (it - self.base.particles[i].acceleration) - id;
                    self.base.particles[i].acceleration = it;
                }
            }
            for j in i+1..self.base.num_particles() { // likely inefficient ordering (not interwoven) and doubt compiler can do well alone
                let secondary_real_index = j * (2*self.ghost_wedges+1);
                // forces between real pair
                // (this one in particular probably kept as delta is used twice (tho far less of an optimisation (it may even be possible when optimised that not coupling them is faster)))
                let delta = self.base.particles[j].position - self.base.particles[i].position; // a-b == -(b-a) which is good to know, so no extra error from that ~~(or the test compiled it out)~~ - the full expression also seems consistent regardless of where the - is, tho spooky (not that it would cause 'more' error, just 'different' error, if anything)
                let sq_mag = delta.sq_mag();
                let a_mag = self.base.constants.G / (sq_mag * sq_mag.sqrt());
                let id = delta * (a_mag * self.base.particles[j].mass) - error[i];
                let jd = delta * (a_mag * -self.base.particles[i].mass) - error[j];
                let it = self.base.particles[i].acceleration + id;
                let jt = self.base.particles[j].acceleration + jd;
                error[i] = (it - self.base.particles[i].acceleration) - id;
                error[j] = (jt - self.base.particles[j].acceleration) - jd;
                self.base.particles[i].acceleration = it;
                self.base.particles[j].acceleration = jt;
                // force from first ghosts on second real
                if i != 0 {
                    for k in 1..2*self.ghost_wedges+1 {
                        let id = self.acceleration_from_source(particles[secondary_real_index], particles[primary_real_index + k], self.base.particles[i].mass) - error[j];
                        let it = self.base.particles[j].acceleration + id;
                        error[i] = (it - self.base.particles[j].acceleration) - id;
                        self.base.particles[j].acceleration = it;
                    }
                }
                // force from second ghosts on first real
                for k in 1..2*self.ghost_wedges+1 {
                    let id = self.acceleration_from_source(particles[primary_real_index], particles[secondary_real_index + k], self.base.particles[j].mass) - error[i];
                    let it = self.base.particles[i].acceleration + id;
                    error[i] = (it - self.base.particles[i].acceleration) - id;
                    self.base.particles[i].acceleration = it;
                }
            }
        }
        error
    }
    fn to_euclidean(&mut self) {
        unimplemented!();
    }
    fn energy(&self) -> f64 {
        self.base.energy()
    }
}