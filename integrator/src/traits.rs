use crate::particle::Particle;

pub trait Simulation { // not sure if this will be used

}
pub trait Jacobian { // implicitly always working in Euclidean as a baseline
    fn to_euclidean(&mut self);
    fn to_jacobian(&mut self);
}

pub trait System { // put onus of being able to get properties on definition of solver  (eg struct solve1<T> where T: System + HasMass)
    fn add_particle(&mut self, particle: Particle);
    fn remove_particle(&mut self, index: usize) -> Particle;
    fn num_particles(&self) -> usize;
    fn get_particle_ref(&self, index: usize) -> &Particle;
    fn set_particle(&mut self, index: usize, new: Particle);
    fn to_euclidean(&mut self);
    fn total_mass(&self) -> f64;
    fn to_com(&mut self);
    fn set_acceleration(&mut self);
    /// Higher accuracy acceleration and returns the final errors
    fn set_acceleration_compensated(&mut self) -> Vec<crate::maths::V3>;
    fn energy(&self) -> f64;
}

pub trait Solver {
    fn step(&mut self);
    fn stepn(&mut self, n: usize); // this may be more complex than step() n times due to loop offsetting
    fn stept(&mut self, t: f64);
    fn step_untilt(&mut self, t: f64);
    fn write_to(&self, writer: &mut std::io::BufWriter<dyn std::io::Write>);
}

macro_rules! impl_step_untilt {
    () => {
        fn step_untilt(&mut self, t:f64) {
            assert!(self.time < t);
            self.stept(t - self.time);
        }
    };
}