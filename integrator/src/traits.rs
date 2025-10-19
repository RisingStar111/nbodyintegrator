use crate::particle::Particle;

/// Collection of particles and expected functionality, with implementations having different internal data structures or additional guarantees/dynamics.
/// Wedge and Sheet in particular have complex acceleration and energy functions.
/// 
/// Generally, a specific System should be used, and constructed via its ::default().
pub trait System {
    fn add_particle(&mut self, particle: Particle);
    fn remove_particle(&mut self, index: usize) -> Particle;
    fn num_particles(&self) -> usize;
    fn get_particle_ref(&self, index: usize) -> &Particle;
    fn set_particle(&mut self, index: usize, new: Particle);
    fn to_euclidean(&mut self);
    /// Sum of all real particle's mass.
    fn total_mass(&self) -> f64;
    /// Move the System to its center-of-mass frame.
    /// 
    /// Implementation may use compensated summation, diverging from REBOUND.
    fn to_com(&mut self);
    /// Set all particle's acceleration.
    fn set_acceleration(&mut self);
    /// Higher accuracy acceleration and returns the final errors.
    fn set_acceleration_compensated(&mut self) -> Vec<crate::maths::V3>;
    /// Total energy (kinetic + gravitational potential) of the system.
    /// 
    /// Implementations may have additional contributions.
    fn energy(&self) -> f64;
}

/// Implements the integration logic over a System.
/// 
/// Generally, a specific Solver should be used, and constructed via its ::default().
pub trait Solver {
    /// Integrate a single step.
    fn step(&mut self);
    /// Integrate multiple steps at once. This may be more efficient than step-ing multiple times.
    fn stepn(&mut self, n: usize); // this may be more complex than step() n times due to loop offsetting
    /// Integrate for the given length of time.
    fn stept(&mut self, t: f64);
    /// Integrate until the Solver's internal time is at least t.
    fn step_untilt(&mut self, t: f64);
    fn write_to(&self, writer: &mut std::io::BufWriter<dyn std::io::Write>);
}

macro_rules! impl_step_untilt { // mostly just to try using a macro; this isn't a great place for it
    () => {
        fn step_untilt(&mut self, t:f64) {
            assert!(self.time < t);
            self.stept(t - self.time);
        }
    };
}