use crate::maths::{self, V3};

#[derive(Debug, Clone)]
pub struct Particle {
    pub mass: f64,
    pub position: V3, // silly me is considering generalising the dimensions but apparently ('proper') gravity gets complicated
    pub velocity: V3,
    pub acceleration: V3
}

impl Particle {
    /// Creates a particle with all parameters set to 0.
    pub fn default() -> Particle {
        Particle { mass: 0., position: [0.; 3].into(), velocity: [0.; 3].into(), acceleration: [0.; 3].into() }
    }
    /// Creates a particle with a given mass, position and velocity.
    pub fn new<T>(mass: f64, position: T, velocity: T) -> Particle 
    where T: Into<V3> {
        Particle { mass, position: position.into(), velocity: velocity.into(), acceleration: [0.; 3].into() }
    }
    /// Creates a particle with a given mass, position, velocity and acceleration.
    pub fn new_with_acceleration<T>(mass: f64, position: T, velocity: T, acceleration: T) -> Particle 
        where T: Into<V3> {
        Particle { mass, position: position.into(), velocity: velocity.into(), acceleration: acceleration.into() }
    }
    /// Creates a particle from orbital parameters. The 'primary' is the target of the new particle's orbit.
    /// 
    /// TODO: limits on and explanations of orbital parameters
    #[allow(non_snake_case)]
    pub fn new_from_orbit(mass: f64, primary: &Particle, G: f64, semi_major: f64, eccentricity: f64, inclination: f64, true_anomaly: f64, pericenter_argument: f64) -> Particle {

        let a = semi_major;
        let e = eccentricity;
        let f = true_anomaly;
        let inc = inclination;
        let m = mass;
        let Omega: f64 = 0.; // longitude of ascending node
        let omega = pericenter_argument;
        
        // from rebound // lacking feasibility checks
        let mut p = Particle::default();
        p.mass = m;
        let r = a*(1.-e*e)/(1. + e*f.cos());
        let v0 = (G*(m+primary.mass)/a/(1.-e*e)).sqrt(); // in this form it works for elliptical and hyperbolic orbits

        let cO = Omega.cos();
        let sO = Omega.sin();
        let co = omega.cos();
        let so = omega.sin();
        let cf = f.cos();
        let sf = f.sin();
        let ci = inc.cos();
        let si = inc.sin();

        // Murray & Dermott Eq 2.122
        p.position.x = primary.position.x + r*(cO*(co*cf-so*sf) - sO*(so*cf+co*sf)*ci);
        p.position.y = primary.position.y + r*(sO*(co*cf-so*sf) + cO*(so*cf+co*sf)*ci);
        p.position.z = primary.position.z + r*(so*cf+co*sf)*si;

        // Murray & Dermott Eq. 2.36 after applying the 3 rotation matrices from Sec. 2.8 to the velocities in the orbital plane
        p.velocity.x = primary.velocity.x + v0*((e+cf)*(-ci*co*sO - cO*so) - sf*(co*cO - ci*so*sO));
        p.velocity.y = primary.velocity.y + v0*((e+cf)*(ci*co*cO - sO*so)  - sf*(co*sO + ci*so*cO));
        p.velocity.z = primary.velocity.z + v0*((e+cf)*co*si - sf*si*so);

        // acceleration 0 by default

        return p;
    }

    /// Vector to another particle.
    pub fn position_delta(&self, p2: &Particle) -> V3 {
        p2.position - self.position
    }
}