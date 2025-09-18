// Heavily based upon work by rein and tremaine [todo] (a lot of functionality is copied 1 to 1)

use crate::{traits::*, systems};

struct IASCounter {
    pub predictor_iterations: usize,
    pub abort_converged: usize,
    pub abort_oscillation: usize,
    pub abort_slow: usize,
    pub step_failed: usize,
}

impl IASCounter {
    pub fn default() -> Self {
        IASCounter { predictor_iterations: 0, abort_converged: 0, abort_oscillation: 0, abort_slow: 0, step_failed: 0 }
    }
}

#[derive(Debug)]
struct Subspacedarray { // this might help the compiler tbh
    pub p0: Vec<f64>,
    pub p1: Vec<f64>,
    pub p2: Vec<f64>,
    pub p3: Vec<f64>,
    pub p4: Vec<f64>,
    pub p5: Vec<f64>,
    pub p6: Vec<f64>,
}

impl Subspacedarray {
    fn with_length(len: usize) -> Self {
        Subspacedarray { p0: vec![0.; len], p1: vec![0.; len], p2: vec![0.; len], p3: vec![0.; len], p4: vec![0.; len], p5: vec![0.; len], p6: vec![0.; len] }
    }
}

pub struct ParticleIAS {
    pub system: systems::ParticleSystem,
    // other settings/etc
    time: f64,
    pub delta_t: f64,
    velocity_dependant_forces: bool,
    epsilon: f64,

    counters: IASCounter,
    steps: usize,

    // persistent storage for compensated sums - rebound clears the error if a particle is added (but not when removed?) - unsure why
    // (depending on when it's cleared, that could cause issues with manipulating particles, eg if it's not cleared when changing positions)
    // could potentially just store the actual particle properties as 'f128's
    error_x: Vec<f64>,
    error_v: Vec<f64>,
    error_a: Vec<f64>,
}

impl ParticleIAS {
    pub fn default() -> Self {
        ParticleIAS { 
            system: systems::ParticleSystem::default(),
            time: 0.,
            delta_t: 0.1,
            velocity_dependant_forces: false,
            epsilon: 1e-9,
            steps: 0,
            counters: IASCounter::default(),
            error_x: vec![],
            error_v: vec![],
            error_a: vec![],
        }
    }

    fn update_acceleration(&mut self) {
        self.error_a = self.system.set_acceleration_compensated().iter().flat_map(|f| f.array()).collect();
        // gravity idk
        // probably just have a default implementation as part of System but here could compensate the force sum (just, it's second order error so probably worthless)
    }
}

const SUBSPACINGS: [f64; 8] = [0.0, 0.0562625605369221464656521910318, 0.180240691736892364987579942780, 0.352624717113169637373907769648, 0.547153626330555383001448554766, 0.734210177215410531523210605558, 0.885320946839095768090359771030, 0.977520613561287501891174488626];
const RR: [f64; 28] = [0.0562625605369221464656522, 0.1802406917368923649875799, 0.1239781311999702185219278, 0.3526247171131696373739078, 0.2963621565762474909082556, 0.1723840253762772723863278, 0.5471536263305553830014486, 0.4908910657936332365357964, 0.3669129345936630180138686, 0.1945289092173857456275408, 0.7342101772154105315232106, 0.6779476166784883850575584, 0.5539694854785181665356307, 0.3815854601022408941493028, 0.1870565508848551485217621, 0.8853209468390957680903598, 0.8290583863021736216247076, 0.7050802551022034031027798, 0.5326962297259261307164520, 0.3381673205085403850889112, 0.1511107696236852365671492, 0.9775206135612875018911745, 0.9212580530243653554255223, 0.7972799218243951369035945, 0.6248958964481178645172667, 0.4303669872307321188897259, 0.2433104363458769703679639, 0.0921996667221917338008147];
const C: [f64; 21] = [-0.0562625605369221464656522, 0.0101408028300636299864818, -0.2365032522738145114532321, -0.0035758977292516175949345, 0.0935376952594620658957485, -0.5891279693869841488271399, 0.0019565654099472210769006, -0.0547553868890686864408084, 0.4158812000823068616886219, -1.1362815957175395318285885, -0.0014365302363708915424460, 0.0421585277212687077072973, -0.3600995965020568122897665, 1.2501507118406910258505441, -1.8704917729329500633517991, 0.0012717903090268677492943, -0.0387603579159067703699046, 0.3609622434528459832253398, -1.4668842084004269643701553, 2.9061362593084293014237913, -2.7558127197720458314421588];


impl Solver for ParticleIAS {
    fn step(&mut self) {
        self.steps += 1;
        self.update_acceleration();
        // rebound converts AoS to SoA at some point for caching, and tbf most of the time it will be negligible cost
        let num_components = self.system.num_particles() * 3;
        let mut x0 = self.system.particles.iter().flat_map(|p| p.position.array()).collect::<Vec<f64>>();
        let mut v0 = self.system.particles.iter().flat_map(|p| p.velocity.array()).collect::<Vec<f64>>();
        let a0 = self.system.particles.iter().flat_map(|p| p.acceleration.array()).collect::<Vec<f64>>();
        // intitialise the compensated sum errors if not present
        if self.error_x.len() < num_components {
            self.error_x = vec![0.; num_components];
            self.error_v = vec![0.; num_components];
            // self.error_a = vec![0.; num_components];
        }
        let csa0 = self.error_a.clone();
        // until converged
        let mut b = Subspacedarray::with_length(num_components);
        let mut csb = Subspacedarray::with_length(num_components);
        let mut g = Subspacedarray::with_length(num_components); // todo set these up as stored like in rebound
        let mut last_predictor_corrector_error = 2.; // ? rebound sets to 2.
        let mut predictor_corrector_error = f64::MAX;
        let mut iterations = 0;
        loop {
            self.counters.predictor_iterations += 1;
            // break if within tolerance
            if predictor_corrector_error < 1e-16 { self.counters.abort_converged += 1; break }
            // or didn't improve
            if iterations > 2 && last_predictor_corrector_error <= predictor_corrector_error { self.counters.abort_oscillation += 1; break } // not fairly arbitrary apparently
            // or took too long
            if iterations >= 12 { self.counters.abort_slow += 1; break } // fairly arbitrary
            iterations += 1;
            last_predictor_corrector_error = predictor_corrector_error;
            predictor_corrector_error = 0.;
            // for each spacing
            for (step, substep) in SUBSPACINGS[1..].iter().enumerate() { // maybe explictly unroll
                // per particle (for each component)
                for i in 0..self.system.num_particles() {
                    // predict position
                    // delta is from start of full step
                    // rebound docs suggest using x * dt * dt rather than storing dt*dt to randomise roundoff error
                    // in this one tho they chain everything in one equation which as well as using divides i'm sus on error accumulation (tho i think it's 3rd order so eh)
                    // tho i suppose, anything 'costly' here will be dwarfed by the forces calc... (and divide should be more precise?)
                    // also recommends x(p/q) as p*x/q
                    // assign error adjusted position to real particles, for the force calc to use // also why the loop is partially unrolled
                    self.system.particles[i].position = 
                        [3*i, 3*i+1, 3*i+2].map(|k|
                            -self.error_x[k]
                             + ((((((((b.p6[k]*7.*substep/9. + b.p5[k])*3.*substep/4. + b.p4[k])*5.*substep/7. + b.p3[k])*2.*substep/3. + b.p2[k])*3.*substep/5. + b.p1[k])*substep/2. + b.p0[k])*substep/3. + a0[k])*self.delta_t*substep/2. + v0[k])*self.delta_t*substep
                             + x0[k]
                        ).into(); // check emmitted code as may not compile well
                }
                if self.velocity_dependant_forces {
                    for i in 0..num_components {
                        // (predict velocity)
                    }
                }
                // calculate (static) force (at new pos/vel) (each 3 component particle at once ofc)
                self.update_acceleration(); // sets self.error_a
                // calculate corrected 'g' coefficient
                let ac = self.system.particles.iter().flat_map(|p| p.acceleration.array()).collect::<Vec<f64>>();
                // update 'b' coefficient with (delta in) 'g'
                match step {
                    0 => {
                        // rebound compensates a lot of stuff here but then doesn't compensate the initial sum set? (similar deal as with the position sum set) - i suppose as long as the error is small enough it's second order so won't change results (and my current sim setup is bad for that due to low velocity initialisation)
                        // possibly because the goal is the final pos set to be machine precision regardless (tho then, why compensate anything - maybe it speeds up convergence, or limits oscillations)
                        // i spose aught to test as per usual
                        for i in 0..num_components {
                            let mut tmp = g.p0[i];
                            let mut gk = ac[i];
                            let mut gk_cs = self.error_a[i];
                            crate::maths::compensated_add(&mut gk, -a0[i], &mut gk_cs);
                            crate::maths::compensated_add(&mut gk, csa0[i], &mut gk_cs);
                            g.p0[i] = gk/RR[0]; // the chained divides minimise operation count to reduce error (tho still if copmensated it would be lower again)
                            tmp = g.p0[i]-tmp;
                            crate::maths::compensated_add(&mut b.p0[i], tmp, &mut csb.p0[i]);
                        }
                    }
                    1 => {
                        for i in 0..num_components {
                            let mut tmp = g.p1[i];
                            let mut gk = ac[i];
                            let mut gk_cs = self.error_a[i];
                            crate::maths::compensated_add(&mut gk, -a0[i], &mut gk_cs);
                            crate::maths::compensated_add(&mut gk, csa0[i], &mut gk_cs);
                            g.p1[i] = (gk/RR[1] - g.p0[i])/RR[2];
                            tmp = g.p1[i]-tmp; // some odd things you can do with the consts that at 2 or more precomp steps but changes the type of error? (also can remove all but last divide by multiplying straight but that would be same expected error? (+1 divide) as all mult)
                            crate::maths::compensated_add(&mut b.p0[i], tmp * C[0], &mut csb.p0[i]);
                            crate::maths::compensated_add(&mut b.p1[i], tmp, &mut csb.p1[i]);
                        }
                    }
                    2 => {
                        for i in 0..num_components {
                            let mut tmp = g.p2[i];
                            let mut gk = ac[i];
                            let mut gk_cs = self.error_a[i];
                            crate::maths::compensated_add(&mut gk, -a0[i], &mut gk_cs);
                            crate::maths::compensated_add(&mut gk, csa0[i], &mut gk_cs);
                            g.p2[i] = ((gk/RR[3] - g.p0[i])/RR[4] - g.p1[i])/RR[5];
                            tmp = g.p2[i]-tmp;
                            crate::maths::compensated_add(&mut b.p0[i], tmp * C[1], &mut csb.p0[i]);
                            crate::maths::compensated_add(&mut b.p1[i], tmp * C[2], &mut csb.p1[i]);
                            crate::maths::compensated_add(&mut b.p2[i], tmp, &mut csb.p2[i]);
                        }
                    }
                    3 => {
                        for i in 0..num_components {
                            let mut tmp = g.p3[i];
                            let mut gk = ac[i];
                            let mut gk_cs = self.error_a[i];
                            crate::maths::compensated_add(&mut gk, -a0[i], &mut gk_cs);
                            crate::maths::compensated_add(&mut gk, csa0[i], &mut gk_cs);
                            g.p3[i] = (((gk/RR[6] - g.p0[i])/RR[7] - g.p1[i])/RR[8] - g.p2[i])/RR[9];
                            tmp = g.p3[i]-tmp;
                            crate::maths::compensated_add(&mut b.p0[i], tmp * C[3], &mut csb.p0[i]);
                            crate::maths::compensated_add(&mut b.p1[i], tmp * C[4], &mut csb.p1[i]);
                            crate::maths::compensated_add(&mut b.p2[i], tmp * C[5], &mut csb.p2[i]);
                            crate::maths::compensated_add(&mut b.p3[i], tmp, &mut csb.p3[i]);
                        }
                    }
                    4 => {
                        for i in 0..num_components {
                            let mut tmp = g.p4[i];
                            let mut gk = ac[i];
                            let mut gk_cs = self.error_a[i];
                            crate::maths::compensated_add(&mut gk, -a0[i], &mut gk_cs);
                            crate::maths::compensated_add(&mut gk, csa0[i], &mut gk_cs);
                            g.p4[i] = ((((gk/RR[10] - g.p0[i])/RR[11] - g.p1[i])/RR[12] - g.p2[i])/RR[13] - g.p3[i])/RR[14];
                            tmp = g.p4[i]-tmp;
                            crate::maths::compensated_add(&mut b.p0[i], tmp * C[6], &mut csb.p0[i]);
                            crate::maths::compensated_add(&mut b.p1[i], tmp * C[7], &mut csb.p1[i]);
                            crate::maths::compensated_add(&mut b.p2[i], tmp * C[8], &mut csb.p2[i]);
                            crate::maths::compensated_add(&mut b.p3[i], tmp * C[9], &mut csb.p3[i]);
                            crate::maths::compensated_add(&mut b.p4[i], tmp, &mut csb.p4[i]);
                        }
                    }
                    5 => {
                        for i in 0..num_components {
                            let mut tmp = g.p5[i];
                            let mut gk = ac[i];
                            let mut gk_cs = self.error_a[i];
                            crate::maths::compensated_add(&mut gk, -a0[i], &mut gk_cs);
                            crate::maths::compensated_add(&mut gk, csa0[i], &mut gk_cs);
                            g.p5[i] = (((((gk/RR[15] - g.p0[i])/RR[16] - g.p1[i])/RR[17] - g.p2[i])/RR[18] - g.p3[i])/RR[19] - g.p4[i])/RR[20];
                            tmp = g.p5[i]-tmp;
                            crate::maths::compensated_add(&mut b.p0[i], tmp * C[10], &mut csb.p0[i]);
                            crate::maths::compensated_add(&mut b.p1[i], tmp * C[11], &mut csb.p1[i]);
                            crate::maths::compensated_add(&mut b.p2[i], tmp * C[12], &mut csb.p2[i]);
                            crate::maths::compensated_add(&mut b.p3[i], tmp * C[13], &mut csb.p3[i]);
                            crate::maths::compensated_add(&mut b.p4[i], tmp * C[14], &mut csb.p4[i]);
                            crate::maths::compensated_add(&mut b.p5[i], tmp, &mut csb.p5[i]);
                        }
                    }
                    6 => {
                        // get delta in b here // global error in rebound terms                
                        let mut max_a = 0.;
                        let mut max_b6d = 0.;
                        for i in 0..num_components {
                            let mut tmp = g.p6[i];
                            let mut gk = ac[i];
                            let mut gk_cs = self.error_a[i];
                            crate::maths::compensated_add(&mut gk, -a0[i], &mut gk_cs);
                            crate::maths::compensated_add(&mut gk, csa0[i], &mut gk_cs);
                            g.p6[i] = ((((((gk/RR[21] - g.p0[i])/RR[22] - g.p1[i])/RR[23] - g.p2[i])/RR[24] - g.p3[i])/RR[25] - g.p4[i])/RR[26] - g.p5[i])/RR[27];
                            tmp = g.p6[i]-tmp;
                            crate::maths::compensated_add(&mut b.p0[i], tmp * C[15], &mut csb.p0[i]);
                            crate::maths::compensated_add(&mut b.p1[i], tmp * C[16], &mut csb.p1[i]);
                            crate::maths::compensated_add(&mut b.p2[i], tmp * C[17], &mut csb.p2[i]);
                            crate::maths::compensated_add(&mut b.p3[i], tmp * C[18], &mut csb.p3[i]);
                            crate::maths::compensated_add(&mut b.p4[i], tmp * C[19], &mut csb.p4[i]);
                            crate::maths::compensated_add(&mut b.p5[i], tmp * C[20], &mut csb.p5[i]);
                            crate::maths::compensated_add(&mut b.p6[i], tmp, &mut csb.p6[i]);

                            // global
                            let tmp_a = ac[i].abs();
                            if tmp_a > max_a {
                                max_a = tmp_a;
                            }
                            let tmp_b6d = tmp.abs();
                            if tmp_b6d > max_b6d {
                                max_b6d = tmp_b6d;
                            }

                            // local
                            // let e = (tmp/ac[i]).abs();
                            // if e > predictor_corrector_error {
                            //     predictor_corrector_error = e;
                            // }
                        }
                        predictor_corrector_error = max_b6d/max_a;
                    }
                    _ => {}
                }
            }
        }
        // (the time set doesn't happen in mine - seems to just be for the mengro thing)
        // (guess new timestep) // using PRS23 (new default)
        let mut min_timescale = f64::INFINITY;
        for i in 0..self.system.num_particles() {
            let mut y2 = 0.;
            let mut y3 = 0.;
            let mut y4 = 0.;
            for k in 3*i..3*(i+1) {
                let tmp = a0[k] + b.p0[k] + b.p1[k] + b.p2[k] + b.p3[k] + b.p4[k] + b.p5[k] + b.p6[k];
                y2 += tmp * tmp;
                let tmp = b.p0[k] + 2.*b.p1[k] + 3.*b.p2[k] + 4.*b.p3[k] + 5.*b.p4[k] + 6.*b.p5[k] + 7.*b.p6[k];
                y3 += tmp * tmp;
                let tmp = 2.*b.p1[k] + 6.*b.p2[k] + 12.*b.p3[k] + 20.*b.p4[k] + 30.*b.p5[k] + 42.*b.p6[k];
                y4 += tmp * tmp;
            }
            let particle_timescale = 2.*y2/(y3+(y4*y2).sqrt());
            // todo check if normal (add similar stuff elsewhere)
            if particle_timescale < min_timescale {
                min_timescale = particle_timescale;
            }
        }
        let old_delta_t = self.delta_t;
        self.delta_t = min_timescale.sqrt() * old_delta_t * (self.epsilon*5040.).powf(1./7.); // rebound has machine independent powf
        // if too low, redo calculations with the smaller step
        // (todo, and lacking a lot of safety due to that)
        // set new pos/vel (rebound does a compensated sum of each part of the b sum - maybe also try that in the above?) (also the dt*dt not being precalced for error reasons apparently)
        for i in 0..num_components {
            crate::maths::compensated_add(&mut x0[i], b.p6[i]/72.*old_delta_t*old_delta_t, &mut self.error_x[i]);
            crate::maths::compensated_add(&mut x0[i], b.p5[i]/56.*old_delta_t*old_delta_t, &mut self.error_x[i]);
            crate::maths::compensated_add(&mut x0[i], b.p4[i]/42.*old_delta_t*old_delta_t, &mut self.error_x[i]);
            crate::maths::compensated_add(&mut x0[i], b.p3[i]/30.*old_delta_t*old_delta_t, &mut self.error_x[i]);
            crate::maths::compensated_add(&mut x0[i], b.p2[i]/20.*old_delta_t*old_delta_t, &mut self.error_x[i]);
            crate::maths::compensated_add(&mut x0[i], b.p1[i]/12.*old_delta_t*old_delta_t, &mut self.error_x[i]);
            crate::maths::compensated_add(&mut x0[i], b.p0[i]/6.*old_delta_t*old_delta_t, &mut self.error_x[i]);
            crate::maths::compensated_add(&mut x0[i], a0[i]/2.*old_delta_t*old_delta_t, &mut self.error_x[i]);
            crate::maths::compensated_add(&mut x0[i], v0[i]*old_delta_t, &mut self.error_x[i]);
            crate::maths::compensated_add(&mut v0[i], b.p6[i]/8.*old_delta_t, &mut self.error_v[i]);
            crate::maths::compensated_add(&mut v0[i], b.p5[i]/7.*old_delta_t, &mut self.error_v[i]);
            crate::maths::compensated_add(&mut v0[i], b.p4[i]/6.*old_delta_t, &mut self.error_v[i]);
            crate::maths::compensated_add(&mut v0[i], b.p3[i]/5.*old_delta_t, &mut self.error_v[i]);
            crate::maths::compensated_add(&mut v0[i], b.p2[i]/4.*old_delta_t, &mut self.error_v[i]);
            crate::maths::compensated_add(&mut v0[i], b.p1[i]/3.*old_delta_t, &mut self.error_v[i]);
            crate::maths::compensated_add(&mut v0[i], b.p0[i]/2.*old_delta_t, &mut self.error_v[i]);
            crate::maths::compensated_add(&mut v0[i], a0[i]*old_delta_t, &mut self.error_v[i]);
        }
        self.time += old_delta_t;
        // set particles
        for i in 0..self.system.num_particles() {
            self.system.particles[i].position = crate::maths::V3 { x: x0[3*i], y: x0[3*i+1], z: x0[3*i+2] };
            self.system.particles[i].velocity = crate::maths::V3 { x: v0[3*i], y: v0[3*i+1], z: v0[3*i+2] };
        }
        // calculate error in final 'b' and predict next step [todo]
    }
    fn stepn(&mut self, n: usize) {
        (0..n).for_each(|_| self.step());
    }
    fn stept(&mut self, t: f64) {
        self.stepn(t as usize);
    }
    impl_step_untilt!();
    fn write_to(&self, writer: &mut std::io::BufWriter<dyn std::io::Write>) {
        self.system.write_to(writer).unwrap();
    }
}