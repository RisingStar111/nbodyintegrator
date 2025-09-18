// Heavily based upon work by rein and tremaine [todo] (a lot of functionality is copied 1 to 1)
// timestep difference could be due to rebound custom pow_1/7 implementation
// aparently the optimisations on the small case are slower in general wat (did the compiler somehow track the low particle count through multiple files but can't track through nonaliasing?)
// specifically the acceleration (or at least, it's something that does change, and changing back is now faster)
// everything (including acc but not just that) just seems to be 20% (a lot) slower than *python* rebound smh
// definitely room to gain by multithreading more at once, especially on the GPU side as <100 particles the transfer latency is high
// simd also tries to multithread - with just accel gpu, expect to put it on par even at 1k+ particles (maybe even up to 4k?? (my gpu implementation probably sucks))

use std::{ops::Add, simd::{f64x4, StdFloat}, sync::Arc, time::{Duration, Instant}};

use cudarc::{driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg}, nvrtc::compile_ptx};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::{maths::compensated_add, systems, traits::*};

#[derive(Debug)]
pub struct IASCounter {
    pub predictor_iterations: usize,
    pub abort_converged: usize,
    pub abort_oscillation: usize,
    pub abort_slow: usize,
    pub step_failed: usize,
    pub timer: Duration,
}

impl IASCounter {
    pub fn default() -> Self {
        IASCounter { predictor_iterations: 0, abort_converged: 0, abort_oscillation: 0, abort_slow: 0, step_failed: 0, timer: Duration::ZERO }
    }
}

#[derive(Debug, Clone)]
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

    fn set_zero(&mut self) {
        for i in 0..self.p0.len() {
            self.p0[i] = 0.;
            self.p1[i] = 0.;
            self.p2[i] = 0.;
            self.p3[i] = 0.;
            self.p4[i] = 0.;
            self.p5[i] = 0.;
            self.p6[i] = 0.;
        }
    }

    fn index_tuple(&self, i: usize) -> (f64, f64, f64, f64, f64, f64, f64) {
        (self.p0[i], self.p1[i], self.p2[i], self.p3[i], self.p4[i], self.p5[i], self.p6[i])
    }
}

pub enum AccelerationCalculationMode {
    Serial,
    Simd,
    GPU,
}

pub struct ParticleIAS {
    pub system: systems::ParticleSystem,
    // other settings/etc
    pub time: f64,
    pub delta_t: f64,
    last_delta_t: f64,
    velocity_dependant_forces: bool,
    epsilon: f64,
    min_delta_t: f64,

    pub counters: IASCounter,
    pub steps: usize,

    // persistent storage for compensated sums - rebound clears the error if a particle is added (but not when removed?) - unsure why
    // (depending on when it's cleared, that could cause issues with manipulating particles, eg if it's not cleared when changing positions)
    // could potentially just store the actual particle properties as 'f128's
    error_x: Vec<f64>,
    error_v: Vec<f64>,
    error_a: Vec<f64>,
    guessed_b: Subspacedarray,
    b: Subspacedarray,
    g: Subspacedarray,
    csb: Subspacedarray,

    pub acceleration_calculation: AccelerationCalculationMode,

    // gpu stuff
    stream: Option<Arc<CudaStream>>, // rest is kept in scope by the stream i think
    accel_f: Option<CudaFunction>,

    // rayon stuff
    pub rayon_threads: usize,
}

// not optimised, specially to make use of shared memory // even with single thread cpu at ~500 particles
const PTX_SRC: &str = "
extern \"C\" __global__ void comp_accel(double* x, double* y, double* z, 
                                    double* ax, double* ay, double* az, 
                                    double* eax, double* eay, double* eaz, 
                                    double* m, int N, double G) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;

    // shared memory for particle positions // size defined by the 'shared_mem_bytes' on the launcher
    extern __shared__ double spos[]; // can't have multiple dynamic shared access
    
    // store reused and tmp assigned params
    double ix = 0.0;
    double iy = 0.0;
    double iz = 0.0;
    if (i < N) {
        ix = x[i];
        iy = y[i];
        iz = z[i];
    }
    double axtmp = 0.0;
    double aytmp = 0.0;
    double aztmp = 0.0;
    double eaxtmp = 0.0;
    double eaytmp = 0.0;
    double eaztmp = 0.0;

    // if (i < N) { // all threads must hit barriers
    for (int segment_offset = 0; segment_offset < N; segment_offset += blockDim.x) {
        // load next block into shared memory // each thread does one particle
        spos[3*threadIdx.x] = 0.0;
        spos[3*threadIdx.x+1] = 0.0;
        spos[3*threadIdx.x+2] = 0.0;
        if (segment_offset+threadIdx.x < N) {
            spos[3*threadIdx.x] = x[segment_offset+threadIdx.x];
            spos[3*threadIdx.x+1] = y[segment_offset+threadIdx.x];
            spos[3*threadIdx.x+2] = z[segment_offset+threadIdx.x];
        }
        // sync before loading from shared
        __syncthreads();

        // each thread computes one particle's acceleration // does not affect other particles to ensure concurrency
        // scan this segment
        for (int k = 0; k < blockDim.x; k++) {
            int pk = k + segment_offset;
            if (pk >= N || i >= N) {break;} // stay within bounds
            if (i == pk) {continue;} // don't accelerate self
            double dx = spos[3*k] - ix;
            double dy = spos[3*k+1] - iy;
            double dz = spos[3*k+2] - iz;
            double sq = dx*dx + dy*dy + dz*dz;
            double sqr = sqrt(sq);
            double a_mag = G / (sq * sqr) * m[pk]; // order to preserve randomness of roundoff

            // compensated sum to accelerations
            double valax = dx * a_mag - eaxtmp;
            double valay = dy * a_mag - eaytmp;
            double valaz = dz * a_mag - eaztmp;
            double addax = axtmp + valax;
            double adday = aytmp + valay;
            double addaz = aztmp + valaz;
            eaxtmp = (addax - axtmp) - valax;
            eaytmp = (adday - aytmp) - valay;
            eaztmp = (addaz - aztmp) - valaz;
            axtmp = addax;
            aytmp = adday;
            aztmp = addaz;
        }

        // sync allowing threads to finish reading
        __syncthreads();
    }
    
    // assign outputs
    if (i < N) {
        ax[i] = axtmp;
        ay[i] = aytmp;
        az[i] = aztmp;
        eax[i] = eaxtmp;
        eay[i] = eaytmp;
        eaz[i] = eaztmp;
    }
}
";

impl ParticleIAS {
    pub fn default() -> Self {
        ParticleIAS { 
            system: systems::ParticleSystem::default(),
            time: 0.,
            delta_t: 0.1,
            last_delta_t: 0.,
            velocity_dependant_forces: false,
            epsilon: 1e-9,
            min_delta_t: 1e-9,
            steps: 0,
            counters: IASCounter::default(),
            error_x: vec![],
            error_v: vec![],
            error_a: vec![],
            guessed_b: Subspacedarray::with_length(0),
            b: Subspacedarray::with_length(0),
            g: Subspacedarray::with_length(0),
            csb: Subspacedarray::with_length(0),
            acceleration_calculation: AccelerationCalculationMode::Serial,
            stream: None,
            accel_f: None,
            rayon_threads: 1,
        }
    }

    pub fn enable_gpu(&mut self) {
        // set acceleration mode
        self.acceleration_calculation = AccelerationCalculationMode::GPU;
        // error handling not good atm
        let ptx = compile_ptx(PTX_SRC).unwrap();
        println!("Compilation succeeded");

        let ctx = CudaContext::new(0).unwrap();
        self.stream = Some(ctx.default_stream());
        println!("Built");

        let module = ctx.load_module(ptx).unwrap();
        self.accel_f = Some(module.load_function("comp_accel").unwrap());
        println!("Loaded");
    }

    fn set_acceleration_compensated_gpu(&mut self) {
        // copy particles into the stream and builder
        let mut x_host = vec![];
        let mut y_host = vec![];
        let mut z_host = vec![];
        let mut ax_host = vec![];
        let mut ay_host = vec![];
        let mut az_host = vec![];
        let mut eax_host = vec![];
        let mut eay_host = vec![];
        let mut eaz_host = vec![];
        let mut m_host = vec![];
        for p in &self.system.particles {
            x_host.push(p.position.x);
            y_host.push(p.position.y);
            z_host.push(p.position.z);
            ax_host.push(0.0f64);
            ay_host.push(0.);
            az_host.push(0.);
            eax_host.push(0.0f64);
            eay_host.push(0.);
            eaz_host.push(0.);
            m_host.push(p.mass);
        }
        let s = self.stream.clone().unwrap();
        let x_dev = s.memcpy_stod(&x_host).unwrap();
        let y_dev = s.memcpy_stod(&y_host).unwrap();
        let z_dev = s.memcpy_stod(&z_host).unwrap();
        let mut ax_dev = s.memcpy_stod(&ax_host).unwrap();
        let mut ay_dev = s.memcpy_stod(&ay_host).unwrap();
        let mut az_dev = s.memcpy_stod(&az_host).unwrap();
        let mut eax_dev = s.memcpy_stod(&eax_host).unwrap();
        let mut eay_dev = s.memcpy_stod(&eay_host).unwrap();
        let mut eaz_dev = s.memcpy_stod(&eaz_host).unwrap();
        let m_dev = s.memcpy_stod(&m_host).unwrap();
        // println!("Copied in {:?}", start.elapsed());

        // run kernel
        let tasks = self.system.num_particles() as u32;
        let threads_per_block = 128; // idk what these numbers should be // 128 seems ok atm
        let blocks_per_grid = (tasks + threads_per_block - 1) / threads_per_block;
        let mut builder = s.launch_builder(self.accel_f.as_ref().unwrap());
        builder.arg(&x_dev);
        builder.arg(&y_dev);
        builder.arg(&z_dev);
        builder.arg(&mut ax_dev);
        builder.arg(&mut ay_dev);
        builder.arg(&mut az_dev);
        builder.arg(&mut eax_dev);
        builder.arg(&mut eay_dev);
        builder.arg(&mut eaz_dev);
        builder.arg(&m_dev);
        let i32tasks = tasks as i32;
        builder.arg(&i32tasks);
        builder.arg(&self.system.constants.G);
        let cfg = LaunchConfig {
            block_dim: (threads_per_block,1,1),
            grid_dim: (blocks_per_grid,1,1),
            shared_mem_bytes: 8*threads_per_block*3, // f64 * threads per block * dimensions
        };
        unsafe { builder.launch(cfg) }.unwrap();

        // get data back from gpu
        s.memcpy_dtoh(&ax_dev, &mut ax_host).unwrap();
        s.memcpy_dtoh(&ay_dev, &mut ay_host).unwrap();
        s.memcpy_dtoh(&az_dev, &mut az_host).unwrap();
        s.memcpy_dtoh(&eax_dev, &mut eax_host).unwrap();
        s.memcpy_dtoh(&eay_dev, &mut eay_host).unwrap();
        s.memcpy_dtoh(&eaz_dev, &mut eaz_host).unwrap();

        // copy data back to particles
        for i in 0..self.system.num_particles() {
            self.system.particles[i].acceleration.x = ax_host[i]; // can ofc make this nicer for compiler
            self.system.particles[i].acceleration.y = ay_host[i];
            self.system.particles[i].acceleration.z = az_host[i];
            self.error_a[3*i] = eax_host[i];
            self.error_a[3*i+1] = eay_host[i];
            self.error_a[3*i+2] = eaz_host[i];
        }
        // println!("Exited in {:?}", start.elapsed());
    }

    fn vec_to_x4(vector: Vec<f64>) -> Vec<f64x4>{ // bad, move elsewhere // tbh i doubt compiler can work out the best way to do this anyway // especially with the extra preprocess from particles
        let mut out = vec![];
        for i in 0..(vector.len()+3)/4 {
            out.push(f64x4::from_array([
                *vector.get(4*i).unwrap_or(&0.),
                *vector.get(4*i+1).unwrap_or(&0.),
                *vector.get(4*i+2).unwrap_or(&0.),
                *vector.get(4*i+3).unwrap_or(&0.),
            ]));
        }
        out
    }
    fn x4_to_vec(vector: Vec<f64x4>) -> Vec<f64>{ // bad, move elsewhere // tbh i doubt compiler can work out the best way to do this anyway // especially with the extra preprocess from particles
        vector.into_iter().flat_map(|v| *v.as_array()).collect::<Vec<f64>>()
    }

    fn set_acceleration_compensated_simd(&mut self) { //(only 0% faster with how bad it is)
        // move particle data into vecs and splat into f64x4s for batch
        // actually a very odd thing to do, and doesn't need padding depending on how you run it, at best results in 2x (or 4 if able to swap to x8s) speedup (since double work as not trianguling)
        let mut x_vec = vec![];
        let mut y_vec = vec![];
        let mut z_vec = vec![];
        let mut splat_x = vec![];
        let mut splat_y = vec![];
        let mut splat_z = vec![];
        let mut splat_m = vec![];
        for p in &self.system.particles {
            x_vec.push(p.position.x);
            y_vec.push(p.position.y);
            z_vec.push(p.position.z);
            splat_x.push(f64x4::splat(p.position.x));
            splat_y.push(f64x4::splat(p.position.y));
            splat_z.push(f64x4::splat(p.position.z));
            splat_m.push(f64x4::splat(p.mass));
        }
        let x = ParticleIAS::vec_to_x4(x_vec);
        let y = ParticleIAS::vec_to_x4(y_vec);
        let z = ParticleIAS::vec_to_x4(z_vec);
        let zero_vec = vec![0.; self.system.particles.len()];
        let mut ax = ParticleIAS::vec_to_x4(zero_vec);
        let mut ay = ax.clone();
        let mut az = ax.clone();
        let mut eax = ax.clone();
        let mut eay = ax.clone();
        let mut eaz = ax.clone();
        let g = f64x4::splat(self.system.constants.G);
        let four_len = x.len();

        // do acceleration dings
        // padding results in / 0 and thus NaNs but this shouldn't ever make it back to the system
        if self.rayon_threads == 1 {
            for four_index in 0..four_len {
                let mut ix = x[four_index];
                let mut iy = y[four_index];
                let mut iz = z[four_index];
                let mut axtmp = f64x4::splat(0.);
                let mut aytmp = f64x4::splat(0.);
                let mut aztmp = f64x4::splat(0.);
                let mut eaxtmp = f64x4::splat(0.);
                let mut eaytmp = f64x4::splat(0.);
                let mut eaztmp = f64x4::splat(0.);
                for j in 0..splat_m.len() {
                    let dx = splat_x[j] - ix;
                    let dy = splat_y[j] - iy;
                    let dz = splat_z[j] - iz;
                    let sq = dx*dx + dy*dy + dz*dz;
                    let sqr = sq.sqrt(); // iirc this doesn't really get sped up by simd
                    let a_mag = g / (sq * sqr) * splat_m[j];
                    // set change to 0 if nan (eg from accelerating self) // likely abysmal atm
                    let a_mag = f64x4::from_array(a_mag.as_array().map(|v| if v.is_finite() {v} else {0.}));

                    // compensated sums
                    let valax = dx * a_mag - eaxtmp;
                    let valay = dy * a_mag - eaytmp;
                    let valaz = dz * a_mag - eaztmp;
                    let addax = axtmp + valax;
                    let adday = aytmp + valay;
                    let addaz = aztmp + valaz;
                    eaxtmp = (addax - axtmp) - valax;
                    eaytmp = (adday - aytmp) - valay;
                    eaztmp = (addaz - aztmp) - valaz;
                    axtmp = addax;
                    aytmp = adday;
                    aztmp = addaz;
                }
                ax[four_index] = axtmp;
                ay[four_index] = aytmp;
                az[four_index] = aztmp;
                eax[four_index] = eaxtmp;
                eay[four_index] = eaytmp;
                eaz[four_index] = eaztmp;
            }
        }
        else if self.rayon_threads > 1 {
            (0..four_len).into_par_iter()
            .zip(ax.par_iter_mut())
            .zip(ay.par_iter_mut())
            .zip(az.par_iter_mut())
            .zip(eax.par_iter_mut())
            .zip(eay.par_iter_mut())
            .zip(eaz.par_iter_mut()).for_each(|((((((four_index, ax), ay), az), eax), eay), eaz)| {
                let mut ix = x[four_index];
                let mut iy = y[four_index];
                let mut iz = z[four_index];
                let mut axtmp = *ax;
                let mut aytmp = *ay;
                let mut aztmp = *az;
                let mut eaxtmp = *eax;
                let mut eaytmp = *eay;
                let mut eaztmp = *eaz;
                for j in 0..splat_m.len() {
                    let dx = splat_x[j] - ix;
                    let dy = splat_y[j] - iy;
                    let dz = splat_z[j] - iz;
                    let sq = dx*dx + dy*dy + dz*dz;
                    let sqr = sq.sqrt(); // iirc this doesn't really get sped up by simd
                    let a_mag = g / (sq * sqr) * splat_m[j];
                    // set change to 0 if nan (eg from accelerating self) // likely abysmal atm
                    let a_mag = f64x4::from_array(a_mag.as_array().map(|v| if v.is_finite() {v} else {0.}));

                    // compensated sums
                    let valax = dx * a_mag - eaxtmp;
                    let valay = dy * a_mag - eaytmp;
                    let valaz = dz * a_mag - eaztmp;
                    let addax = axtmp + valax;
                    let adday = aytmp + valay;
                    let addaz = aztmp + valaz;
                    eaxtmp = (addax - axtmp) - valax;
                    eaytmp = (adday - aytmp) - valay;
                    eaztmp = (addaz - aztmp) - valaz;
                    axtmp = addax;
                    aytmp = adday;
                    aztmp = addaz;
                }
                *ax = axtmp;
                *ay = aytmp;
                *az = aztmp;
                *eax = eaxtmp;
                *eay = eaytmp;
                *eaz = eaztmp;
            }
            ); // par iter
        }

        // move data back to particles
        let ax_vec = ParticleIAS::x4_to_vec(ax);
        let ay_vec = ParticleIAS::x4_to_vec(ay);
        let az_vec = ParticleIAS::x4_to_vec(az);
        let eax_vec = ParticleIAS::x4_to_vec(eax);
        let eay_vec = ParticleIAS::x4_to_vec(eay);
        let eaz_vec = ParticleIAS::x4_to_vec(eaz);
        self.error_a.clear();
        for i in 0..self.system.particles.len() {
            self.system.particles[i].acceleration = [ax_vec[i], ay_vec[i], az_vec[i]].into();
            self.error_a.push(eax_vec[i]);
            self.error_a.push(eay_vec[i]);
            self.error_a.push(eaz_vec[i]);
        }
    }

    fn update_acceleration(&mut self) {
        let now = Instant::now();
        match self.acceleration_calculation {
            AccelerationCalculationMode::Serial => {
                self.error_a = self.system.set_acceleration_compensated().iter().flat_map(|f| f.array()).collect();
            }
            AccelerationCalculationMode::Simd => {
                self.set_acceleration_compensated_simd();
            }
            AccelerationCalculationMode::GPU => {
                self.set_acceleration_compensated_gpu();
            }
        }
        self.counters.timer += now.elapsed();
    }

    fn predict_next_step(&mut self, timestep_ratio: f64) {
        if timestep_ratio > 20. { // reset values if massive step delta (this should never happen if safety factor is enabled)
            self.guessed_b = Subspacedarray::with_length(self.b.p0.len());
            self.b = Subspacedarray::with_length(self.b.p0.len());
            return;
        }

        let q1 = timestep_ratio;
        let q2 = q1 * q1;
        let q3 = q1 * q2;
        let q4 = q2 * q2;
        let q5 = q2 * q3;
        let q6 = q3 * q3;
        let q7 = q3 * q4;

        for i in 0..self.b.p0.len() {
            let be0 = self.b.p0[i] - self.guessed_b.p0[i];
            let be1 = self.b.p1[i] - self.guessed_b.p1[i];
            let be2 = self.b.p2[i] - self.guessed_b.p2[i];
            let be3 = self.b.p3[i] - self.guessed_b.p3[i];
            let be4 = self.b.p4[i] - self.guessed_b.p4[i];
            let be5 = self.b.p5[i] - self.guessed_b.p5[i];
            let be6 = self.b.p6[i] - self.guessed_b.p6[i];

            let p6 = self.b.p6[i];
            let p5 = self.b.p5[i];
            let p4 = self.b.p4[i];
            let p3 = self.b.p3[i];
            let p2 = self.b.p2[i];
            let p1 = self.b.p1[i];
            let p0 = self.b.p0[i];
            self.guessed_b.p0[i] = q1*(p6* 7.0 + p5* 6.0 + p4* 5.0 + p3* 4.0 + p2* 3.0 + p1*2.0 + p0);
            self.guessed_b.p1[i] = q2*(p6*21.0 + p5*15.0 + p4*10.0 + p3* 6.0 + p2* 3.0 + p1);
            self.guessed_b.p2[i] = q3*(p6*35.0 + p5*20.0 + p4*10.0 + p3* 4.0 + p2);
            self.guessed_b.p3[i] = q4*(p6*35.0 + p5*15.0 + p4* 5.0 + p3);
            self.guessed_b.p4[i] = q5*(p6*21.0 + p5* 6.0 + p4);
            self.guessed_b.p5[i] = q6*(p6* 7.0 + p5);
            self.guessed_b.p6[i] = q7* p6;

            self.b.p0[i] = self.guessed_b.p0[i] + be0;
            self.b.p1[i] = self.guessed_b.p1[i] + be1;
            self.b.p2[i] = self.guessed_b.p2[i] + be2;
            self.b.p3[i] = self.guessed_b.p3[i] + be3;
            self.b.p4[i] = self.guessed_b.p4[i] + be4;
            self.b.p5[i] = self.guessed_b.p5[i] + be5;
            self.b.p6[i] = self.guessed_b.p6[i] + be6; // why is the guessed b stored separate to the guess+delta (and why is just guess used as last guess in next prediction (it's very important tho))?
        }
    }
    fn try_step(&mut self) -> Option<()> {
        // rebound converts AoS to SoA at some point for caching, and tbf most of the time it will be negligible cost
        let num_components = self.system.num_particles() * 3;
        if self.error_x.len() < num_components {
            self.error_x = vec![0.; num_components];
            self.error_v = vec![0.; num_components];
            self.error_a = vec![0.; num_components];
            self.b = Subspacedarray::with_length(num_components);
            self.guessed_b = Subspacedarray::with_length(num_components);
            self.g = Subspacedarray::with_length(num_components);
            self.csb = Subspacedarray::with_length(num_components);
        }
        // for i in (0..self.system.num_particles()).rev() { // why does this break it // cuz it doesn't break the normal once
        //     self.system.particles[i].position = self.system.particles[i].position - self.system.particles[0].position;
        //     self.system.particles[i].velocity = self.system.particles[i].velocity - self.system.particles[0].velocity;
        //     self.system.particles[i].acceleration = self.system.particles[i].acceleration - self.system.particles[0].acceleration;
        // }
        self.update_acceleration();
        // the actual allocation of memory seems to be somewhat slow for small numbers of particles
        // the '0' ones aren't slow tho idk (unless accessors were slowing it down more than gained)
        let mut x0 = self.system.particles.iter().flat_map(|p| p.position.array()).collect::<Vec<f64>>();
        let mut v0 = self.system.particles.iter().flat_map(|p| p.velocity.array()).collect::<Vec<f64>>();
        let a0 = self.system.particles.iter().flat_map(|p| p.acceleration.array()).collect::<Vec<f64>>();
        // intitialise the compensated sum errors if not present
        let csa0 = self.error_a.clone(); // faster than storing it // maybe there's weird compiler things going on
        // until converged
        self.csb.set_zero();
        self.g.set_zero();
        for k in 0..num_components {
            let p6 = self.b.p6[k]; // again didn't compile out repeated accessors
            let p5 = self.b.p5[k]; // gpt suggests this is due to them being in (same) struct and rust not tracking noalias down far enough, which tracks
            let p4 = self.b.p4[k];
            let p3 = self.b.p3[k];
            let p2 = self.b.p2[k];
            let p1 = self.b.p1[k];
            let p0 = self.b.p0[k];
            self.g.p0[k] = p6*D[15] + p5*D[10] + p4*D[6] + p3*D[3]  + p2*D[1]  + p1*D[0]  + p0;
            self.g.p1[k] = p6*D[16] + p5*D[11] + p4*D[7] + p3*D[4]  + p2*D[2]  + p1;
            self.g.p2[k] = p6*D[17] + p5*D[12] + p4*D[8] + p3*D[5]  + p2;
            self.g.p3[k] = p6*D[18] + p5*D[13] + p4*D[9] + p3;
            self.g.p4[k] = p6*D[19] + p5*D[14] + p4;
            self.g.p5[k] = p6*D[20] + p5;
            self.g.p6[k] = p6;
        }
        let mut last_predictor_corrector_error = 2.; // ? rebound sets to 2.
        let mut predictor_corrector_error = f64::MAX;
        let mut iterations = 0;
        loop {
            self.counters.predictor_iterations += 1; // mine averages 3 full loops before breaking, while rebound paper suggests they get 2
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
                // let now = Instant::now();
                for i in 0..self.system.num_particles() {
                    // predict position
                    // delta is from start of full step
                    // rebound docs suggest using x * dt * dt rather than storing dt*dt to randomise roundoff error
                    // in this one tho they chain everything in one equation which as well as using divides i'm sus on error accumulation (tho i think it's 3rd order so eh)
                    // tho i suppose, anything 'costly' here will be dwarfed by the forces calc... (and divide should be more precise?)
                    // also recommends x(p/q) as p*x/q
                    // assign error adjusted position to real particles, for the force calc to use // also why the loop is partially unrolled
                    // self.system.particles[i].position = 
                    // [3*i, 3*i+1, 3*i+2].map(|k|
                    //     -self.error_x[k]
                    //     + ((((((((b.p6[k]*7.*substep/9. + b.p5[k])*3.*substep/4. + b.p4[k])*5.*substep/7. + b.p3[k])*2.*substep/3. + b.p2[k])*3.*substep/5. + b.p1[k])*substep/2. + b.p0[k])*substep/3. + a0[k])*self.delta_t*substep/2. + v0[k])*self.delta_t*substep
                    //     + x0[k]
                    // ).into(); // check emmitted code as may not compile well // indeed below is 50% faster

                    let k = 3*i;
                    let e = -self.error_x[k] + ((((((((self.b.p6[k]*7.*substep/9. + self.b.p5[k])*3.*substep/4. + self.b.p4[k])*5.*substep/7. + self.b.p3[k])*2.*substep/3. + self.b.p2[k])*3.*substep/5. + self.b.p1[k])*substep/2. + self.b.p0[k])*substep/3. + a0[k])*self.delta_t*substep/2. + v0[k])*self.delta_t*substep + x0[k];
                    let k = 3*i+1;
                    let q = -self.error_x[k] + ((((((((self.b.p6[k]*7.*substep/9. + self.b.p5[k])*3.*substep/4. + self.b.p4[k])*5.*substep/7. + self.b.p3[k])*2.*substep/3. + self.b.p2[k])*3.*substep/5. + self.b.p1[k])*substep/2. + self.b.p0[k])*substep/3. + a0[k])*self.delta_t*substep/2. + v0[k])*self.delta_t*substep + x0[k];
                    let k = 3*i+2;
                    let x = -self.error_x[k] + ((((((((self.b.p6[k]*7.*substep/9. + self.b.p5[k])*3.*substep/4. + self.b.p4[k])*5.*substep/7. + self.b.p3[k])*2.*substep/3. + self.b.p2[k])*3.*substep/5. + self.b.p1[k])*substep/2. + self.b.p0[k])*substep/3. + a0[k])*self.delta_t*substep/2. + v0[k])*self.delta_t*substep + x0[k];
                    self.system.particles[i].position = [e, q, x].into();
                    
                    // let k = 3*i; // slower
                    // self.system.particles[i].position.x = -self.error_x[k] + ((((((((self.b.p6[k]*7.*substep/9. + self.b.p5[k])*3.*substep/4. + self.b.p4[k])*5.*substep/7. + self.b.p3[k])*2.*substep/3. + self.b.p2[k])*3.*substep/5. + self.b.p1[k])*substep/2. + self.b.p0[k])*substep/3. + a0[k])*self.delta_t*substep/2. + v0[k])*self.delta_t*substep + x0[k];
                    // let k = 3*i+1;
                    // self.system.particles[i].position.y = -self.error_x[k] + ((((((((self.b.p6[k]*7.*substep/9. + self.b.p5[k])*3.*substep/4. + self.b.p4[k])*5.*substep/7. + self.b.p3[k])*2.*substep/3. + self.b.p2[k])*3.*substep/5. + self.b.p1[k])*substep/2. + self.b.p0[k])*substep/3. + a0[k])*self.delta_t*substep/2. + v0[k])*self.delta_t*substep + x0[k];
                    // let k = 3*i+2;
                    // self.system.particles[i].position.z = -self.error_x[k] + ((((((((self.b.p6[k]*7.*substep/9. + self.b.p5[k])*3.*substep/4. + self.b.p4[k])*5.*substep/7. + self.b.p3[k])*2.*substep/3. + self.b.p2[k])*3.*substep/5. + self.b.p1[k])*substep/2. + self.b.p0[k])*substep/3. + a0[k])*self.delta_t*substep/2. + v0[k])*self.delta_t*substep + x0[k];

                    // let k = 3*i; // equal
                    // let p0 = self.b.p0[k];
                    // let p1 = self.b.p1[k];
                    // let p2 = self.b.p2[k];
                    // let p3 = self.b.p3[k];
                    // let p4 = self.b.p4[k];
                    // let p5 = self.b.p5[k];
                    // let p6 = self.b.p6[k];
                    // let aa = a0[k];
                    // let vv = v0[k];
                    // let xx = x0[k];
                    // let e = -self.error_x[k] + ((((((((p6*7.*substep/9. + p5)*3.*substep/4. + p4)*5.*substep/7. + p3)*2.*substep/3. + p2)*3.*substep/5. + p1)*substep/2. + p0)*substep/3. + aa)*self.delta_t*substep/2. + vv)*self.delta_t*substep + xx;
                    // let k = 3*i+1;
                    // let p0 = self.b.p0[k];
                    // let p1 = self.b.p1[k];
                    // let p2 = self.b.p2[k];
                    // let p3 = self.b.p3[k];
                    // let p4 = self.b.p4[k];
                    // let p5 = self.b.p5[k];
                    // let p6 = self.b.p6[k];
                    // let aa = a0[k];
                    // let vv = v0[k];
                    // let xx = x0[k];
                    // let q = -self.error_x[k] + ((((((((p6*7.*substep/9. + p5)*3.*substep/4. + p4)*5.*substep/7. + p3)*2.*substep/3. + p2)*3.*substep/5. + p1)*substep/2. + p0)*substep/3. + aa)*self.delta_t*substep/2. + vv)*self.delta_t*substep + xx;
                    // let k = 3*i+2;
                    // let p0 = self.b.p0[k];
                    // let p1 = self.b.p1[k];
                    // let p2 = self.b.p2[k];
                    // let p3 = self.b.p3[k];
                    // let p4 = self.b.p4[k];
                    // let p5 = self.b.p5[k];
                    // let p6 = self.b.p6[k];
                    // let aa = a0[k];
                    // let vv = v0[k];
                    // let xx = x0[k];
                    // let x = -self.error_x[k] + ((((((((p6*7.*substep/9. + p5)*3.*substep/4. + p4)*5.*substep/7. + p3)*2.*substep/3. + p2)*3.*substep/5. + p1)*substep/2. + p0)*substep/3. + aa)*self.delta_t*substep/2. + vv)*self.delta_t*substep + xx;
                    // self.system.particles[i].position = [e, q, x].into();
                }
                // self.counters.timer += now.elapsed();
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
                            let mut tmp = self.g.p0[i];
                            let mut gk = ac[i];
                            let mut gk_cs = self.error_a[i];
                            compensated_add(&mut gk, -a0[i], &mut gk_cs);
                            compensated_add(&mut gk, csa0[i], &mut gk_cs);
                            self.g.p0[i] = gk/RR[0]; // the chained divides minimise operation count to reduce error (tho still if copmensated it would be lower again)
                            tmp = self.g.p0[i]-tmp;
                            compensated_add(&mut self.b.p0[i], tmp, &mut self.csb.p0[i]);
                        }
                    }
                    1 => {
                        for i in 0..num_components {
                            let mut tmp = self.g.p1[i];
                            let mut gk = ac[i];
                            let mut gk_cs = self.error_a[i];
                            compensated_add(&mut gk, -a0[i], &mut gk_cs);
                            compensated_add(&mut gk, csa0[i], &mut gk_cs);
                            
                            let p0 = self.g.p0[i];
                            self.g.p1[i] = (gk/RR[1] - p0)/RR[2];
                            tmp = self.g.p1[i]-tmp; // some odd things you can do with the consts that at 2 or more precomp steps but changes the type of error? (also can remove all but last divide by multiplying straight but that would be same expected error? (+1 divide) as all mult)
                            compensated_add(&mut self.b.p0[i], tmp * C[0], &mut self.csb.p0[i]);
                            compensated_add(&mut self.b.p1[i], tmp, &mut self.csb.p1[i]);
                        }
                    }
                    2 => {
                        for i in 0..num_components {
                            let mut tmp = self.g.p2[i];
                            let mut gk = ac[i];
                            let mut gk_cs = self.error_a[i];
                            compensated_add(&mut gk, -a0[i], &mut gk_cs);
                            compensated_add(&mut gk, csa0[i], &mut gk_cs);

                            let p0 = self.g.p0[i];
                            let p1 = self.g.p1[i];
                            self.g.p2[i] = ((gk/RR[3] - p0)/RR[4] - p1)/RR[5];
                            tmp = self.g.p2[i]-tmp;
                            compensated_add(&mut self.b.p0[i], tmp * C[1], &mut self.csb.p0[i]);
                            compensated_add(&mut self.b.p1[i], tmp * C[2], &mut self.csb.p1[i]);
                            compensated_add(&mut self.b.p2[i], tmp, &mut self.csb.p2[i]);
                        }
                    }
                    3 => {
                        for i in 0..num_components {
                            let mut tmp = self.g.p3[i];
                            let mut gk = ac[i];
                            let mut gk_cs = self.error_a[i];
                            compensated_add(&mut gk, -a0[i], &mut gk_cs);
                            compensated_add(&mut gk, csa0[i], &mut gk_cs);

                            let p0 = self.g.p0[i];
                            let p1 = self.g.p1[i];
                            let p2 = self.g.p2[i];
                            self.g.p3[i] = (((gk/RR[6] - p0)/RR[7] - p1)/RR[8] - p2)/RR[9];
                            tmp = self.g.p3[i]-tmp;
                            // self.b.p0[i] += tmp * C[3];
                            // self.b.p1[i] += tmp * C[4];
                            // self.b.p2[i] += tmp * C[5];
                            // self.b.p3[i] += tmp;
                            compensated_add(&mut self.b.p0[i], tmp * C[3], &mut self.csb.p0[i]);
                            compensated_add(&mut self.b.p1[i], tmp * C[4], &mut self.csb.p1[i]);
                            compensated_add(&mut self.b.p2[i], tmp * C[5], &mut self.csb.p2[i]);
                            compensated_add(&mut self.b.p3[i], tmp, &mut self.csb.p3[i]);
                        }
                    }
                    4 => {
                        for i in 0..num_components {
                            let mut tmp = self.g.p4[i];
                            let mut gk = ac[i];
                            let mut gk_cs = self.error_a[i];
                            compensated_add(&mut gk, -a0[i], &mut gk_cs);
                            compensated_add(&mut gk, csa0[i], &mut gk_cs);

                            let p0 = self.g.p0[i];
                            let p1 = self.g.p1[i];
                            let p2 = self.g.p2[i];
                            let p3 = self.g.p3[i];
                            self.g.p4[i] = ((((gk/RR[10] - p0)/RR[11] - p1)/RR[12] - p2)/RR[13] - p3)/RR[14];
                            tmp = self.g.p4[i]-tmp;
                            compensated_add(&mut self.b.p0[i], tmp * C[6], &mut self.csb.p0[i]);
                            compensated_add(&mut self.b.p1[i], tmp * C[7], &mut self.csb.p1[i]);
                            compensated_add(&mut self.b.p2[i], tmp * C[8], &mut self.csb.p2[i]);
                            compensated_add(&mut self.b.p3[i], tmp * C[9], &mut self.csb.p3[i]);
                            compensated_add(&mut self.b.p4[i], tmp, &mut self.csb.p4[i]);
                        }
                    }
                    5 => {
                        for i in 0..num_components {
                            let mut tmp = self.g.p5[i];
                            let mut gk = ac[i];
                            let mut gk_cs = self.error_a[i];
                            compensated_add(&mut gk, -a0[i], &mut gk_cs);
                            compensated_add(&mut gk, csa0[i], &mut gk_cs);

                            let p0 = self.g.p0[i];
                            let p1 = self.g.p1[i];
                            let p2 = self.g.p2[i];
                            let p3 = self.g.p3[i];
                            let p4 = self.g.p4[i];
                            self.g.p5[i] = (((((gk/RR[15] - p0)/RR[16] - p1)/RR[17] - p2)/RR[18] - p3)/RR[19] - p4)/RR[20];
                            tmp = self.g.p5[i]-tmp;
                            compensated_add(&mut self.b.p0[i], tmp * C[10], &mut self.csb.p0[i]);
                            compensated_add(&mut self.b.p1[i], tmp * C[11], &mut self.csb.p1[i]);
                            compensated_add(&mut self.b.p2[i], tmp * C[12], &mut self.csb.p2[i]);
                            compensated_add(&mut self.b.p3[i], tmp * C[13], &mut self.csb.p3[i]);
                            compensated_add(&mut self.b.p4[i], tmp * C[14], &mut self.csb.p4[i]);
                            compensated_add(&mut self.b.p5[i], tmp, &mut self.csb.p5[i]);
                        }
                    }
                    6 => {
                        // get delta in b here // global error in rebound terms
                        let mut max_a: f64 = 0.;
                        let mut max_b6d: f64 = 0.;
                        for i in 0..num_components {
                            let mut tmp = self.g.p6[i];
                            let mut gk = ac[i];
                            max_a = max_a.max(gk.abs()); // avoid regrabbing ac[i] later
                            let mut gk_cs = self.error_a[i];
                            compensated_add(&mut gk, -a0[i], &mut gk_cs);
                            compensated_add(&mut gk, csa0[i], &mut gk_cs);
                            
                            let p0 = self.g.p0[i];
                            let p1 = self.g.p1[i];
                            let p2 = self.g.p2[i];
                            let p3 = self.g.p3[i];
                            let p4 = self.g.p4[i];
                            let p5 = self.g.p5[i];
                            // let (p0, p1, p2, p3, p4, p5, p6) = self.g.index_tuple(i); // why is this slower
                            self.g.p6[i] = ((((((gk/RR[21] - p0)/RR[22] - p1)/RR[23] - p2)/RR[24] - p3)/RR[25] - p4)/RR[26] - p5)/RR[27];
                            tmp = self.g.p6[i]-tmp;
                            max_b6d = max_b6d.max(tmp.abs()); // possibly avoid uncaching tmp
                            compensated_add(&mut self.b.p0[i], tmp * C[15], &mut self.csb.p0[i]);
                            compensated_add(&mut self.b.p1[i], tmp * C[16], &mut self.csb.p1[i]);
                            compensated_add(&mut self.b.p2[i], tmp * C[17], &mut self.csb.p2[i]);
                            compensated_add(&mut self.b.p3[i], tmp * C[18], &mut self.csb.p3[i]);
                            compensated_add(&mut self.b.p4[i], tmp * C[19], &mut self.csb.p4[i]);
                            compensated_add(&mut self.b.p5[i], tmp * C[20], &mut self.csb.p5[i]);
                            compensated_add(&mut self.b.p6[i], tmp, &mut self.csb.p6[i]);
                            
                            // global // handled above
                            // max_a = max_a.max(ac[i].abs());
                            // max_b6d = max_b6d.max(tmp.abs());
                            
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
            let mut a0i = 0.;
            let mut y2 = 0.;
            let mut y3 = 0.;
            let mut y4 = 0.;
            for k in 3*i..3*(i+1) {
                let p0 = self.b.p0[k];
                let p1 = self.b.p1[k];
                let p2 = self.b.p2[k];
                let p3 = self.b.p3[k];
                let p4 = self.b.p4[k];
                let p5 = self.b.p5[k];
                let p6 = self.b.p6[k];
                a0i += a0[k] * a0[k];
                let tmp = a0[k] + p0 + p1 + p2 + p3 + p4 + p5 + p6;
                y2 += tmp * tmp;
                let tmp = p0 + 2.*p1 + 3.*p2 + 4.*p3 + 5.*p4 + 6.*p5 + 7.*p6;
                y3 += tmp * tmp;
                let tmp = 2.*p1 + 6.*p2 + 12.*p3 + 20.*p4 + 30.*p5 + 42.*p6;
                y4 += tmp * tmp;
            }
            if !a0i.is_normal() {continue;} // this was apparently very important in specific situations
            let particle_timescale = 2.*y2/(y3+(y4*y2).sqrt());
            // todo check if normal (add similar stuff elsewhere)
            if particle_timescale.is_normal() && particle_timescale < min_timescale {
                min_timescale = particle_timescale;
            }
        }
        let old_delta_t = self.delta_t;
        // todo doesn't handle negative timestep yet
        self.delta_t = self.min_delta_t.max(min_timescale.sqrt() * old_delta_t * (self.epsilon*5040.).powf(1./7.)); // rebound has machine independent powf
        // if too low, redo calculations with the smaller step
        let safety_factor = 0.25; // what rebound uses
        let timestep_ratio = (self.delta_t / old_delta_t).abs();
        if timestep_ratio < safety_factor { // including this abort leads to significantly different (assumedly better) results
            // reset particles
            for i in 0..self.system.num_particles() {
                let x = 3*i;
                let y = 3*i+1;
                let z = 3*i+2;
                self.system.particles[i].position = [x0[x], x0[y], x0[z]].into();
                self.system.particles[i].velocity = [v0[x], v0[y], v0[z]].into();
                self.system.particles[i].acceleration = [a0[x], a0[y], a0[z]].into();
            }
            if self.last_delta_t != 0. {
                let timestep_ratio = (self.delta_t / self.last_delta_t).abs();
                // todo this is where the er/br things in rebound are used
                // but also this isn't the biggest issue since this shouldn't activate on step 1
                self.predict_next_step(timestep_ratio);
            }
            self.counters.step_failed += 1;
            return None;
        }
        // limit stepsize increase
        if self.delta_t * safety_factor > old_delta_t {
            self.delta_t = old_delta_t / safety_factor; // (safety < 1 => new > old)
        }
        // (todo, and lacking a lot of safety due to that)
        // set new pos/vel (rebound does a compensated sum of each part of the b sum - maybe also try that in the above?) (also the dt*dt not being precalced for error reasons apparently)
        for i in 0..num_components {
            let mut errx = self.error_x[i]; // this is like 30% faster hmm // compiler maybe doesn't like optimising accessors
            compensated_add(&mut x0[i], self.b.p6[i]/72.*old_delta_t*old_delta_t, &mut errx);
            compensated_add(&mut x0[i], self.b.p5[i]/56.*old_delta_t*old_delta_t, &mut errx);
            compensated_add(&mut x0[i], self.b.p4[i]/42.*old_delta_t*old_delta_t, &mut errx);
            compensated_add(&mut x0[i], self.b.p3[i]/30.*old_delta_t*old_delta_t, &mut errx);
            compensated_add(&mut x0[i], self.b.p2[i]/20.*old_delta_t*old_delta_t, &mut errx);
            compensated_add(&mut x0[i], self.b.p1[i]/12.*old_delta_t*old_delta_t, &mut errx);
            compensated_add(&mut x0[i], self.b.p0[i]/6.*old_delta_t*old_delta_t, &mut errx);
            compensated_add(&mut x0[i], a0[i]/2.*old_delta_t*old_delta_t, &mut errx);
            compensated_add(&mut x0[i], v0[i]*old_delta_t, &mut errx);
            self.error_x[i] = errx;
            let mut errv = self.error_v[i]; // same for v0 doesn't help
            compensated_add(&mut v0[i], self.b.p6[i]/8.*old_delta_t, &mut errv);
            compensated_add(&mut v0[i], self.b.p5[i]/7.*old_delta_t, &mut errv);
            compensated_add(&mut v0[i], self.b.p4[i]/6.*old_delta_t, &mut errv);
            compensated_add(&mut v0[i], self.b.p3[i]/5.*old_delta_t, &mut errv);
            compensated_add(&mut v0[i], self.b.p2[i]/4.*old_delta_t, &mut errv);
            compensated_add(&mut v0[i], self.b.p1[i]/3.*old_delta_t, &mut errv);
            compensated_add(&mut v0[i], self.b.p0[i]/2.*old_delta_t, &mut errv);
            compensated_add(&mut v0[i], a0[i]*old_delta_t, &mut errv);
            self.error_v[i] = errv;
        }
        self.time += old_delta_t;
        self.last_delta_t = old_delta_t;
        // set particles
        for i in 0..self.system.num_particles() {
            self.system.particles[i].position = [x0[3*i], x0[3*i+1], x0[3*i+2]].into();
            self.system.particles[i].velocity = [v0[3*i], v0[3*i+1], v0[3*i+2]].into();
        }
        // calculate error in final 'b' and predict next step [todo]
        self.predict_next_step(timestep_ratio);
        Some(())
    }
}

const SUBSPACINGS: [f64; 8] = [0.0, 0.0562625605369221464656521910318, 0.180240691736892364987579942780, 0.352624717113169637373907769648, 0.547153626330555383001448554766, 0.734210177215410531523210605558, 0.885320946839095768090359771030, 0.977520613561287501891174488626];
const RR: [f64; 28] = [0.0562625605369221464656522, 0.1802406917368923649875799, 0.1239781311999702185219278, 0.3526247171131696373739078, 0.2963621565762474909082556, 0.1723840253762772723863278, 0.5471536263305553830014486, 0.4908910657936332365357964, 0.3669129345936630180138686, 0.1945289092173857456275408, 0.7342101772154105315232106, 0.6779476166784883850575584, 0.5539694854785181665356307, 0.3815854601022408941493028, 0.1870565508848551485217621, 0.8853209468390957680903598, 0.8290583863021736216247076, 0.7050802551022034031027798, 0.5326962297259261307164520, 0.3381673205085403850889112, 0.1511107696236852365671492, 0.9775206135612875018911745, 0.9212580530243653554255223, 0.7972799218243951369035945, 0.6248958964481178645172667, 0.4303669872307321188897259, 0.2433104363458769703679639, 0.0921996667221917338008147];
const C: [f64; 21] = [-0.0562625605369221464656522, 0.0101408028300636299864818, -0.2365032522738145114532321, -0.0035758977292516175949345, 0.0935376952594620658957485, -0.5891279693869841488271399, 0.0019565654099472210769006, -0.0547553868890686864408084, 0.4158812000823068616886219, -1.1362815957175395318285885, -0.0014365302363708915424460, 0.0421585277212687077072973, -0.3600995965020568122897665, 1.2501507118406910258505441, -1.8704917729329500633517991, 0.0012717903090268677492943, -0.0387603579159067703699046, 0.3609622434528459832253398, -1.4668842084004269643701553, 2.9061362593084293014237913, -2.7558127197720458314421588];
const D: [f64; 21] = [0.0562625605369221464656522, 0.0031654757181708292499905, 0.2365032522738145114532321, 0.0001780977692217433881125, 0.0457929855060279188954539, 0.5891279693869841488271399, 0.0000100202365223291272096, 0.0084318571535257015445000, 0.2535340690545692665214616, 1.1362815957175395318285885, 0.0000005637641639318207610, 0.0015297840025004658189490, 0.0978342365324440053653648, 0.8752546646840910912297246, 1.8704917729329500633517991, 0.0000000317188154017613665, 0.0002762930909826476593130, 0.0360285539837364596003871, 0.5767330002770787313544596, 2.2485887607691597933926895, 2.7558127197720458314421588];

impl Solver for ParticleIAS {
    fn step(&mut self) {
        while self.try_step().is_none() {}
        self.steps += 1;
    }
    fn stepn(&mut self, n: usize) {
        (0..n).for_each(|_| self.step());
    }
    fn stept(&mut self, t: f64) {
        unimplemented!()
    }
    fn step_untilt(&mut self, t: f64) {
        while self.time < t {
            self.step();
        }
    }
    // impl_step_untilt!();
    fn write_to(&self, writer: &mut std::io::BufWriter<dyn std::io::Write>) {
        self.system.write_to(writer).unwrap();
    }
}