use std::io::{Read, Write};

use crate::{particle::Particle, systems::{ParticleSystem, SystemConstants}, traits::System};

// Any change to this file should increment the version number
const ARCHIVE_VERSION: u64 = 0;

pub enum Archivable {
    ParticleSystem
}

impl From<u8> for Archivable {
    fn from(value: u8) -> Self {
        match value {
            0 => Archivable::ParticleSystem,
            _ => panic!("undefined archivable enum")
        }
    }
}

fn read_u64(reader: &mut std::io::BufReader<std::fs::File>) -> u64 {
    let mut buf64 = [0; 8];
    reader.read_exact(&mut buf64);
    u64::from_le_bytes(buf64)
}

/// Wrapper to safely obtain a buffer writer
pub fn bufferwriter_from_file(filepath: &str, appending: bool) -> Option<std::io::BufWriter<std::fs::File>> {
    if let Ok(exists) = std::path::Path::try_exists(std::path::Path::new(filepath)) {
        if !exists {
            println!("{} does not exist", filepath);
            return None
        }
        // Path exists and is readable
        let writer = std::fs::File::options().append(appending).create(true).open(filepath);
        return Some(std::io::BufWriter::new(writer.unwrap()))
    }
    println!("{} could not be verified (try_exists errored)", filepath);
    return None
}

/// Wrapper to safely obtain a reader
pub fn bufferreader_from_file(filepath: &str) -> Option<std::io::BufReader<std::fs::File>> {
    if let Ok(exists) = std::path::Path::try_exists(std::path::Path::new(filepath)) {
        if !exists {
            println!("{} does not exist", filepath);
            return None
        }
        // Path exists and is readable
        let reader = std::fs::File::open(filepath);
        return Some(std::io::BufReader::new(reader.unwrap()))
    }
    println!("{} could not be verified (try_exists errored)", filepath);
    return None
}

pub fn read_component(reader: &mut std::io::BufReader<std::fs::File>) -> Option<(Archivable, Vec<u8>)> {
    let mut component: [u8; 1] = [100]; // start as invalid value
    reader.read_exact(&mut component);
    match component[0].into() {
        Archivable::ParticleSystem => Some((component[0].into(), ParticleSystem::read_bytes(reader))),
    }
}

// todo: move the to_le_bytes stuff here so archiving can be properly tracked
// todo: add acceleration to archived particle properties

impl SystemConstants {
    fn read_bytes(reader: &mut std::io::BufReader<std::fs::File>) -> Vec<u8> {
        let mut out = [0; 4*8];
        reader.read_exact(&mut out);
        out.to_vec()
    }
}
impl Particle {
    fn read_bytes(reader: &mut std::io::BufReader<std::fs::File>) -> Vec<u8> {
        let mut out = [0; 7*8];
        reader.read_exact(&mut out);
        out.to_vec()
    }
}

impl ParticleSystem {
    pub fn write_to(&self, writer: &mut std::io::BufWriter<dyn std::io::Write>) -> std::io::Result<()> {
        // Type being written
        writer.write(&[Archivable::ParticleSystem as u8])?;
        // Version number [8 bytes] (different versions likely incompatible)
        writer.write(&ARCHIVE_VERSION.to_le_bytes())?;
        // Constants
        writer.write(&self.constants.to_le_bytes())?;
        // // Particles
        // Number of particles [8 bytes]
        writer.write(&self.num_particles().to_le_bytes())?;
        // Actual particles
        for particle in &self.particles {
            writer.write(&particle.to_le_bytes())?;
        }
        writer.flush()
    }
    pub fn read_bytes(reader: &mut std::io::BufReader<std::fs::File>) -> Vec<u8> {
        // type already read
        // read version
        let version = read_u64(reader);
        if version != ARCHIVE_VERSION {println!("Reading from an outdated archive ({}). Current version is {}. Errors may occur", version, ARCHIVE_VERSION)}
        // read constants (should maybe have a way to only save these once but fine for now)
        let mut out = SystemConstants::read_bytes(reader);
        // read number of particles
        let particle_num = read_u64(reader);
        // actually read the particles
        for _ in 0..particle_num {
            out.append(&mut Particle::read_bytes(reader));
        }
        out
    }
}