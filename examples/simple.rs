// Import the benchmark library
use benchfun::{single::Ackley, Bounded, SingleObjective};

fn main() {
    // Print some info about the ackley function
    println!("Minmimum: {:?}", Ackley::MINIMUM);
    println!("Minmizer: {:?}", Ackley::minimizer(5));
    println!("Minmizer: {:?}", Ackley::BOUNDS);
}
