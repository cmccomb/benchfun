[![CI](https://github.com/cmccomb/benchfun/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/cmccomb/benchfun/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/benchfun.svg)](https://crates.io/crates/benchfun)
[![docs.rs](https://docs.rs/benchfun/badge.svg)](https://docs.rs/benchfun)

# About
This crate provides functionality for several functions that are commonly
used to benchmark new optimization algorithms. More specifically, function is part of a struct
that contains the objective function as well as other important information (bounds of the
canonical problem, the known minimum value, and a function that returns the global minimizer.

This crate provides access to several single- and multi-objective funtions. For exhaustive lists, check [here](single/index.html) and [here](multi/index.html), respectively.


# Example Usage
Using this crate is easy! Simply add this crate as a dependency and then import the pieces you need explicitly:
```rust
use benchfun::{single::Ackley, Bounded, SingleObjective};

fn main() {
    // Print some info about the ackley function
    println!("Minmimum: {:?}", Ackley::MINIMUM);
    println!("Minmizer: {:?}", Ackley::minimizer(5));
    println!("Minmizer: {:?}", Ackley::BOUNDS);
}
```
