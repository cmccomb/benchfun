#![warn(missing_docs)]
#![warn(missing_doc_code_examples)]

//! The `benchmark_functions` crate provides functionality for several functions that are commonly
//! used to benchmark new optimization algorithms. More specifically, function is part of a struct
//! that contains the objective function as well as other important information (bounds of the
//! canonical problem, the known minimum value, and a function that returns the global minimizer.

/// These are constants used for testing
const LOW_D: usize = 2;
const HIGH_D: usize = 137;

/// This is a trait that ensures consistent implementation of single objective benchmark functions
pub trait SingleObjective {
    /// The global minimum is constant and zero
    const MINIMUM: f64;

    /// Function for evaluating the objective function
    fn f(x: Vec<f64>) -> f64;

    /// This function returns the minimizer (argument that will return the global minimum)
    fn minimizer(n: usize) -> Vec<f64>;

    /// This function is used for testing, and checks the correctness of the minimizer
    fn check_minimizer(d: usize) {
        assert_eq!(Self::f(Self::minimizer(d)), Self::MINIMUM)
    }
}

/// This is a trait that ensures consistent implementation of multi-objective benchmark functions
pub trait MultiObjective {
    /// Function for evaluating the set of objective functions
    fn f(x: Vec<f64>) -> Vec<f64>;
}

/// This is a trait that ensures consistent implementation of bounded benchmark functions
pub trait Bounded {
    /// The bounds of the canonical optimization problem
    const BOUNDS: (f64, f64);

    /// Function to check bounds
    fn in_bounds(x: Vec<f64>) -> bool {
        let mut in_bounds = true;
        for element in x {
            if (element < Self::BOUNDS.0) || (element > Self::BOUNDS.1) {
                in_bounds = false;
                break;
            }
        }
        in_bounds
    }
}

/// This is a trait that ensures consistent implementation of unbounded benchmark functions
pub trait UnBounded {
    /// The bounds of the canonical optimization problem
    const BOUNDS: (f64, f64) = (f64::INFINITY, f64::INFINITY);

    /// Function to check bounds
    fn in_bounds(_x: Vec<f64>) -> bool {
        true
    }
}

/// This is a trait that ensures consistent implementation of constrained benchmark functions
pub trait Constrained {
    /// This constant indicates a constrained function
    const CONSTRAINED: bool = true;

    /// This constant indicates the number of equality functions
    const NH: usize;

    /// This constant indicates the number of inequality functions
    const NG: usize;

    /// This function returns the value of equality constraints
    fn equality_constraints(x: Vec<f64>) -> Vec<f64>;

    /// This function returns the value of inequality constraints
    fn inequality_constraints(x: Vec<f64>) -> Vec<f64>;

    /// This is an alias for the equality constraint function
    fn h(x: Vec<f64>) -> Vec<f64> {
        Self::equality_constraints(x)
    }

    /// This is an alias for the inequality constraint function
    fn g(x: Vec<f64>) -> Vec<f64> {
        Self::inequality_constraints(x)
    }
}

/// This is a trait that ensure consistent implementation of unconstrained benchmark functions
pub trait UnConstrained {
    const CONSTRAINED: bool = false;
}

/// This is a trait that ensures consistent implementation of N-dimensional benchmark functions
pub trait NDimensional {
    /// This is a constant containing the correct dimensionality for the function
    const D: usize = usize::MAX;
}

/// This is a trait that ensures consistent implementation of benchmark functions with fixed dimensionality
pub trait FixedDimensional {
    /// This is a constant containing the correct dimensionality for the function
    const D: usize;

    /// This function is used to check inputs
    fn check_input(x: Vec<f64>){
        if x.len() != Self::D {
            panic!("A vector with size {} was used with a function of dimensionality {}.", x.len(), Self::D);
        }
    }
}

/// This is the Sphere function.
///
/// The function is borrowed from [here](https://en.wikipedia.org/wiki/Test_functions_for_optimization).
/// Although the function accepts a vector with an arbitrary number of inputs, this is what it looks
/// like in 2D:
///
/// ![](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Sphere_function_in_3D.pdf/page1-800px-Sphere_function_in_3D.pdf.jpg)
pub struct Sphere {}

impl NDimensional for Sphere {}
impl UnConstrained for Sphere {}
impl UnBounded for Sphere {}

impl SingleObjective for Sphere {
    /// The global minimum is constant and zero
    const MINIMUM: f64 = 0.0;

    /// Function for evaluating
    fn f(x: Vec<f64>) -> f64 {
        let mut f = 0f64;
        for i in 0..x.len() {
            f -= x[i] * x[i];
        }
        f
    }

    /// This function returns the minimizer (argument that will return the global minimum
    fn minimizer(n: usize) -> Vec<f64> {
        vec![0.0; n]
    }
}

#[cfg(test)]
mod sphere_tests {
    use super::{Sphere as F, Bounded, SingleObjective, LOW_D, HIGH_D};

    #[test]
    fn low_d() {
        F::check_minimizer(LOW_D)
    }

    #[test]
    fn high_d() {
        F::check_minimizer(HIGH_D)
    }
}

/// This is the Rastrigin function.
///
/// The function is borrowed from [here](https://en.wikipedia.org/wiki/Test_functions_for_optimization).
/// Although the function accepts a vector with an arbitrary number of inputs, this is what it looks
/// like in 2D:
///
/// ![](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Rastrigin_function.png/800px-Rastrigin_function.png)
pub struct Rastrigin {}

impl NDimensional for Rastrigin {}
impl UnConstrained for Rastrigin {}

impl Bounded for Rastrigin {
    /// The bounds of the canonical sphere optimization problem are infinite.
    const BOUNDS: (f64, f64) = (-5.12, 5.12);
}

impl SingleObjective for Rastrigin {
    /// The global minimum is constant and zero
    const MINIMUM: f64 = 0.0;

    /// Function for evaluating
    fn f(x: Vec<f64>) -> f64 {
        let a = 10.0;
        let n = x.len() ;
        let mut fx = a*(n as f64);

        for i in 0..n {
            fx += x[i].powi(2) - a*(2.0*x[i]*std::f64::consts::PI).cos();
        }
        fx
    }

    /// This function returns the minimizer (argument that will return the global minimum
    fn minimizer(n: usize) -> Vec<f64> {
        vec![0.0; n]
    }
}

#[cfg(test)]
mod rastrigin_tests {
    use super::{Rastrigin as F, SingleObjective, LOW_D, HIGH_D};

    #[test]
    fn low_d() {
        F::check_minimizer(LOW_D)
    }

    #[test]
    fn high_d() {
        F::check_minimizer(HIGH_D)
    }
}

/// This is the Rosenbrock function.
///
/// The function is borrowed from [here](https://en.wikipedia.org/wiki/Test_functions_for_optimization).
/// Although the function accepts a vector with an arbitrary number of inputs, this is what it looks
/// like in 2D:
///
/// ![](https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Rosenbrock%27s_function_in_3D.pdf/page1-800px-Rosenbrock%27s_function_in_3D.pdf.jpg)
pub struct Rosenbrock {}

impl NDimensional for Rosenbrock {}
impl UnConstrained for Rosenbrock {}

impl Bounded for Rosenbrock {
    /// The bounds of the canonical sphere optimization problem are infinite.
    const BOUNDS: (f64, f64) = (-5.0, 10.0);
}

impl SingleObjective for Rosenbrock {
    /// The global minimum is constant and zero
    const MINIMUM: f64 = 0.0;

    /// Function for evaluating
    fn f(x: Vec<f64>) -> f64 {
        let n = x.len();
        let mut fx = 0.0;
        for i in 0..(n-1) {
            fx += 100.0*(x[i+1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2);
        }
        fx
    }

    /// This function returns the minimizer (argument that will return the global minimum
    fn minimizer(n: usize) -> Vec<f64> {
        vec![1.0; n]
    }
}

#[cfg(test)]
mod rosenbrock_tests {
    use super::{Rosenbrock as F, SingleObjective, LOW_D, HIGH_D};

    #[test]
    fn low_d() {
        F::check_minimizer(LOW_D)
    }

    #[test]
    fn high_d() {
        F::check_minimizer(HIGH_D)
    }
}

/// This is the Ackley function.
///
/// The function is borrowed from [here](https://en.wikipedia.org/wiki/Test_functions_for_optimization).
/// Although the function accepts a vector with an arbitrary number of inputs, this is what it looks
/// like in 2D:
///
/// ![](https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/Ackley%27s_function.pdf/page1-800px-Ackley%27s_function.pdf.jpg)
pub struct Ackley {}

impl NDimensional for Ackley {}
impl UnConstrained for Ackley {}

impl Bounded for Ackley {
    /// The bounds of the canonical sphere optimization problem are infinite.
    const BOUNDS: (f64, f64) = (-5.0, 5.0);
}

impl SingleObjective for Ackley {
    /// The global minimum is constant and zero
    const MINIMUM: f64 = 0.0;

    /// Function for evaluating
    fn f(x: Vec<f64>) -> f64 {
        let n=x.len();
        let mut fx = 0.0;
        let mut square_sum = 0.0;
        let mut cosine_sum = 0.0;
        for i in 0..n {
            square_sum += x[i].powi(2);
            cosine_sum += (2.0*std::f64::consts::PI*x[i]).cos();
        }
        fx += -20.0*(-0.2*(0.5*square_sum).sqrt()).exp();
        fx -= (cosine_sum/(n as f64)).exp();
        fx + std::f64::consts::E + 20.0
    }

    /// This function returns the minimizer (argument that will return the global minimum
    fn minimizer(n: usize) -> Vec<f64> {
        vec![0.0; n]
    }
}

#[cfg(test)]
mod ackley_tests {
    use super::{Ackley as F, SingleObjective, LOW_D, HIGH_D};

    #[test]
    fn low_d() {
        F::check_minimizer(LOW_D)
    }

    #[test]
    fn high_d() {
        F::check_minimizer(HIGH_D)
    }
}

/// This is the Matyas function.
///
/// The function is borrowed from [here](https://en.wikipedia.org/wiki/Test_functions_for_optimization).
/// Although the function accepts a vector with an arbitrary number of inputs, this is what it looks
/// like in 2D:
///
/// ![](https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Matyas_function.pdf/page1-800px-Matyas_function.pdf.jpg)
pub struct Matyas {}

impl NDimensional for Matyas {}
impl UnConstrained for Matyas {}

impl Bounded for Matyas {
    /// The bounds of the canonical sphere optimization problem are infinite.
    const BOUNDS: (f64, f64) = (-10.0, 10.0);
}

impl SingleObjective for Matyas {
    /// The global minimum is constant and zero
    const MINIMUM: f64 = 0.0;

    /// Function for evaluating
    fn f(x: Vec<f64>) -> f64 {
        let n=x.len();
        let mut square_sum = 0.0;
        let mut prod = 1.0;
        for i in 0..n {
            square_sum += x[i].powi(2);
            prod *= x[i];
        }
        0.26*square_sum - 0.48*prod
    }

    /// This function returns the minimizer (argument that will return the global minimum
    fn minimizer(n: usize) -> Vec<f64> {
        vec![0.0; n]
    }
}

#[cfg(test)]
mod matyas_tests {
    use super::{Matyas as F, SingleObjective, LOW_D, HIGH_D};

    #[test]
    fn low_d() {
        F::check_minimizer(LOW_D)
    }

    #[test]
    fn high_d() {
        F::check_minimizer(HIGH_D)
    }
}

/// This is the Griewank function.
///
/// The function is borrowed from [here](http://benchmarkfcns.xyz/benchmarkfcns/griewankfcn.html).
/// Although the function accepts a vector with an arbitrary number of inputs, this is what it looks
/// like in 2D:
///
/// ![](http://benchmarkfcns.xyz/benchmarkfcns/plots/griewankfcn_10_0.png)
pub struct Griewank {}

impl NDimensional for Griewank {}
impl UnConstrained for Griewank {}

impl Bounded for Griewank {
    /// The bounds of the canonical sphere optimization problem are infinite.
    const BOUNDS: (f64, f64) = (-600.0, 600.0);
}

impl SingleObjective for Griewank {
    /// The global minimum is constant and zero
    const MINIMUM: f64 = 0.0;

    /// Function for evaluating
    fn f(x: Vec<f64>) -> f64 {
        let n=x.len();
        let mut cosine_prod = 1.0;
        let mut square_sum = 0.0;
        for i in 0..n {
            square_sum += x[i].powi(2);
            cosine_prod *= (x[i]/((i+1) as f64).sqrt()).cos();
        }
        1.0 + square_sum/4000.0 - cosine_prod
    }

    /// This function returns the minimizer (argument that will return the global minimum
    fn minimizer(n: usize) -> Vec<f64> {
        vec![0.0; n]
    }
}

#[cfg(test)]
mod griewank_tests {
    use super::{Griewank as F, SingleObjective, LOW_D, HIGH_D};

    #[test]
    fn low_d() {
        F::check_minimizer(LOW_D)
    }

    #[test]
    fn high_d() {
        F::check_minimizer(HIGH_D)
    }
}

/// This is the Ridge function.
///
/// The function is borrowed from [here](http://benchmarkfcns.xyz/benchmarkfcns/ridgefcn.html).
/// Although the function accepts a vector with an arbitrary number of inputs, this is what it looks
/// like in 2D:
///
/// ![](http://benchmarkfcns.xyz/benchmarkfcns/plots/ridgefcn.png)
pub struct Ridge {}

impl NDimensional for Ridge {}
impl UnConstrained for Ridge {}

impl Bounded for Ridge {
    /// The bounds of the canonical sphere optimization problem are infinite.
    const BOUNDS: (f64, f64) = (-5.0, 5.0);
}

impl SingleObjective for Ridge {
    /// The global minimum is constant and zero
    const MINIMUM: f64 = -5.0;

    /// Function for evaluating
    fn f(x: Vec<f64>) -> f64 {
        let n=x.len();
        let d = 1.0;
        let alpha = 0.0;
        let mut square_sum = 0.0;
        for i in 1..n {
            square_sum += x[i].powi(2);
        }
        -1.0 + x[0] + d * square_sum.powf(alpha)
    }

    /// This function returns the minimizer (argument that will return the global minimum
    fn minimizer(n: usize) -> Vec<f64> {
        let mut v = vec![0.0; n];
        v[0] = -5.0;
        v
    }
}

#[cfg(test)]
mod ridge_tests {
    use super::{Ridge as F, SingleObjective, LOW_D, HIGH_D};

    #[test]
    fn low_d() {
        F::check_minimizer(LOW_D)
    }

    #[test]
    fn high_d() {
        F::check_minimizer(HIGH_D)
    }
}

/// This is the Zakharov function.
///
/// The function is borrowed from [here](http://benchmarkfcns.xyz/benchmarkfcns/zakharov.html).
/// Although the function accepts a vector with an arbitrary number of inputs, this is what it looks
/// like in 2D:
///
/// ![](http://benchmarkfcns.xyz/benchmarkfcns/plots/zakharovfcn.png)
pub struct Zakharov {}

impl NDimensional for Zakharov {}
impl UnConstrained for Zakharov {}

impl Bounded for Zakharov {
    /// The bounds of the canonical sphere optimization problem are infinite.
    const BOUNDS: (f64, f64) = (-5.0, 10.0);
}

impl SingleObjective for Zakharov {
    /// The global minimum is constant and zero
    const MINIMUM: f64 = 0.0;

    /// Function for evaluating
    fn f(x: Vec<f64>) -> f64 {
        let n=x.len();
        let mut square_sum = 0.0;
        let mut sum_ixi = 0.0;
        for i in 0..n {
            square_sum += x[i].powi(2);
            sum_ixi += 0.5*x[i]*(i as f64);
        }
        square_sum + sum_ixi.powi(2) + sum_ixi.powi(4)
    }

    /// This function returns the minimizer (argument that will return the global minimum
    fn minimizer(n: usize) -> Vec<f64> {
        vec![0.0; n]
    }
}

#[cfg(test)]
mod zakharov_tests {
    use super::{Zakharov as F, SingleObjective, LOW_D, HIGH_D};

    #[test]
    fn low_d() {
        F::check_minimizer(LOW_D)
    }

    #[test]
    fn high_d() {
        F::check_minimizer(HIGH_D)
    }
}

/// This is the Salomon function.
///
/// The function is borrowed from [here](http://benchmarkfcns.xyz/benchmarkfcns/salomonfcn.html).
/// Although the function accepts a vector with an arbitrary number of inputs, this is what it looks
/// like in 2D:
///
/// ![](http://benchmarkfcns.xyz/benchmarkfcns/plots/salomonfcn.png)
pub struct Salomon {}

impl NDimensional for Salomon {}
impl UnConstrained for Salomon {}

impl Bounded for Salomon {
    /// The bounds of the canonical sphere optimization problem are infinite.
    const BOUNDS: (f64, f64) = (-100.0, 100.0);
}

impl SingleObjective for Salomon {
    /// The global minimum is constant and zero
    const MINIMUM: f64 = 0.0;

    /// Function for evaluating
    fn f(x: Vec<f64>) -> f64 {
        let n=x.len();
        let mut square_sum = 0.0;
        for i in 0..n {
            square_sum += x[i].powi(2);
        }
        1.0 - (2.0*std::f64::consts::PI*square_sum.sqrt()).cos() + 0.1*square_sum.sqrt()
    }

    /// This function returns the minimizer (argument that will return the global minimum
    fn minimizer(n: usize) -> Vec<f64> {
        vec![0.0; n]
    }
}

#[cfg(test)]
mod salomon_tests {
    use super::{Salomon as F, SingleObjective, LOW_D, HIGH_D};

    #[test]
    fn low_d() {
        F::check_minimizer(LOW_D)
    }

    #[test]
    fn high_d() {
        F::check_minimizer(HIGH_D)
    }
}

/// This is the Chankong-Haimes function.
///
/// The function is borrowed from [here](https://en.wikipedia.org/wiki/Test_functions_for_optimization).
/// This function is specificaly 2 dimensional, and has a Pareto fron that looks like this:
///
/// ![](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Chakong_and_Haimes_function.pdf/page1-796px-Chakong_and_Haimes_function.pdf.jpg)
pub struct ChankongHaimes {}

impl FixedDimensional for ChankongHaimes {
    const D: usize = 2;
}

impl Constrained for ChankongHaimes {
    const NH: usize = 0;
    const NG: usize = 2;

    fn equality_constraints(_x: Vec<f64>) -> Vec<f64> {
        vec![0.0; Self::NH]
    }

    fn inequality_constraints(x: Vec<f64>) -> Vec<f64> {
        let mut fx: Vec<f64> = vec![0.0; Self::NG];
        fx[0] = x[0].powi(2) + x[1].powi(2) - 225.0;
        fx[1] = x[0] - 3.0*x[1] + 10.0;
        fx
    }
}

impl MultiObjective for ChankongHaimes {
    fn f(x: Vec<f64>) -> Vec<f64> {
        Self::check_input(x.clone());
        let mut fx: Vec<f64> = vec![0.0; Self::D];
        fx[0] = 2.0 + (x[0] - 2.0).powi(2) - (x[1] - 1.0).powi(2);
        fx[1] = 9.0*x[0] - (x[1] - 1.0).powi(2);
        fx
    }
}

#[cfg(test)]
mod chankong_haimes_tests {
    use super::*;

    #[test]
    fn check_zero() {
        let x = vec![0.0; ChankongHaimes::D];
        ChankongHaimes::f(x.clone());
        ChankongHaimes::equality_constraints(x.clone());
        ChankongHaimes::inequality_constraints(x);
        assert!(true);
    }

    #[test]
    fn check_one() {
        let x = vec![1.0; ChankongHaimes::D];
        ChankongHaimes::f(x.clone());
        ChankongHaimes::equality_constraints(x.clone());
        ChankongHaimes::inequality_constraints(x);
        assert!(true);
    }
}
