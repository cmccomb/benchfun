//! This module contains single-objective functions

use crate::{
    Bounded, Constrained, FixedDimensional, NDimensional, SingleObjective, UnBounded, UnConstrained,
};

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
        x.into_iter().map(|xi| xi.powi(2)).sum::<f64>()
    }

    /// This function returns the minimizer (argument that will return the global minimum
    fn minimizer(n: usize) -> Vec<f64> {
        vec![0.0; n]
    }
}

#[cfg(test)]
mod sphere_tests {
    use super::{NDimensional, SingleObjective, Sphere as F};

    #[test]
    fn low_d() {
        F::check_minimizer(F::LOW_D);
    }

    #[test]
    fn high_d() {
        F::check_minimizer(F::HIGH_D);
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
    #[allow(clippy::cast_precision_loss)]
    fn f(x: Vec<f64>) -> f64 {
        let a = 10.0;
        let n = x.len();
        let mut fx = a * (n as f64);

        for xi in x {
            fx += xi.powi(2) - a * (2.0 * xi * std::f64::consts::PI).cos();
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
    use super::{NDimensional, Rastrigin as F, SingleObjective};

    #[test]
    fn low_d() {
        F::check_minimizer(F::LOW_D);
    }

    #[test]
    fn high_d() {
        F::check_minimizer(F::HIGH_D);
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
        for i in 0..(n - 1) {
            fx += 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2);
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
    use super::{NDimensional, Rosenbrock as F, SingleObjective};

    #[test]
    fn low_d() {
        F::check_minimizer(F::LOW_D);
    }

    #[test]
    fn high_d() {
        F::check_minimizer(F::HIGH_D);
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
    #[allow(clippy::cast_precision_loss)]
    fn f(x: Vec<f64>) -> f64 {
        let n = x.len();
        let mut fx = 0.0;
        let mut square_sum = 0.0;
        let mut cosine_sum = 0.0;
        for xi in x {
            square_sum += xi.powi(2);
            cosine_sum += (2.0 * std::f64::consts::PI * xi).cos();
        }
        fx -= 20.0 * (-0.2 * (0.5 * square_sum).sqrt()).exp();
        fx -= (cosine_sum / (n as f64)).exp();
        fx + std::f64::consts::E + 20.0
    }

    /// This function returns the minimizer (argument that will return the global minimum
    fn minimizer(n: usize) -> Vec<f64> {
        vec![0.0; n]
    }
}

#[cfg(test)]
mod ackley_tests {
    use super::{Ackley as F, NDimensional, SingleObjective};

    #[test]
    fn low_d() {
        F::check_minimizer(F::LOW_D);
    }

    #[test]
    fn high_d() {
        F::check_minimizer(F::HIGH_D);
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
        let (square_sum, prod) = x
            .into_iter()
            .fold((0.0, 1.0), |(sum, prod), xi| (sum + xi.powi(2), prod * xi));
        0.26 * square_sum - 0.48 * prod
    }

    /// This function returns the minimizer (argument that will return the global minimum
    fn minimizer(n: usize) -> Vec<f64> {
        vec![0.0; n]
    }
}

#[cfg(test)]
mod matyas_tests {
    use super::{Matyas as F, NDimensional, SingleObjective};

    #[test]
    fn low_d() {
        F::check_minimizer(F::LOW_D);
    }

    #[test]
    fn high_d() {
        F::check_minimizer(F::HIGH_D);
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
    #[allow(clippy::cast_precision_loss)]
    fn f(x: Vec<f64>) -> f64 {
        let mut cosine_prod = 1.0;
        let mut square_sum = 0.0;
        for (i, xi) in x.iter().enumerate() {
            square_sum += xi.powi(2);
            cosine_prod *= (xi / ((i + 1) as f64).sqrt()).cos();
        }
        1.0 + square_sum / 4000.0 - cosine_prod
    }

    /// This function returns the minimizer (argument that will return the global minimum
    fn minimizer(n: usize) -> Vec<f64> {
        vec![0.0; n]
    }
}

#[cfg(test)]
mod griewank_tests {
    use super::{Griewank as F, NDimensional, SingleObjective};

    #[test]
    fn low_d() {
        F::check_minimizer(F::LOW_D);
    }

    #[test]
    fn high_d() {
        F::check_minimizer(F::HIGH_D);
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
    /// The global minimum equals the first coordinate of the minimizer.
    const MINIMUM: f64 = -5.0;

    /// Function for evaluating
    fn f(x: Vec<f64>) -> f64 {
        let d = 1.0;
        let alpha = 0.0;
        let mut square_sum = 0.0;
        for xi in x.iter().skip(1) {
            square_sum += xi.powi(2);
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
    use super::{NDimensional, Ridge as F, SingleObjective};

    #[test]
    fn low_d() {
        F::check_minimizer(F::LOW_D);
    }

    #[test]
    fn high_d() {
        F::check_minimizer(F::HIGH_D);
    }

    #[test]
    fn f_matches_minimum() {
        let minimizer = F::minimizer(F::LOW_D);

        assert_eq!(F::f(minimizer), F::MINIMUM);
    }

    #[test]
    fn minimizer_respects_dimension() {
        let dimension = 5;
        let minimizer = F::minimizer(dimension);

        assert_eq!(minimizer.len(), dimension);
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
    #[allow(clippy::cast_precision_loss)]
    fn f(x: Vec<f64>) -> f64 {
        let mut square_sum: f64 = 0.0;
        let mut sum_ixi: f64 = 0.0;
        for (i, xi) in x.iter().enumerate() {
            square_sum += xi.powi(2);
            sum_ixi += 0.5 * xi * (i as f64);
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
    use super::{NDimensional, SingleObjective, Zakharov as F};

    #[test]
    fn low_d() {
        F::check_minimizer(F::LOW_D);
    }

    #[test]
    fn high_d() {
        F::check_minimizer(F::HIGH_D);
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
        let mut square_sum = 0.0;
        for xi in x {
            square_sum += xi.powi(2);
        }
        1.0 - (2.0 * std::f64::consts::PI * square_sum.sqrt()).cos() + 0.1 * square_sum.sqrt()
    }

    /// This function returns the minimizer (argument that will return the global minimum
    fn minimizer(n: usize) -> Vec<f64> {
        vec![0.0; n]
    }
}

#[cfg(test)]
mod salomon_tests {
    use super::{NDimensional, Salomon as F, SingleObjective};

    #[test]
    fn low_d() {
        F::check_minimizer(F::LOW_D);
    }

    #[test]
    fn high_d() {
        F::check_minimizer(F::HIGH_D);
    }
}

/// This is a constrained version of the Rosenbrock function.
///
/// The function is borrowed from [here](https://en.wikipedia.org/wiki/Test_functions_for_optimization).
/// This function is specifically 2 dimensional, and has a feasible region that looks like this:
///
///
/// ![](https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/ConstrTestFunc04.png/664px-ConstrTestFunc04.png)
pub struct RosenbrockConst1 {}

impl UnBounded for RosenbrockConst1 {}

impl Constrained for RosenbrockConst1 {
    const NH: usize = 0;
    const NG: usize = 2;

    fn equality_constraints(_x: Vec<f64>) -> Vec<f64> {
        vec![0.0; Self::NH]
    }

    fn inequality_constraints(x: Vec<f64>) -> Vec<f64> {
        let mut fx: Vec<f64> = vec![0.0; Self::NG];
        fx[0] = (x[0] - 1.0).powi(3) - x[1] + 1.0;
        fx[1] = x[0] + x[1] - 2.0;
        fx
    }
}

impl FixedDimensional for RosenbrockConst1 {
    const D: usize = 2;
}

impl SingleObjective for RosenbrockConst1 {
    /// The global minimum is constant and zero
    const MINIMUM: f64 = 0.0;

    /// Function for evaluating
    fn f(x: Vec<f64>) -> f64 {
        Self::check_input(x.clone());
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
    }

    /// This function returns the minimizer (argument that will return the global minimum
    fn minimizer(_n: usize) -> Vec<f64> {
        vec![1.0; 2]
    }
}

#[cfg(test)]
mod rosenbrock_const1_tests {
    use super::{FixedDimensional, RosenbrockConst1 as F, SingleObjective};

    #[test]
    fn low_d() {
        F::check_minimizer(F::D);
    }
}

/// This is a constrained version of the Rosenbrock function.
///
/// The function is borrowed from [here](https://en.wikipedia.org/wiki/Test_functions_for_optimization).
/// This function is specifically 2 dimensional, and has a feasible region that looks like this:
///
///
/// ![](https://upload.wikimedia.org/wikipedia/commons/3/38/ConstrTestFunc03.png)
pub struct RosenbrockConst2 {}

impl UnBounded for RosenbrockConst2 {}

impl Constrained for RosenbrockConst2 {
    const NH: usize = 0;
    const NG: usize = 1;

    fn equality_constraints(_x: Vec<f64>) -> Vec<f64> {
        vec![0.0; Self::NH]
    }

    fn inequality_constraints(x: Vec<f64>) -> Vec<f64> {
        let mut fx: Vec<f64> = vec![0.0; Self::NG];
        fx[0] = x[0].powi(2) + x[1].powi(2) - 2.0;
        fx
    }
}

impl FixedDimensional for RosenbrockConst2 {
    const D: usize = 2;
}

impl SingleObjective for RosenbrockConst2 {
    /// The global minimum is constant and zero
    const MINIMUM: f64 = 0.0;

    /// Function for evaluating
    fn f(x: Vec<f64>) -> f64 {
        Self::check_input(x.clone());
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
    }

    /// This function returns the minimizer (argument that will return the global minimum
    fn minimizer(_n: usize) -> Vec<f64> {
        vec![1.0; 2]
    }
}

#[cfg(test)]
mod rosenbrock_const2_tests {
    use super::{FixedDimensional, RosenbrockConst2 as F, SingleObjective};

    #[test]
    fn low_d() {
        F::check_minimizer(F::D);
    }
}
