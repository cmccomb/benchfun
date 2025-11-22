//! This module contains multi-objective functions

use crate::{Bounded, Constrained, FixedDimensional, MultiObjective, NDimensional, UnConstrained};

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

impl Bounded for ChankongHaimes {
    const BOUNDS: (f64, f64) = (-20.0, 20.0);
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
        fx[1] = x[0] - 3.0 * x[1] + 10.0;
        fx
    }
}

impl MultiObjective for ChankongHaimes {
    const NF: usize = 2;

    fn f(x: Vec<f64>) -> Vec<f64> {
        Self::check_input(x.clone());
        let mut fx: Vec<f64> = vec![0.0; Self::NF];
        fx[0] = 2.0 + (x[0] - 2.0).powi(2) - (x[1] - 1.0).powi(2);
        fx[1] = 9.0 * x[0] - (x[1] - 1.0).powi(2);
        fx
    }
}

#[cfg(test)]
mod chankong_haimes_tests {
    use super::{ChankongHaimes as F, Constrained, FixedDimensional, MultiObjective};

    #[test]
    fn check_zero() {
        let x = vec![0.0; F::D];
        let fx = F::f(x.clone());
        let h = F::h(x.clone());
        let g = F::g(x);

        assert_eq!(fx.len(), F::NF);
        assert_eq!(h.len(), F::NH);
        assert_eq!(g.len(), F::NG);
    }

    #[test]
    fn check_one() {
        let x = vec![1.0; F::D];
        let fx = F::f(x.clone());
        let h = F::h(x.clone());
        let g = F::g(x);

        assert_eq!(fx.len(), F::NF);
        assert_eq!(h.len(), F::NH);
        assert_eq!(g.len(), F::NG);
    }
}

/// This is the Fonseca-Fleming function.
///
/// The function is borrowed from [here](https://en.wikipedia.org/wiki/Test_functions_for_optimization).
/// Although the function accepts a vector with an arbitrary number of inputs, this is what the
/// Pareto front looks like in 2D:
///
/// ![](https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/Fonseca_and_Fleming_function.pdf/page1-796px-Fonseca_and_Fleming_function.pdf.jpg)
pub struct FonsecaFlemming {}

impl NDimensional for FonsecaFlemming {}
impl UnConstrained for FonsecaFlemming {}

impl Bounded for FonsecaFlemming {
    const BOUNDS: (f64, f64) = (-4.0, 4.0);
}

impl MultiObjective for FonsecaFlemming {
    const NF: usize = 2;

    #[allow(clippy::cast_precision_loss)]
    fn f(x: Vec<f64>) -> Vec<f64> {
        let mut fx: Vec<f64> = vec![0.0; Self::NF];
        let n = x.len();
        let mut sumxminus: f64 = 0.0;
        let mut sumxplus: f64 = 0.0;
        let nsqrt = (n as f64).sqrt();
        for xi in x {
            sumxminus += (xi - 1.0 / nsqrt).powi(2);
            sumxplus += (xi + 1.0 / nsqrt).powi(2);
        }
        fx[0] = 1.0 - (-sumxminus).exp();
        fx[1] = 1.0 - (-sumxplus).exp();
        fx
    }
}

#[cfg(test)]
mod flemingfonseca_tests {
    use super::{FonsecaFlemming as F, MultiObjective, NDimensional};

    #[test]
    fn check_zero() {
        let fx_low = F::f(vec![0.0; F::LOW_D]);
        let fx_high = F::f(vec![0.0; F::HIGH_D]);

        assert_eq!(fx_low.len(), F::NF);
        assert_eq!(fx_high.len(), F::NF);
    }

    #[test]
    fn check_one() {
        let fx_low = F::f(vec![1.0; F::LOW_D]);
        let fx_high = F::f(vec![1.0; F::HIGH_D]);

        assert_eq!(fx_low.len(), F::NF);
        assert_eq!(fx_high.len(), F::NF);
    }
}

/// This is the Viennet function.
///
/// The function is borrowed from [here](https://en.wikipedia.org/wiki/Test_functions_for_optimization).
/// This function is specifically 2 dimensional, and has a Pareto fron that looks like this:
///
/// ![](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f2/Viennet_function.pdf/page1-796px-Viennet_function.pdf.jpg)
pub struct Viennet {}

impl UnConstrained for Viennet {}

impl FixedDimensional for Viennet {
    const D: usize = 2;
}

impl Bounded for Viennet {
    const BOUNDS: (f64, f64) = (-3.0, 3.0);
}

impl MultiObjective for Viennet {
    const NF: usize = 3;

    fn f(x: Vec<f64>) -> Vec<f64> {
        Self::check_input(x.clone());
        let mut fx: Vec<f64> = vec![0.0; Self::NF];
        let x2y2 = x[0].powi(2) + x[1].powi(2);
        fx[0] = 0.5 * x2y2 + x2y2.sin();
        fx[1] = (3.0 * x[0] - 2.0 * x[1] + 4.0).powi(2) / 8.0
            + (x[0] - x[1] + 1.0).powi(2) / 27.0
            + 15.0;
        fx[2] = 1.0 / (x2y2 + 1.0) - 1.1 * (-x2y2).exp();
        fx
    }
}

#[cfg(test)]
mod viennet_tests {
    use super::{FixedDimensional, MultiObjective, Viennet as F};
    use crate::Bounded;

    #[test]
    fn check_zero() {
        let x = vec![0.0; F::D];
        let fx = F::f(x.clone());

        assert_eq!(fx.len(), F::NF);
        assert!(F::in_bounds(x));
    }

    #[test]
    fn check_one() {
        let x = vec![0.0; F::D];
        let fx = F::f(x.clone());

        assert_eq!(fx.len(), F::NF);
        assert!(F::in_bounds(x));
    }
}
