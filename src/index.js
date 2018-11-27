'use strict';

const cospi = require('math-cospi');

/**
 * Evaluates a Chebyshev polynomial approximation (sum(c_i * T_i) - c_0 / 2).
 * @param {number[]} c The coefficients of the approximation. c[i] corresponds to the Chebyshev polynomial T_i.
 * @param {number} t The evaluation parameter between -1 and 1.
 * @param {number} a (Optional) The lower value of the interval for which the polynomial is defined. Default is -1.
 * @param {number} b (Optional) The upper value of the interval for which the polynomial is defined. Default is 1.
 * @returns {number} The evaluation of the approximation at the parameter value.
 */
function chebyshevEval (c, t, a = -1, b = 1) {
  const n = c.length;
  const x = (t - (0.5 * (b + a))) / (0.5 * (b - a));
  const alpha = 2 * x;
  const beta = -1;
  let y1 = 0;
  let y2 = 0;
  for (let k = n - 1; k >= 1; --k) {
    const tmp = y1;
    y1 = alpha * y1 + beta * y2 + c[k];
    y2 = tmp;
  }
  return x * y1 - y2 + 0.5 * c[0];
}

/**
 * Computes the ith coefficient of the Chebyshev approximation.
 * @param {number} j The (T_i)th coefficient.
 * @param {number} n The number of terms in the Chebyshev approximation.
 * @param {number} a The lower value of the interval for which the polynomial is defined.
 * @param {number} b The upper value of the interval for which the polynomial is defined.
 * @param {Function} f The evaluation function.
 */
function chebyshevFitCoefficient (j, n, a, b, f) {
  let sum = 0;
  const h = 0.5;
  for (let i = 0; i < n; ++i) {
    const x = cospi((i + h) / n);
    const fx = f(x * (h * (b - a)) + (h * (b + a)));
    sum += fx * cospi(j * (i + h) / n);
  }
  return 2 * sum / n;
}

/**
 * Calculates a polynomial approximation using Chebyshev polynomials. Coefficients are such that f(x) = sum(c_i * T_i) - c_0 / 2.
 * @param {Function} F Function to approximate that maps reals to reals.
 * @param {number} n The integer degree of the polynomial for which the function is to be approximated.
 * @param {number} a (Optional) The lower value of the interval for which the polynomial is defined. Default is -1.
 * @param {number} b (Optional) The upper value of the interval for which the polynomial is defined. Default is 1.
 * @returns {number[]} The array of coefficients for the Chebyshev polynomials. c[i] corresponds to T_i(x).
 */
function chebyshevFit (F, N, a = -1, b = 1) {
  const n = Math.floor(N);
  const c = new Float64Array(n);
  for (let j = 0; j < n; ++j) {
    c[j] = chebyshevFitCoefficient(j, n, a, b, F);
  }
  return c;
}

/**
 * Computes the derivative of a Chebyshev-approximated function (f(x) = sum(c_i * T_i) - c_0 / 2).
 * @param {number[]} c The array of Chebyshev polynomial coefficients as given by the Chebyshev fit algorithm.
 * @param {number} a (Optional) The lower value of the interval for which the polynomial is defined. Default is -1.
 * @param {number} b (Optional) The upper value of the interval for which the polynomial is defined. Default is 1.
 * @returns {number[]} The derivative of the approximation in terms of the Chebyshev polynomials in terms of Chebyshev polynomials. c[i] corresponds to T_i(x).
 */
function chebyshevDerivative (c, a = -1, b = 1) {
  const n = c.length;
  const d = new Float64Array(n);
  d[n - 1] = 0;
  d[n - 2] = 2 * (n - 1) * c[n - 1];
  for (let k = n - 3; k >= 0; --k) {
    d[k] = d[k + 2] + 2 * (k + 1) * c[k + 1];
  }
  for (let k = 0; k < n; ++k) {
    d[k] *= 2 / (b - a);
  }
  return d;
}

/**
 * Computes the antiderivative of a Chebyshev-approximated function (f(x) = sum(c_i * T_i) - c_0 / 2).
 * @param {number[]} c The array of Chebyshev polynomial coefficients as given by the Chebyshev fit algorithm.
 * @param {number} a (Optional) The lower value of the interval for which the polynomial is defined. Default is -1.
 * @param {number} b (Optional) The upper value of the interval for which the polynomial is defined. Default is 1.
 * @returns {number[]} The array of Chebyshev polynomial coefficients for the integral approximation.
 */
function chebyshevAntiderivative (c, a = -1, b = 1) {
  const n = c.length;
  const i = new Float64Array(n);
  let sum = 0;
  let fac = 1;
  const C = 0.25 * (b - a);
  for (let j = 1; j <= n - 2; ++j) {
    i[j] = C * (c[j - 1] - c[j + 1]) / j;
    sum += fac * i[j];
    fac = -fac;
  }
  i[n - 1] = C * c[n - 2] / (n - 1);
  sum += fac * i[n - 1];
  i[0] = 2 * sum;
  return i;
}

/**
 * Adaptively fits a Chebyshev function approximation so that the max error is 1e-15.
 * @param {Function} F Function to approximate that maps reals to reals.
 * @param {number} a (Optional) The lower value of the interval for which the polynomial is defined. Default is -1.
 * @param {number} b (Optional) The upper value of the interval for which the polynomial is defined. Default is 1.
 * @returns {number[]} The array of coefficients for the Chebyshev polynomials. c[i] corresponds to T_i(x).
 */
function chebyshevAdaptiveFit (F, a = -1, b = 1) {
  const TOL = 1e-15;  // just above machine epsilon, near the upper limit of precision
  let N = 4;
  let c;

  // use doubling strategy for general sizing
  do {
    N *= 2;
    // TODO: implement rougher estimation method for speed (e.g. Gauss-Lobatto)
    c = chebyshevFit(F, N + 1, a, b);
  } while (Math.abs(c[N]) > TOL);

  for (let k = N / 2; k <= N; ++k) {
    if (Math.abs(c[k - 1]) < TOL && Math.abs(c[k]) < TOL) {
      N = k + 1;
      break;
    }
  }
  return chebyshevFit(F, N, a, b);
}

/**
 * Determines if the function approximation is an odd function.
 * @param {number[]} C The array of Chebyshev series coefficients.
 * @returns {boolean} True if the approximation is odd, false if not.
 */
function isOddFunction(C) {
  const n = C.length;
  let isOdd = true;
  for (let i = 0; i < n; i += 2) {
    isOdd &= Math.abs(C[i]) < 1e-15;
  }
  return isOdd;
} 

/**
 * Computes the definite integral of the given Chebyshev approximation using Clenshaw-Curtis quadrature.
 * @param {Function} c The Chebyshev approximation to the function.
 * @param {number} a (Optional) The lower value of the interval for which the polynomial is defined. Default is -1.
 * @param {number} b (Optional) The upper value of the interval for which the polynomial is defined. Default is 1.
 * @returns {number} The definite integral of the function approximation from a to b.
 */
function chebyshevIntegrate (c, a = -1, b = 1) {
  const N = c.length;
  if (isOddFunction(c)) {
    // do Clenshaw-Curtis quadrature
    let sum = 0;
    // construct f(x) / x

    for (let k = Math.floorN / 2; k < N / 2; ++k) {
      const d = (2 * k + 1) * (2 * k - 1);
      sum -= c[2 * k] / d;
    }
    return;
  } else {
    // do Clenshaw-Curtis quadrature
    let sum = c[0] / 2;
    for (let k = 1; k < N / 2; ++k) {
      const d = (2 * k + 1) * (2 * k - 1);
      sum -= c[2 * k] / d;
    }
    return sum * (b - a);
  }

  
}

const Chebyshev = {
  evaluate: chebyshevEval,
  approximate: chebyshevFit,
  getAdaptiveApproximation: chebyshevAdaptiveFit,
  getDerivative: chebyshevDerivative,
  getIntegral: chebyshevAntiderivative,
  integrate: chebyshevIntegrate
};
module.exports = Chebyshev;
