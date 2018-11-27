'use strict';
/* global describe it */

const Chebyshev = require('../src/index.js');
const expect = require('chai').expect;

describe('Chebyshev Polynomial Functions', () => {
  describe('Chebyshev Approximation Evaluation', () => {
    it('Gauss Error Function', () => {
      const F = (x) => Math.pow(Math.E, -x * x);
      const C = Chebyshev.approximate(F, 5);
      const t = 0.25;
      const gx = Chebyshev.evaluate(C, t);
      const fx = F(t);

      // error check; no odd terms in this expansion.
      const C1 = Chebyshev.approximate(F, 7);
      const error = Math.abs(gx - fx);
      const errorBound = Math.abs(C1[6]);
      console.log(`Error: ${error}`);
      console.log(`Bound: ${errorBound}`);
      expect(error < errorBound).to.be.true; // eslint-disable-line
    });
    it('Damped Oscillations', () => {
      const F = (x) => Math.cos(9 * x) / (1 + Math.pow(Math.E, 9 * x));
      const C = Chebyshev.approximate(F, 6);
      const t = 0.25;
      const gx = Chebyshev.evaluate(C, t);
      const fx = F(t);

      // error check; no odd terms in this expansion.
      const C1 = Chebyshev.approximate(F, 7);
      const error = Math.abs(gx - fx);
      const errorBound = Math.abs(C1[6]);
      console.log(`Error: ${error}`);
      console.log(`Bound: ${errorBound}`);
      expect(error < errorBound).to.be.true; // eslint-disable-line
    });
  });
  describe('Chebyshev Definite Integration', () => {
    it('Gauss Error Function', () => {
      const F = (x) => Math.pow(Math.E, -x * x);
      const erfApprox = Chebyshev.getAdaptiveApproximation(F, -1, 1);
      const erfApproxValue = Chebyshev.integrate(erfApprox, -1, 1);
      const erf1 = 1.493648265624854050798934872263; // sqrt(pi) * erf(1) from Mathematica

      const diff = Math.abs(erfApproxValue - erf1);
      console.log(`Error: ${diff}`);
      expect(diff < 1e-15).to.be.true; // eslint-disable-line
      // console.log(`Error Bound = ${errorBound}`);
    });
  });
  describe('Chebyshev Antiderivative', () => {
    it('Gauss Error Function', () => {
      const F = (x) => Math.pow(Math.E, -x * x);
      const f = Chebyshev.approximate(F, 5);

      const x = 1 / Math.sqrt(2);
      const fx = Chebyshev.evaluate(f, x);
      const df = Chebyshev.getDerivative(f);
      const ffx = Chebyshev.integrate(df, -1, x);

      console.log(`F(x) = ${fx}`);
      console.log(`int{dF(x)}_{-1}^{x} = ${ffx}`);
    });
  });
});
