// Import csv parser from the standard library
import {
  solve,
  Solver,
} from "https://deno.land/x/classylala@v0.4.1/src/mod.ts";
import {
  Matrix,
} from "https://deno.land/x/vectorizer@v0.0.20/mod.ts";
import { useNormalArray } from "https://deno.land/x/denouse@v0.0.6/mod.ts";

/** Generate an array of 1M random numbers around 107 */
const x = useNormalArray(1e7, 24, 5);
/** Compute another array using a slope and intercept */
const y = x.map((n) => 8 * n + 60);

/** Train the model */

const [weights, bias] = solve(
  {
    silent: false,
    fit_intercept: true,
  },
  Solver.OLS,
  new Matrix(Float64Array.from(x), [x.length, 1]),
  new Matrix(new Float64Array(y), [y.length]),
);

/** Compute another array for testing */
const x_test = useNormalArray(1e7, 64, 1);

/** Calculate RMSE */
let err = 0;
x_test.forEach((n) => {
  const y_test = weights.dot(new Matrix(Float64Array.from([n]), [1, 1])) + bias;
  err += (8 * n + 60 - y_test) ** 2;
});
console.log("RMSE:", Math.sqrt(err / x_test.length));
console.log("Weights:", weights.data);
console.log("Intercept:", bias);
