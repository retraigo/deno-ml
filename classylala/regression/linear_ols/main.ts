import { useNormalArray } from "https://deno.land/x/denouse@v0.0.6/mod.ts";

import { LinearRegressor } from "https://deno.land/x/classylala@v0.2.0/src/native.ts";

/** Generate an array of 1M random numbers around 107 */
const x = useNormalArray(1e7, 107, 5);
/** Compute another array using a slope and intercept */
const y = x.map((n) => 8 * n + 60);

/** 
 * Initialize Linear Regressor 
 * LinearRegressor uses Ordinary Least Squares
 */
const reg = new LinearRegressor();

/** Train the model */
reg.train(x, y);

/** Compute another array for testing */
const x_test = useNormalArray(1e7, 120, 1);

/** Calculate RMSE */
let err = 0;
x_test.forEach((n) => {
  const y_test = reg.predict(n);
  err += (8 * n + 60 - y_test) ** 2;
});
console.log("RMSE:", Math.sqrt(err / x_test.length));
console.log("Slope:", reg.slope)
console.log("Intercept:", reg.intercept)
console.log("R2 Score:", reg.r2)
