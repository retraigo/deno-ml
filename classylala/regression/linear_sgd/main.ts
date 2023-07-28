// Import csv parser from the standard library
import { parse } from "https://deno.land/std@0.188.0/csv/parse.ts";
// Import helper to split dataset
import {
  Sliceable,
  useSplit,
} from "https://deno.land/x/denouse@v0.0.12/use/array/split.ts";
// Import SGD Linear Regressor
import { SgdLinearRegressor } from "https://deno.land/x/classylala@v0.2.2/src/native.ts";
import { Matrix } from "https://deno.land/x/vectorizer@v0.0.12/mod.ts";

// Read the training dataset
const _data = Deno.readTextFileSync("datasets/Student_Performance.csv");
const data = parse(_data);
// Get the independent variables (x) and map text to numbers
const x = new Matrix(Float32Array, [data.length, 5]);
data.forEach((fl, i) => {
  x.setRow(
    i,
    [fl[0], fl[1], fl[2] === "Yes" ? 1 : 0, fl[3], fl[4]].map(Number),
  );
});
// Get dependent variables (y)
const y = data.map((fl) => Number(fl[5]));

// Split the data for training and testing
// Cast to original types because useSplit returns a weird type
const [train, test] = useSplit(
  { ratio: [7, 3], shuffle: true },
  x as Sliceable<Float32Array>,
  y,
) as [[typeof x, typeof y], [typeof x, typeof y]];

// Initialize Regressor
// Train for 10k epochs
// silent=false makes it log the RMSE every 100 epochs
// set to an optimal learning rate.
const reg = new SgdLinearRegressor({
  epochs: 10000,
  silent: false,
  learningRate: 0.001,
});

// Fit the regressor to the training data
reg.train(train[0], train[1]);

// Compute RMSE
let err = 0;
for (let i = 0; i < test[0].nRows; i += 1) {
  const y_test = reg.predict(test[0].row(i));
  err += (test[1][i] - y_test) ** 2;
}
console.log("RMSE:", Math.sqrt(err / test[0].length));
