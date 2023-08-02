// Import csv parser from the standard library
import { parse } from "https://deno.land/std@0.188.0/csv/parse.ts";
// Import helper to split dataset
import {
  Sliceable,
  useSplit,
} from "https://deno.land/x/denouse@v0.0.12/use/array/split.ts";
// Import classy
import {
  LossFunction,
  Model,
  Optimizer,
  solve,
  Solver,
} from "https://deno.land/x/classylala@v0.4.1/src/mod.ts";
import {
  Scheduler,
} from "https://deno.land/x/classylala@v0.4.1/src/helpers.ts";
import {
  Matrix,
} from "https://deno.land/x/vectorizer@v0.0.20/mod.ts";

// Read the training dataset
const _data = Deno.readTextFileSync("datasets/Student_Performance.csv");
const data = parse(_data);
// Get the independent variables (x) and map text to numbers
const x = new Matrix(Float64Array, [data.length, 5]);
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
console.log(x.length, y.length)
const [train, test] = useSplit(
  { ratio: [7, 3], shuffle: true },
  x as Sliceable<Float64Array>,
  y,
) as [[typeof x, typeof y], [typeof x, typeof y]];

// Fit the regressor to the training data
const [weights, bias] = solve(
  {
    epochs: 10000,
    silent: false,
    n_batches: 5,
    fit_intercept: true,
    learning_rate: 0.005,
    loss: LossFunction.MSE,
    model: Model.None,
    c: 0.00001,
    optimizer: {
      type: Optimizer.None, // Adam converges too fast
    },
    scheduler: {
      type: Scheduler.OneCycleScheduler,
      config: {
        max_lr: 0.01,
        cycle_steps: 100,
      },
    },
  },
  Solver.SGD,
  train[0],
  new Matrix(new Float64Array(train[1]), [1, train[1].length]),
);
// Compute RMSE
let err = 0;
for (let i = 0; i < test[0].nRows; i += 1) {
  const y_test = weights.dot(new Matrix(Float64Array.from(test[0].row(i)), [1, test[0].nCols])) + bias;
  err += (test[1][i] - y_test) ** 2;
}
console.log("RMSE:", Math.sqrt(err / test[0].length));
