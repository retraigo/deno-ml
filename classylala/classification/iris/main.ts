/**
 * Fisher,R. A.. (1988). Iris. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76.
 * The dataset contains a set of records under 5 attributes -
 * Petal Length, Petal Width, Sepal Length, Sepal width and Class(Species).
 * For demonstration with Logistic Regression, only two classes are used.
 */

// Import csv parser from the standard library
import { parse } from "https://deno.land/std@0.188.0/csv/parse.ts";
// Import helper to split dataset
import {
  Sliceable,
  useSplit,
} from "https://deno.land/x/denouse@v0.0.12/use/array/split.ts";
// Import Logistic Regressor
import {
  LossFunction,
  Model,
  Optimizer,
  solve,
  Solver,
} from "https://deno.land/x/classylala@v0.4.1/src/mod.ts";
import {
  accuracyScore,
  ConfusionMatrix,
  precisionScore,
  Scheduler,
  sensitivityScore,
  sigmoid,
  specificityScore,
} from "https://deno.land/x/classylala@v0.4.1/src/helpers.ts";
// Import CountVectorizer and TfIdf Transformer to convert text into tf-idf features
import {
  Matrix,
} from "https://deno.land/x/vectorizer@v0.0.20/mod.ts";
// Import helpers for metrics


// Define classes
const ymap = ["Setosa", "Versicolor"];

// Read the training dataset
const _data = Deno.readTextFileSync("datasets/iris.csv");
const data = parse(_data);

// Get the predictors (x) and classes (y)
const x = new Matrix(Float64Array, [data.length, 4]);
data.forEach((fl, i) => x.setRow(i, fl.slice(0, 4).map(Number)));
const y = new Matrix(new Float64Array(data.map((fl) => ymap.indexOf(fl[4]))), [data.length]);

const [train, test] = useSplit(
  { ratio: [7, 3], shuffle: true },
  x as Sliceable<typeof x.data>,
  y as Sliceable<typeof x.data>,
) as unknown as [
  [typeof x, typeof y],
  [typeof x, typeof y],
];

const [weights] = solve(
  {
    epochs: 1000,
    model: Model.Logit,
    silent: false,
    loss: LossFunction.BinCrossEntropy,
    n_batches: 5,
    optimizer: {
      type: Optimizer.None,
    },
    scheduler: {
      type: Scheduler.DecayScheduler,
      config: {
        rate: 0.99,
        step_size: 50,
      },
    },
  },
  Solver.SGD,
  train[0],
  train[1],
);

// Test for accuracy
console.log("Training Complete");
// Check Metrics
let [tp, fn, fp, tn] = [0, 0, 0, 0];
for (let i = 0; i < test[0].nRows; i += 1) {
  const guess =
    sigmoid(weights.dot(new Matrix(new Float64Array(test[0].row(i)), [1]))) >
        0.5
      ? 1
      : 0;

  if (guess === 0 && test[1].item(i, 0) === 0) tn += 1;
  else if (guess === 0 && test[1].item(i, 0) === 1) fn += 1;
  else if (guess === 1 && test[1].item(i, 0) === 0) fp += 1;
  else tp += 1;
}

const cMatrix = new ConfusionMatrix([tp, fn, fp, tn]);

console.log("Confusion Matrix: ", cMatrix);
console.log("Accuracy: ", `${accuracyScore(cMatrix) * 100}%`);
console.log("Precision: ", `${precisionScore(cMatrix) * 100}%`);
console.log("Sensitivity / Recall: ", `${sensitivityScore(cMatrix) * 100}%`);
console.log("Specificity: ", `${specificityScore(cMatrix) * 100}%`);