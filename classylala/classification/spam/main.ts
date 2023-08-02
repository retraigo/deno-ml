/**
 * Almeida,Tiago and Hidalgo,Jos. (2012). SMS Spam Collection. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.
 * The dataset contains text messages that are marked spam and ham (not spam).
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
  CountVectorizer,
  Matrix,
  TfIdfTransformer,
} from "https://deno.land/x/vectorizer@v0.0.20/mod.ts";
// Import helpers for metrics

// Define classes
const ymap = ["spam", "ham"];

// Read the training dataset
const _data = Deno.readTextFileSync("datasets/spam.csv");
const data = parse(_data);

// Get the predictors (messages)
const x = data.map((msg) => msg[1]);

// Get the classes
const y = new Matrix(
  new Float64Array(data.map((msg) => ymap.indexOf(msg[0]))),
  [data.length, 1],
);
const [train, test] = useSplit(
  { ratio: [5, 5], shuffle: true },
  x as Sliceable<Float64Array>,
  y as Sliceable<Float64Array>,
) as [
  [typeof x, typeof y],
  [typeof x, typeof y],
];
// Vectorize the text messages
const vec = new CountVectorizer({ stopWords: "english", lowercase: true }).fit(
  train[0],
);
const x_vec = vec.transform(train[0]);
const tfidf = new TfIdfTransformer().fit(x_vec);

const x_tfidf = tfidf.transform(x_vec);
const [weights] = solve(
  {
    epochs: 1000,
    model: Model.Logit,
    silent: false,
    loss: LossFunction.BinCrossEntropy,
    n_batches: 5,
    optimizer: {
      type: Optimizer.Adam,
      config: {
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
      },
    },
    scheduler: {
      type: Scheduler.OneCycleScheduler,
      config: {
        max_lr: 0.01,
        cycle_steps: 50,
      },
    },
  },
  Solver.Minibatch,
  x_tfidf,
  train[1],
);

const xvec_test = tfidf.transform(vec.transform(test[0]));

// Test for accuracy
console.log("Training Complete");
// Check Metrics
let [tp, fn, fp, tn] = [0, 0, 0, 0];
for (let i = 0; i < xvec_test.nRows; i += 1) {
  const guess =
    sigmoid(weights.dot(new Matrix(new Float64Array(xvec_test.row(i)), [1]))) >
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
