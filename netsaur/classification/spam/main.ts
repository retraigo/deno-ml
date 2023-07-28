import {
  Cost,
  CPU,
  DenseLayer,
  ReluLayer,
  Sequential,
  setupBackend,
  SigmoidLayer,
  tensor,
  tensor2D,
} from "https://deno.land/x/netsaur@0.2.8/mod.ts";

import { parse } from "https://deno.land/std@0.188.0/csv/parse.ts";

// Used to split the dataset
import { useSplit } from "https://deno.land/x/denouse@v0.0.6/mod.ts";

// Import helpers for metrics
import {
  accuracyScore,
  ConfusionMatrix,
  precisionScore,
  sensitivityScore,
  specificityScore,
} from "https://deno.land/x/classylala@v0.2.0/src/helpers.ts";

import {
  CountVectorizer,
  TfIdfTransformer,
} from "https://deno.land/x/vectorizer@v0.0.8/mod.ts";
import {
  AdamOptimizer,
} from "https://deno.land/x/netsaur@0.2.8/src/core/api/optimizer.ts";

// Define classes
const ymap = ["spam", "ham"];

// Read the training dataset
const _data = Deno.readTextFileSync("datasets/spam.csv");
const data = parse(_data);

// Get the predictors (messages)
const x = data.map((msg) => msg[1]);

// Get the classes
const y = data.map((msg) => ymap.indexOf(msg[0]));

// Split the dataset for training and testing
const [train, test] = useSplit({ ratio: [8, 2], shuffle: false }, x, y) as [
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

// Setup the CPU backend for Netsaur
await setupBackend(CPU);

// Create a sequential neural network
const net = new Sequential({
  // Set number of minibatches to 4
  // Set size of output to 2
  size: [4, x_tfidf.nCols],
  /**
   * Defines the layers of a neural network in the XOR function example.
   * The neural network has two input neurons and one output neuron.
   * The layers are defined as follows:
   * - A dense layer with 3 neurons.
   * - sigmoid activation layer.
   * - A dense layer with 1 neuron.
   * -A sigmoid activation layer.
   */
  layers: [
    // A dense layer with 256 neurons
    DenseLayer({ size: [256] }),
    // A relu activation layer
    ReluLayer(),
    // A dense layer with 8 neurons
    DenseLayer({ size: [8] }),
    // A relu activation layer
    ReluLayer(),
    // A dense layer with 8 neurons
    DenseLayer({ size: [8] }),
    // A relu activation layer
    ReluLayer(),
    // A dense layer with 1 neuron
    DenseLayer({ size: [1] }),
    // A sigmoid activation layer
    SigmoidLayer(),
  ],

  // We are using MSE for finding cost
  cost: Cost.MSE,
  optimizer: AdamOptimizer(),
});

const time = performance.now();
const inputs = tensor(x_tfidf.data, x_tfidf.shape);
// Train the network
net.train(
  [
    {
      inputs: inputs,
      outputs: tensor2D(train[1].map((x) => [x])),
    },
  ],
  // Train for 10000 epochs
  500,
  5,
  0.01,
);

console.log(`training time: ${performance.now() - time}ms`);

const x_vec_test = tfidf.transform(vec.transform(test[0]));

// Calculate metrics
let [tp, fn, fp, tn] = [0, 0, 0, 0];
for (let i = 0; i < test[0].length; i += 1) {
  const pred = await net.predict(tensor(x_vec_test.row(i), [x_vec_test.nCols]));
  //  console.log(`Should be ${test[1][i]} got ${pred.data}`)
  const res = (pred).data[0] <
      0.5
    ? 0
    : 1;
  if (res === 1 && test[1][i] == 1) tp += 1;
  if (res === 0 && test[1][i] == 1) fn += 1;
  if (res === 1 && test[1][i] == 0) fp += 1;
  if (res === 0 && test[1][i] == 0) tn += 1;
}

const cMatrix = new ConfusionMatrix([tp, fn, fp, tn]);
console.log("Confusion Matrix: ", cMatrix);
console.log("Accuracy: ", `${accuracyScore(cMatrix) * 100}%`);
console.log("Precision: ", `${precisionScore(cMatrix) * 100}%`);
console.log("Sensitivity / Recall: ", `${sensitivityScore(cMatrix) * 100}%`);
console.log("Specificity: ", `${specificityScore(cMatrix) * 100}%`);
