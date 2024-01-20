Deno.env.set("RUST_BACKTRACE", "1");
import {
  AdamOptimizer,
  Cost,
  CPU,
  DenseLayer,
  OneCycle,
  ReluLayer,
  Sequential,
  setupBackend,
  SoftmaxLayer,
  tensor,
  tensor1D,
  tensor2D,
} from "https://deno.land/x/netsaur@0.2.15/mod.ts";

import { parse } from "https://deno.land/std@0.204.0/csv/parse.ts";

// Import helpers for metrics
import {
  ClassificationReport,
  // Split the dataset
  useSplit,
  CategoricalEncoder,
} from "https://deno.land/x/vectorizer@v0.3.4/mod.ts";

// Read the training dataset
const _data = Deno.readTextFileSync("datasets/iris.csv");
const data = parse(_data);

// Get the predictors (x) and targets (y)
const x = data.map((fl) => fl.slice(0, 4).map(Number));
const y = data.map((fl) => fl[4]);

const encoder = new CategoricalEncoder();

const encodedY = encoder.fit(y).transform<"f32">(y, "f32");

// Split the dataset for training and testing
const [[x_train, y_train], [x_test, y_test]] = useSplit(
  { ratio: [7, 3], shuffle: true },
  x,
  encodedY
);

// Setup the CPU backend for Netsaur
await setupBackend(CPU);

// Create a sequential neural network
const net = new Sequential({
  // Set number of minibatches to 4
  // Set size of output to 4
  size: [6, 4],

  // Disable logging during training
  silent: false,

  // Define each layer of the network
  layers: [
    // A dense layer with 16 neurons
    DenseLayer({ size: [16] }),
    // A ReLu activation layer
    ReluLayer(),
    // A dense layer with 3 neurons
    DenseLayer({ size: [3] }),
    // A Softmax activation layer
    SoftmaxLayer(),
  ],
  optimizer: AdamOptimizer(),
  // We are using CrossEntropy for finding cost
  cost: Cost.CrossEntropy,
  scheduler: OneCycle({ max_rate: 0.05, step_size: 50 }),
});

const time = performance.now();

// Train the network
net.train(
  [
    {
      inputs: tensor2D(x_train),
      outputs: tensor(y_train.data, [y_train.nRows, y_train.nCols]),
    },
  ],
  // Train for 300 epochs
  300,
  1,
  0.02
);

console.log(`training time: ${performance.now() - time}ms`);

// Calculate metrics
const res = await Promise.all(
  x_test.map((input) => net.predict(tensor1D(input)))
);
const y1 = res.map((x) => x.data.indexOf(Math.max(...x.data)));
const y0 = Array.from(y_test.rows()).map((x) => x.indexOf(1));

console.log(y1.map((x, i) => [y0[i], x]));
const cMatrix = new ClassificationReport(y0, y1);
console.log(cMatrix);
console.log(
  "Total Accuracy: ",
  y1.filter((x, i) => x === y0[i]).length / y1.length
);
