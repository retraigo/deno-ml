import {
  Cost,
  CPU,
  DenseLayer,
  ReluLayer,
  Sequential,
  setupBackend,
  tensor1D,
  tensor2D,
} from "https://deno.land/x/netsaur@0.2.8/mod.ts";

// Import csv parser from the standard library
import { parse } from "https://deno.land/std@0.188.0/csv/parse.ts";
// Import helper to split dataset
import { useSplit } from "https://deno.land/x/denouse@v0.0.6/mod.ts";
import { AdamOptimizer } from "https://deno.land/x/netsaur@0.2.8/src/core/api/optimizer.ts";

// Read the training dataset
const _data = Deno.readTextFileSync("datasets/Student_Performance.csv");
const data = parse(_data);
// Get the independent variables (x) and map text to numbers
const x = data.map((fl) =>
  [fl[0], fl[1], fl[2] === "Yes" ? 1 : 0, fl[3], fl[4]].map(Number)
);
// Get dependent variables (y)
const y = data.map((fl) => Number(fl[5]));

// Split the data for training and testing
// Cast to original types because useSplit returns a weird type
const [train, test] = useSplit({ ratio: [7, 3], shuffle: true }, x, y) as [
  [typeof x, typeof y],
  [typeof x, typeof y],
];

// Setup the CPU backend for Netsaur
await setupBackend(CPU);

// Create a sequential neural network
const net = new Sequential({
  // Set number of minibatches to 4
  // Set size of output to 2
  size: [4, 5],

  // Disable logging during training
  silent: false,

  // Define each layer of the network
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
  ],
  // We are using Adam as the optimizer
  optimizer: AdamOptimizer(),
  // We are using MSE for finding cost
  cost: Cost.MSE,
});

const time = performance.now();

// Train the network
net.train(
  [
    {
      inputs: tensor2D(train[0]),
      outputs: tensor2D(train[1].map((x) => [x])),
    },
  ],
  // Train for 10000 epochs
  1000,
  1,
  0.02,
);

console.log(`training time: ${performance.now() - time}ms`);

// Compute RMSE
let err = 0;
for (const i in test[0]) {
  const y_test = await net.predict(tensor1D(test[0][i]));
  err += (test[1][i] - y_test.data[0]) ** 2;
//  console.log(y_test.data[0], test[1][i]);
}
console.log("RMSE:", Math.sqrt(err / test[0].length));
