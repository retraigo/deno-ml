/**
 * Fisher,R. A.. (1988). Iris. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76.
 * The dataset contains a set of records under 5 attributes -
 * Petal Length, Petal Width, Sepal Length, Sepal width and Class(Species).
 * For demonstration with Logistic Regression, only two classes are used.
 */

// Import csv parser from the standard library
import { parse } from "https://deno.land/std@0.188.0/csv/parse.ts";
// Import Logistic Regressor
import { LogisticRegressor } from "https://deno.land/x/classylala@v0.2.0/src/native.ts";
// Import helper to split dataset
import {
  accuracyScore,
  precisionScore,
  sensitivityScore,
  specificityScore,
} from "https://deno.land/x/classylala@v0.2.0/src/helpers.ts";
// Used to split the dataset
import { useSplit } from "https://deno.land/x/denouse@v0.0.6/mod.ts";

// Define classes
const ymap = ["Setosa", "Versicolor"];

// Read the training dataset
const _data = Deno.readTextFileSync("datasets/iris.csv");
const data = parse(_data);

// Train for 10000 epochs
const reg = new LogisticRegressor({ epochs: 10000, silent: true });

// Get the predictors (x) and classes (y)
const x = data.map((fl) => fl.slice(0, 4).map(Number));
const y = data.map((fl) => ymap.indexOf(fl[4]));

const [train, test] = useSplit({ ratio: [7, 3], shuffle: true }, x, y) as [
  [typeof x, typeof y],
  [typeof x, typeof y],
];

// Train the model with the training data
reg.train(train[0], train[1]);

console.log("Trained Complete");

// Check Metrics
const cMatrix = reg.confusionMatrix(test[0], test[1]);

console.log("Confusion Matrix: ", cMatrix);
console.log("Accuracy: ", `${accuracyScore(cMatrix) * 100}%`);
console.log("Precision: ", `${precisionScore(cMatrix) * 100}%`);
console.log("Sensitivity / Recall: ", `${sensitivityScore(cMatrix) * 100}%`);
console.log("Specificity: ", `${specificityScore(cMatrix) * 100}%`);
