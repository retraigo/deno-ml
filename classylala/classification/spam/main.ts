/**
 * Almeida,Tiago and Hidalgo,Jos. (2012). SMS Spam Collection. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.
 * The dataset contains text messages that are marked spam and ham (not spam).
 */

// Import csv parser from the standard library
import { parse } from "https://deno.land/std@0.188.0/csv/parse.ts";
// Import helper to split dataset
import { useSplit } from "https://deno.land/x/denouse@v0.0.6/mod.ts";
// Import Logistic Regressor
import { LogisticRegressor } from "https://deno.land/x/classylala@v0.2.0/src/native.ts";
// Import CountVectorizer and TfIdf Transformer to convert text into tf-idf features
import {
  CountVectorizer,
  TfIdfTransformer,
} from "https://deno.land/x/vectorizer@v0.0.4/mod.ts";
// Import helpers for metrics
import {
  accuracyScore,
  precisionScore,
  sensitivityScore,
  specificityScore,
} from "https://deno.land/x/classylala@v0.2.0/src/helpers.ts";

// Define classes
const ymap = ["spam", "ham"];

// Read the training dataset
const _data = Deno.readTextFileSync("datasets/spam.csv");
const data = parse(_data);

// Get the predictors (messages)
const x = data.map((msg) => msg[1]);

// Get the classes
const y = data.map((msg) => ymap.indexOf(msg[0]));

const [train, test] = useSplit({ ratio: [7, 3], shuffle: true }, x, y) as [[typeof x, typeof y], [typeof x, typeof y]];

// Vectorize the text messages
const vec = new CountVectorizer({ stopWords: "english", lowercase: true }).fit(
  train[0],
);
const x_vec = vec.transform(train[0]);
const tfidf = new TfIdfTransformer().fit(x_vec);

const x_tfidf = tfidf.transform(x_vec);

// Initialize logistic regressor and train for 100 epochs
const reg = new LogisticRegressor({ epochs: 200, silent: false });
reg.train(x_tfidf, train[1]);

const xvec_test = tfidf.transform(vec.transform(test[0]));

// Test for accuracy
console.log("Trained Complete");

// Check Metrics
const cMatrix = reg.confusionMatrix(xvec_test, test[1]);

console.log("Confusion Matrix: ", cMatrix);
console.log("Accuracy: ", `${accuracyScore(cMatrix) * 100}%`);
console.log("Precision: ", `${precisionScore(cMatrix) * 100}%`);
console.log("Sensitivity / Recall: ", `${sensitivityScore(cMatrix) * 100}%`);
console.log("Specificity: ", `${specificityScore(cMatrix) * 100}%`);
