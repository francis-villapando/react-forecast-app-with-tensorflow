import React, { useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";

export default function InventoryPredictor() {
  const [prediction, setPrediction] = useState([]);
  const [products, setProducts] = useState ([]);
  useEffect(() => {
    fetch("https://mrjson.com/api?count=%7Bindex%7C1%7D&product={lorem|1,3}&stock={number|1,100}&avgSales={number|1,100}&leadTime={number|1,7}&rows=100")
    .then(res => res.json())
    .then(data => setProducts(data))
  }, []);

  // Example training data (stock, avgSales, leadTime)
  const trainingData = products.length> 0
  ? tf.tensor2d(products.map(p => [p.stock, p.avgSales, p.leadTime]))
  : null; // creates a 2D tensor (matrix) from the provided data.

  // Labels: 1 = reorder, 0 = don't reorder
  const outputData =
  products.length > 0
    ? tf.tensor2d(
        products.map(p =>
          p.stock < p.avgSales * p.leadTime ? [1] : [0]
        )
      )
    : null;
  const handlePredict = async () => {
    if (!trainingData || !outputData) return;
    
    // 1. Create model
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [3], units: 8, activation: "relu" }));
    model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

    // 2. Compile model
    model.compile({
      optimizer: "adam",
      loss: "binaryCrossentropy",
      metrics: ["accuracy"],
    });

    // 3. Train model
    await model.fit(trainingData, outputData, {
      epochs: 200,
      shuffle: true,
    });

    // 4. Predict a new produc
    const predictions = [];
    for (let p of products) {
      const input = tf.tensor2d([[p.stock, p.avgSales, p.leadTime]]);
      const result = model.predict(input);
      const value = (await result.data())[0];
      predictions.push(value > 0.5 ? "Reorder" : "No Reorder");
    }
    setPrediction(predictions);
  };

  return (
    <div style={{ padding: 20 }}>
      <h2>Inventory Reorder Predictor</h2>
      <button onClick={handlePredict}>Predict</button>

      <table border="1" style={{ marginTop: 20, width: "100%", textAlign: "left" }}>
        <thead>
          <tr>
            <th>#</th>
            <th>Product</th>
            <th>Inventory</th>
            <th>Average Sales</th>
            <th>Lead Time</th>
            <th>Prediction</th>
          </tr>
        </thead>
        <tbody>
          {products.map((p, idx) => (
            <tr key={idx}>
              <td>{p.count}</td>
              <td>{p.product}</td>
              <td>{p.stock}</td>
              <td>{p.avgSales}</td>
              <td>{p.leadTime}</td>
              <td>{prediction[idx] || "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
