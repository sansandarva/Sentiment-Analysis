import { useState } from "react";
import axios from "axios";
import { Pie } from "react-chartjs-2";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";
import "./App.css";

ChartJS.register(ArcElement, Tooltip, Legend);

function App() {
  const [text, setText] = useState("");
  const [sentiment, setSentiment] = useState("");
  const [file, setFile] = useState(null);
  const [stats, setStats] = useState(null);

  const analyzeText = async () => {
    const res = await axios.post("http://127.0.0.1:8000/predict", {
      text: text,
    });
    setSentiment(res.data.sentiment);
  };

  const uploadFile = async () => {
    const formData = new FormData();
    formData.append("file", file);

    const res = await axios.post(
      "http://127.0.0.1:8000/analyze-file",
      formData
    );

    setStats(res.data);
  };

  return (
    <div className="container">
      <h1>🚀 Sentiment Analyzer</h1>

      {/* TEXT */}
      <div className="card">
        <textarea
          placeholder="Enter text..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <button onClick={analyzeText}>Analyze</button>

        {sentiment && <p>Sentiment: {sentiment}</p>}
      </div>

      {/* FILE */}
      <div className="card">
        <input type="file" onChange={(e) => setFile(e.target.files[0])} />
        <button onClick={uploadFile}>Upload & Analyze</button>

        {stats && (
          <>
            <p>Positive: {stats.positive}</p>
            <p>Negative: {stats.negative}</p>

            <Pie
  data={{
    labels: ["Positive", "Negative"],
    datasets: [
      {
        data: [stats.positive, stats.negative],

        // ✅ ADD COLORS HERE
        backgroundColor: [
          "#22c55e", // green
          "#ef4444", // red
        ],

        borderColor: [
          "#16a34a",
          "#dc2626",
        ],

        borderWidth: 1,
        hoverOffset: 10,
      },
    ],
  }}
/>
          </>
        )}
      </div>
    </div>
  );
}

export default App;
