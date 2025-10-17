import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import { buildEmbeddings, queryInsights, debugChroma } from "./ragService.js";

dotenv.config();

const app = express();          // ✅ Create the Express app first
app.use(cors());                // ✅ Then enable CORS
app.use(express.json());

// Serve static files (for index.html)
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
app.use(express.static(__dirname));

// Endpoints
app.post("/ai/insight", async (req, res) => {
  const { query } = req.body;
  const insight = await queryInsights(query);
  res.json({ insight });
});

app.post("/ai/embed", async (req, res) => {
  await buildEmbeddings();
  res.send("✅ Embeddings built.");
});

app.get("/ai/debug", async (req, res) => {
  const data = await debugChroma();
  res.json(data);
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
