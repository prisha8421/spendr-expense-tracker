import { ChromaClient } from "chromadb";
import { ChatOpenAI } from "@langchain/openai";
import { Pool } from "pg";
import dotenv from "dotenv";
import { pipeline } from "@xenova/transformers";

dotenv.config();

// --- PostgreSQL setup ---
const pool = new Pool({
  user: process.env.DB_USER,
  host: process.env.DB_HOST,
  database: process.env.DB_NAME,
  password: process.env.DB_PASSWORD,
  port: process.env.DB_PORT,
});

// --- Chroma ---
const chroma = new ChromaClient({ path: process.env.CHROMA_HOST || "http://localhost:8000" });
let collection;
let model;

// --- Local embeddings ---
let embedPipeline;
async function initEmbed() {
  if (!embedPipeline) {
    embedPipeline = await pipeline(
      "feature-extraction",
      process.env.EMBED_MODEL || "Xenova/all-MiniLM-L6-v2",
      { local: true }
    );
  }
}

async function getEmbedding(text) {
  await initEmbed();
  const output = await embedPipeline(text, { pooling: "mean", normalize: true });
  if (Array.isArray(output)) return output[0];
  else if (output.data) return Array.from(output.data);
  else throw new Error("Unexpected embedding output format");
}

// --- Initialize Chroma + LLM ---
async function initRAG() {
  if (!collection) {
    try {
      await chroma.createCollection({ name: "expenses" });
    } catch (e) {
      console.log("ℹ️ Collection may already exist:", e.message);
    }
    collection = await chroma.getCollection({ name: "expenses" });
    console.log("✅ Chroma collection ready");
  }

  if (!model) {
    model = new ChatOpenAI({
      apiKey: process.env.OPENROUTER_KEY,
      model: process.env.LLM_MODEL || "mistralai/mistral-7b-instruct:free",
      maxTokens: 1500,
      configuration: { baseURL: "https://openrouter.ai/api/v1" },
    });
    console.log("✅ Mistral 7B (via OpenRouter) initialized");
  }
}

// --- Build embeddings ---
export async function buildEmbeddings() {
  await initRAG();

  const res = await pool.query("SELECT id, note, amount, expense_date FROM expenses");
  const expenses = res.rows;

  console.log(`🧾 Found ${expenses.length} expenses to embed...`);

  for (const exp of expenses) {
    try {
      const desc = `Expense: ${exp.note}, Amount: $${exp.amount}, Date: ${exp.expense_date}`;
      const vector = await getEmbedding(desc);

      await collection.add({
        ids: [exp.id.toString()],
        embeddings: [vector],
        metadatas: [
          { 
            text: `Expense: ${exp.note}, Amount: $${exp.amount}, Date: ${exp.expense_date}`,
            note: exp.note,
            amount: Number(exp.amount),
            date: exp.expense_date
          },
        ],
      });
      
    } catch (err) {
      console.error(`⚠️ Failed to embed expense ID ${exp.id}:`, err.message);
    }
  }

  console.log("✅ Expense embeddings stored in Chroma!");
}

// --- Query insights ---
export async function queryInsights(userQuery) {
  await initRAG();

  // 1️⃣ Retrieve top 3 similar expenses
  const queryVec = await getEmbedding(userQuery);
  const results = await collection.query({
    queryEmbeddings: [queryVec],
    nResults: 3,
  });

  const expenses = [];
  if (results?.metadatas && Array.isArray(results.metadatas)) {
    results.metadatas.flat().forEach((m) => {
      const amt = Number(m.amount);
      if (!isNaN(amt)) {
        expenses.push({
          note: m.note,
          amount: amt,
          date: m.date,
        });
      }
    });
  }

  if (expenses.length === 0) return "⚠️ No valid expenses found for the query.";

  // 2️⃣ Compute totals & month-over-month trends in JS
  let total = 0;
  const monthMap = {};
  expenses.forEach((exp) => {
    total += exp.amount;
    const month = new Date(exp.date).toLocaleString("default", { month: "long", year: "numeric" });
    monthMap[month] = (monthMap[month] || 0) + exp.amount;
  });

  let trendSummary = "";
  const months = Object.keys(monthMap).sort();
  if (months.length >= 2) {
    const lastMonth = months[months.length - 2];
    const thisMonth = months[months.length - 1];
    const diff = monthMap[thisMonth] - monthMap[lastMonth];
    trendSummary = `Compared to ${lastMonth}, your spending in ${thisMonth} is ${diff >= 0 ? "higher" : "lower"} by $${Math.abs(diff).toFixed(2)}.`;
  }

  const structuredContext = `
Top expenses:
${expenses.map((e) => `- ${e.note}: $${e.amount} on ${e.date}`).join("\n")}

Total spending: $${total.toFixed(2)}
${trendSummary}
`;

  // 3️⃣ Build friendly prompt for LLM
  const prompt = `
You are Spendr, a friendly personal finance AI assistant.
Here is a summary of some recent expenses:

${structuredContext}

Please provide a clear, warm, and helpful insight, including:
- Notable spending patterns
- Month-over-month trend (if available)
- 1-2 actionable suggestions to improve spending
`;

  // 4️⃣ Send to LLM
  const messages = [{ role: "user", content: prompt }];
  let response = await model.invoke(messages);

  let insight = response?.content?.trim() || "";
  insight = insight.replace(/<s>|\[OUT\]/g, "").trim();

  // Retry once if empty
  if (!insight || insight.length < 10) {
    const simplePrompt = `Write a short, friendly summary of these expenses:\n${structuredContext}`;
    const retryResp = await model.invoke([{ role: "user", content: simplePrompt }]);
    insight = retryResp?.content?.trim() || insight;
    insight = insight.replace(/<s>|\[OUT\]/g, "").trim();
  }

  return insight || "⚠️ The model didn’t return meaningful insights.";
}

// --- Debug endpoint ---
export async function debugChroma() {
  await initRAG();
  return await collection.get({ limit: 10 });
}
