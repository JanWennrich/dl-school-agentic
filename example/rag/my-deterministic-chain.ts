#!/usr/bin/env node
/**
 * Deterministic RAG over my own index — fixed pipeline: always retrieve → prompt → LLM.
 *
 * Copy of deterministic-chain.ts, retrieving from the namespace seeded with
 * `npm run seed:mine` (board game rulebooks: HEAT, Dune: Imperium – Uprising).
 *
 * Usage:
 *   npm run example:rag:mine
 *   npm run example:rag:mine -- "Wie funktioniert der Windschatten bei HEAT?"
 */

import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";
import { loadEnv } from "../shared/load-env.js";

const DEFAULT_QUESTION = "Wie funktioniert der Windschatten bei HEAT?";

const INDEX_NAME = "handbook";
const NAMESPACE = "vc-bootcamp-janw";
const EMBEDDING_MODEL = "text-embedding-3-small";
const TOP_K = 4;
const REFUSAL = "Dazu steht nichts in den Spielanleitungen.";

/**
 * Games in the index. If the question names a game, retrieval is filtered to its
 * rulebook and the name is removed from the embedded query — otherwise the name
 * dominates the embedding and generic chunks (intro, credits) outrank the answer.
 */
const GAMES = [
  { pattern: /\bheat\b/gi, sourceFile: "heat-824968091319-regel.pdf" },
  {
    pattern: /\bdune\b\s*:?\s*(imperium)?\s*[-–]?\s*(uprising)?/gi,
    sourceFile: "dune-imperium-uprising-4015566605350-regel-1.1.pdf",
  },
];

interface RulebookHit {
  id: string;
  text: string;
  sourceFile: string;
  section: string;
  page?: number;
  chunkIndex?: number;
  score?: number;
}

function routeQuestion(question: string): { query: string; sourceFile?: string } {
  const game = GAMES.find((g) => question.match(g.pattern));
  if (!game) return { query: question };
  // drop the game name together with a preposition in front of it ("bei HEAT")
  const name = new RegExp(`(\\b(bei|in|von|für|zu)\\s+)?(${game.pattern.source})`, "gi");
  const query = question
    .replace(name, " ")
    .replace(/\s+([?.!,])/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
  return { query, sourceFile: game.sourceFile };
}

async function searchRulebooks(
  query: string,
  sourceFile?: string,
): Promise<RulebookHit[]> {
  const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY! });
  const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

  const host = (await pc.describeIndex(INDEX_NAME)).host;
  const ns = pc.index(INDEX_NAME, host).namespace(NAMESPACE);

  const embedding = await openai.embeddings.create({
    model: EMBEDDING_MODEL,
    input: query,
  });
  const results = await ns.query({
    vector: embedding.data[0].embedding,
    topK: TOP_K,
    includeMetadata: true,
    ...(sourceFile ? { filter: { source_file: { $eq: sourceFile } } } : {}),
  });

  return (results.matches ?? [])
    .filter((m) => typeof m.metadata?.chunk_text === "string")
    .map((m) => {
      const meta = m.metadata!;
      return {
        id: m.id,
        text: String(meta.chunk_text),
        sourceFile: String(meta.source_file ?? meta.doc ?? "unbekannt"),
        section: String(meta.section ?? ""),
        page: typeof meta.page === "number" ? meta.page : undefined,
        chunkIndex:
          typeof meta.chunk_index === "number" ? meta.chunk_index : undefined,
        score: m.score,
      };
    });
}

/** Citation label the model must copy verbatim, e.g. [heat-…-regel.pdf, S. 6, „6. Windschatten“] */
function citation(hit: RulebookHit): string {
  const page = hit.page != null ? `S. ${hit.page}` : null;
  return `[${[hit.sourceFile, page, `„${hit.section}“`].filter(Boolean).join(", ")}]`;
}

function formatContext(hits: RulebookHit[]): string {
  if (!hits.length) {
    return "Es wurden keine passenden Abschnitte gefunden.";
  }

  return hits
    .map((h, i) => {
      const score = h.score != null ? ` (score ${h.score.toFixed(3)})` : "";
      return `[${i + 1}] Zitat: ${citation(h)}${score}\n${h.text}`;
    })
    .join("\n\n");
}

const SYSTEM_PROMPT = [
  "Du beantwortest Regelfragen zu zwei Brettspielen: HEAT (heat-…-regel.pdf) und Dune: Imperium – Uprising (dune-imperium-uprising-…-regel-1.1.pdf).",
  "Antworte auf Deutsch, knapp und ausschließlich auf Basis der Abschnitte im Kontext. Nutze kein Vorwissen über die Spiele.",
  "Belege jede Regelaussage mit dem Zitat des verwendeten Abschnitts, genau so, wie es hinter „Zitat:“ steht.",
  "Achte darauf, dass der Abschnitt zum gefragten Spiel gehört (Quelldatei).",
  `Steht die Antwort nicht im Kontext, antworte ausschließlich mit: „${REFUSAL}“`,
].join("\n");

async function main(): Promise<void> {
  loadEnv();

  if (!process.env.PINECONE_API_KEY) {
    throw new Error("Missing PINECONE_API_KEY — set in dl-agentic/.env");
  }
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("Missing OPENAI_API_KEY — set in dl-agentic/.env");
  }

  const question =
    process.argv.slice(2).join(" ").trim() || DEFAULT_QUESTION;

  console.log("Pattern: Deterministic RAG (retrieve → prompt → LLM)\n");
  console.log(`Index: ${INDEX_NAME}/${NAMESPACE}`);
  console.log(`Question: ${question}\n`);

  console.log("→ retrieve (always runs, model does not decide)…");
  const { query, sourceFile } = routeQuestion(question);
  console.log(`  query: ${query}`);
  console.log(`  filter: ${sourceFile ? `source_file = ${sourceFile}` : "none"}\n`);
  const hits = await searchRulebooks(query, sourceFile);
  const context = formatContext(hits);

  console.log("Retrieved context:\n");
  console.log(context);
  console.log();

  const llm = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });

  const response = await llm.invoke([
    new SystemMessage(SYSTEM_PROMPT),
    new HumanMessage(`Kontext:\n${context}\n\nFrage: ${question}`),
  ]);

  const answer =
    typeof response.content === "string"
      ? response.content
      : JSON.stringify(response.content);

  console.log("--- Answer ---\n");
  console.log(answer);
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});
