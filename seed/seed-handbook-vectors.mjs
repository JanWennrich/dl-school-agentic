#!/usr/bin/env node
/**
 * Embed handbook-corpus.json chunks and upsert to Pinecone.
 *
 * Usage:
 *   npm run seed
 *   node seed/seed-handbook-vectors.mjs
 *
 * Own corpus into an own namespace (skips the handbook PDF regeneration):
 *   WORKSHOP_PINECONE_NAMESPACE=my-namespace \
 *     node seed/seed-handbook-vectors.mjs --corpus data/my-corpus.json [--reset]
 *
 * --reset deletes all vectors in the namespace first (not allowed for workshop-shared).
 */

import { readFileSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { execSync } from "node:child_process";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");

const INDEX_NAME = process.env.WORKSHOP_PINECONE_INDEX ?? "handbook";
const SHARED_NAMESPACE = "workshop-shared";
const DEFAULT_CORPUS = "data/handbook-corpus.json";
const CLOUD = process.env.WORKSHOP_PINECONE_CLOUD ?? "aws";
const REGION = process.env.WORKSHOP_PINECONE_REGION ?? "us-east-1";
const EMBEDDING_MODEL = "text-embedding-3-small";
const EMBEDDING_DIM = 1536;
const BATCH_SIZE = 50; // keeps each upsert request below Pinecone's 2 MB limit
const VERIFY_ATTEMPTS = 15;
const VERIFY_DELAY_MS = 2000;

function loadEnvLocal() {
  for (const name of [".env.local", ".env"]) {
    const path = join(ROOT, name);
    if (!existsSync(path)) continue;
    for (const line of readFileSync(path, "utf8").split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq === -1) continue;
    const key = trimmed.slice(0, eq).trim();
    let val = trimmed.slice(eq + 1).trim();
    if (
      (val.startsWith('"') && val.endsWith('"')) ||
      (val.startsWith("'") && val.endsWith("'"))
    ) {
      val = val.slice(1, -1);
    }
    if (process.env[key] === undefined || process.env[key] === "") {
      process.env[key] = val;
    }
    }
  }
}

function corpusArg() {
  const i = process.argv.indexOf("--corpus");
  if (i !== -1 && process.argv[i + 1]) return process.argv[i + 1];
  return process.env.WORKSHOP_CORPUS || DEFAULT_CORPUS;
}

function loadCorpus(path) {
  return JSON.parse(readFileSync(resolve(ROOT, path), "utf8"));
}

async function ensureIndex(pc) {
  const existing = await pc.listIndexes();
  const names = existing.indexes?.map((i) => i.name) ?? [];
  if (names.includes(INDEX_NAME)) {
    const desc = await pc.describeIndex(INDEX_NAME);
    if (desc.dimension !== EMBEDDING_DIM) {
      throw new Error(
        `Index ${INDEX_NAME} has dimension ${desc.dimension}, expected ${EMBEDDING_DIM}`,
      );
    }
    console.log(`Using existing index "${INDEX_NAME}" (${desc.dimension}d)`);
    return;
  }

  console.log(`Creating index "${INDEX_NAME}" (${EMBEDDING_DIM}d, cosine)…`);
  await pc.createIndex({
    name: INDEX_NAME,
    dimension: EMBEDDING_DIM,
    metric: "cosine",
    spec: {
      serverless: {
        cloud: CLOUD,
        region: REGION,
      },
    },
    waitUntilReady: true,
  });
  console.log(`✓ Index "${INDEX_NAME}" ready`);
}

async function embedBatch(openai, texts) {
  const response = await openai.embeddings.create({
    model: EMBEDDING_MODEL,
    input: texts,
  });
  return response.data
    .sort((a, b) => a.index - b.index)
    .map((row) => row.embedding);
}

function chunkMetadata(chunk, doc) {
  const meta = {
    doc: chunk.source_file ?? doc,
    page: chunk.page,
    section: chunk.section,
    section_id: chunk.section_id,
    chunk_text: chunk.chunk_text,
  };
  if (chunk.source_file) meta.source_file = chunk.source_file;
  if (chunk.chunk_index !== undefined) meta.chunk_index = chunk.chunk_index;
  if (chunk._note) meta._note = chunk._note;
  return meta;
}

async function main() {
  loadEnvLocal();

  const NAMESPACE = process.env.WORKSHOP_PINECONE_NAMESPACE || SHARED_NAMESPACE;
  const corpusPath = corpusArg();
  const isOwnCorpus = resolve(ROOT, corpusPath) !== resolve(ROOT, DEFAULT_CORPUS);
  if (isOwnCorpus && NAMESPACE === SHARED_NAMESPACE) {
    throw new Error(
      `Refusing to seed ${corpusPath} into "${SHARED_NAMESPACE}" — set WORKSHOP_PINECONE_NAMESPACE to your own namespace`,
    );
  }

  const pineconeKey = process.env.PINECONE_API_KEY;
  const openaiKey = process.env.OPENAI_API_KEY;
  if (!pineconeKey) {
    throw new Error("Missing PINECONE_API_KEY — set in dl-agentic/.env or .env.local");
  }
  if (!openaiKey) {
    throw new Error("Missing OPENAI_API_KEY — set in dl-agentic/.env or .env.local");
  }

  const reset = process.argv.includes("--reset");
  if (reset && NAMESPACE === SHARED_NAMESPACE) {
    throw new Error(`Refusing to --reset the shared namespace "${SHARED_NAMESPACE}"`);
  }

  if (!isOwnCorpus && !process.argv.includes("--skip-pdf")) {
    console.log("Regenerating employee_handbook.pdf…");
    execSync("python3 data/generate-handbook-pdf.py", {
      cwd: ROOT,
      stdio: "inherit",
    });
  }

  const corpus = loadCorpus(corpusPath);
  const chunks = corpus.chunks;
  console.log(`Embedding ${chunks.length} chunks from ${corpus.doc}…`);

  const { Pinecone } = await import("@pinecone-database/pinecone");
  const { default: OpenAI } = await import("openai");

  const pc = new Pinecone({ apiKey: pineconeKey });
  const openai = new OpenAI({ apiKey: openaiKey });

  await ensureIndex(pc);

  const indexHost = (await pc.describeIndex(INDEX_NAME)).host;
  const index = pc.index(INDEX_NAME, indexHost);
  const ns = index.namespace(NAMESPACE);

  if (reset) {
    const stats = await index.describeIndexStats();
    if (stats.namespaces?.[NAMESPACE]?.recordCount) {
      console.log(`Deleting all vectors in ${INDEX_NAME}/${NAMESPACE}…`);
      await ns.deleteAll();
    }
  }

  const texts = chunks.map((c) => c.chunk_text);
  const embeddings = await embedBatch(openai, texts);

  const vectors = chunks.map((chunk, i) => ({
    id: chunk.id,
    values: embeddings[i],
    metadata: chunkMetadata(chunk, corpus.doc),
  }));

  console.log(`Upserting ${vectors.length} vectors to ${INDEX_NAME}/${NAMESPACE}…`);
  for (let i = 0; i < vectors.length; i += BATCH_SIZE) {
    await ns.upsert({ records: vectors.slice(i, i + BATCH_SIZE) });
  }

  // Serverless indexes are eventually consistent: retry until all ids are readable.
  let missing = chunks.map((c) => c.id);
  for (let attempt = 1; missing.length && attempt <= VERIFY_ATTEMPTS; attempt++) {
    if (attempt > 1) await new Promise((r) => setTimeout(r, VERIFY_DELAY_MS));
    const found = new Set();
    for (let i = 0; i < missing.length; i += BATCH_SIZE) {
      const fetched = await ns.fetch({ ids: missing.slice(i, i + BATCH_SIZE) });
      for (const [id, record] of Object.entries(fetched.records ?? {})) {
        if (record?.values?.length) found.add(id);
      }
    }
    missing = missing.filter((id) => !found.has(id));
  }
  if (missing.length) {
    throw new Error(`Upsert verification failed: ${missing.join(", ")}`);
  }

  console.log(`✓ Synced ${vectors.length} vectors to Pinecone`);
  for (const chunk of chunks) {
    console.log(`  · ${chunk.id} (p.${chunk.page}, ${chunk.section})`);
  }

  // Stats are eventually consistent, so the count can lag right after an upsert.
  const stats = await index.describeIndexStats();
  const count = stats.namespaces?.[NAMESPACE]?.recordCount ?? 0;
  console.log(`Namespace "${NAMESPACE}" in index "${INDEX_NAME}": ${count} vectors`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
