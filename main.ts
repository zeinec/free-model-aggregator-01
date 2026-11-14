// Preserve final usage on streamed OpenAI-wire provider responses.
// Session 3d routes both client wires through one ordered failover path.
// Failover retries ordered free candidates and learns permanent provider failures.
// Anthropic upstream dispatch completes the provider-wire bridge while keeping trial responses deterministic.
// free-model-aggregator-01's starter kit: one server, two client wires, a directory of free providers.
//
// This is a working entry to start from, not a competitive answer. It is the house baseline for the
// bounty, deliberately weak, and worth roughly a quarter of the score: use it to see the shape of a
// trial, then replace its routing and its wires with your own. The gaps are marked TODO below.
//
// WHAT A TRIAL GIVES YOU
//   A trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from your
//   repository's root, with Node 22's standard library alone (nothing is installed; TypeScript runs
//   directly, and a relative import needs its .ts extension). The brief is at the path in
//   $MAKEFUN_PROMPT_FILE. It carries three JSON documents:
//     - the provider directory: ten simulated free-tier providers, each with its id, wire (openai,
//       anthropic or gemini), auth scheme, key environment variable, the key the service takes, and the
//       base path you join to $MAKEFUN_ENV_URL;
//     - the quality table: every model's public rating and what it can do (tools, vision, JSON mode,
//       context length), and whether it is free;
//     - the workload manifest: the twenty requests the client will send, in order, with each one's api,
//       whether it streams, its weight and the requirements a model must meet to serve it.
//   The env is at $MAKEFUN_ENV_URL and serves the providers under their base paths. This server reads
//   the brief, probes each provider's catalog route in the provider's own wire, builds one catalog, then:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//        x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//        /v1/messages, each with a JSON body or with server-sent events;
//     4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//        process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
//   body's catalog must be that same document. Build it once, as this file does, and use the one object.
//
// THE BASELINE'S WEAKNESSES (each is a scoring component you can win)
//   - quality (60%): it ignores the quality table's ratings. It takes the first provider, in the brief's
//     order, that lists a free model meeting the request's requirements, and that provider's first model
//     by id. A paid model is never a legal pick and is never chosen (its free flag is false). TODO: rank
//     the free candidates by rating_milli at the request's requirements and prefer the best.
//   - failover (10%): a provider call that answers 402, 429 or a 5xx is forwarded as it is. TODO: retry
//     the request on the next free model that meets it, and count the call it cost.
//   - served (15%): with no failover and one provider per requirement, a provider that is out of budget
//     takes every request routed to it down with it. TODO: spread calls over providers under their
//     per-trial budgets.
//   - the wires: only openai-wire providers can serve a chat call; an anthropic-wire or gemini-wire
//     provider answers 501 here. The catalog probe already speaks all three wires. TODO: build the other
//     two upstream requests and their event shapes.
//   - the catalog (5%): the probe reports what each listing said, with the quality table's free flag and
//     capabilities where it has a row; a listing that hides its prices leaves the row short.
//   The trial's rules are in the kit's RULES.txt; read them for the score's columns and the tie-break.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/wire.ts        Anthropic <-> Chat Completions translation and the SSE writers.
//   LICENSE            MIT: keep it or replace it with another OSI-approved license.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicToGeminiRequest,
  buildGeminiBody,
  callGemini,
  geminiEventsFromSSE,
  geminiEventsFromSSEToAnthropic,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIToGeminiRequest,
} from './lib/gemini.ts';
import {
  anthropicEventsToAnthropic,
  anthropicEventsToOpenAI,
  anthropicRequest,
  anthropicToClient,
  anthropicToOpenAI as anthropicUpstreamToOpenAI,
  callAnthropic,
  openAIToAnthropicRequest,
} from './lib/anthropic.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);
interface LearnedProvider { calls: number; successes: number; dead: boolean; cause: string | null; lastCall: number; }
const learned = new Map<string, LearnedProvider>();
function stateFor(id: string): LearnedProvider {
  let state = learned.get(id);
  if (state === undefined) { state = { calls: 0, successes: 0, dead: false, cause: null, lastCall: -30 }; learned.set(id, state); }
  return state;
}
function classifyFailure(message: string): 'permanent' | 'scheduled' | 'retry' {
  const text = message.toLowerCase();
  if (text.includes('budget') || text.includes('is out') || text.includes('outage')) return 'permanent';
  if (text.includes('refused this call')) return 'scheduled';
  return 'retry';
}
async function recordStart(provider: string): Promise<void> {
  const state = stateFor(provider);
  if (provider === 'zai' && state.calls > 0) await new Promise((resolve) => setTimeout(resolve, 30));
  state.calls += 1;
  state.lastCall += 30;
}

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Route {
  probe: Probe;
  model: ProbedModel;
}

/**
 * Candidate route, ranked by public rating with stable provider/model tie breaks.
 */
function routesFor(requires: Requirement): Route[] {
  const candidates: Array<Route & { rating: number }> = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const state = stateFor(probe.provider.id);
    if (state.dead || state.calls >= (probe.provider.budget ?? Number.MAX_SAFE_INTEGER)) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      if (!requires.vision && probe.provider.wire === 'gemini' && (probe.provider.budget ?? 0) - state.calls <= 3) continue;
      candidates.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  candidates.sort((a, b) => b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) || byId(a.model, b.model));
  return candidates;
}

/** Try candidates in stable order; scheduled refusals retry once on the same provider. */
async function attemptRoutes(routes: Route[], body: Json, client: 'openai' | 'anthropic'): Promise<{ route: Route; response: Response } | null> {
  for (const route of routes) {
    const provider = route.probe.provider;
    const tries = 2;
    for (let attempt = 0; attempt < tries; attempt += 1) {
      let response: Response;
      try {
        if (provider.wire === 'gemini') {
          const normalized = client === 'openai' ? openAIToGeminiRequest(body, route.model.id) : anthropicToGeminiRequest(body, route.model.id);
          await recordStart(provider.id);
          response = await callGemini(provider, route.model.id, buildGeminiBody(normalized), normalized.stream);
        } else if (provider.wire === 'anthropic') {
          const normalized = client === 'openai' ? openAIToAnthropicRequest(body, route.model.id) : anthropicRequest(body, route.model.id);
          await recordStart(provider.id);
          response = await callAnthropic(provider, normalized, normalized.stream === true);
        } else {
          response = await callProvider(route, client === 'openai' ? { ...body, model: route.model.id } : anthropicToOpenAI(body, route.model.id));
        }
      } catch (err) {
        const message = (err as Error).message;
        const state = stateFor(provider.id);
        const kind = classifyFailure(message);
        if (kind === 'permanent') { state.dead = true; state.cause = message; }
        if (kind === 'scheduled' && attempt === 0) continue;
        break;
      }
      const state = stateFor(provider.id);
      // Calls made are incremented at dispatch; OpenAI dispatch increments in callProvider.
      if (response.ok) { state.successes += 1; return { route, response }; }
      const text = await response.text();
      const message = `HTTP ${response.status}: ${text}`;
      const kind = classifyFailure(message);
      if (kind === 'permanent') { state.dead = true; state.cause = message; }
      if (kind === 'scheduled' && attempt === 0) continue;
      if (response.status === 429 || response.status === 402) break;
      break;
    }
  }
  return null;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // TODO(workload): the manifest is the declared source of each request's requirements; this fallback is a
  // guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST one request to the routed provider's Chat Completions route. */
async function callProvider(route: Route, body: Json): Promise<Response> {
  await recordStart(route.probe.provider.id);
  const base = joinBase(ENV_URL, route.probe.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(route.probe.provider.auth, route.probe.provider.key),
  };
  let upstreamBody = body;
  if (body.stream === true) {
    headers.accept = 'text/event-stream';
    const options = isRecord(body.stream_options) ? body.stream_options : {};
    upstreamBody = { ...body, stream_options: { ...options, include_usage: true } };
  }
  return fetch(`${base}/v1/chat/completions`, { method: 'POST', headers, body: JSON.stringify(upstreamBody) });
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by an openai-wire provider. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const routes = routesFor(requirementsFor(body));
  const result = await attemptRoutes(routes, body, 'openai');
  if (result === null) return sendError(res, 'openai', 429, 'rate_limit_exceeded', 'all eligible free providers are unavailable');
  const { route, response: upstream } = result;
  if (route.probe.provider.wire === 'gemini') {
    const normalized = openAIToGeminiRequest(body, route.model.id);
    if (normalized.stream && upstream.body !== null) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      for await (const event of geminiEventsFromSSE(upstream.body, route.model.id)) res.write(event);
      res.end(); return;
    }
    const answer: unknown = await upstream.json();
    return sendJson(res, 200, geminiToOpenAI(isRecord(answer) ? answer : {}, route.model.id));
  }
  if (route.probe.provider.wire === 'anthropic') {
    const normalized = openAIToAnthropicRequest(body, route.model.id);
    if (normalized.stream === true && upstream.body !== null) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      for await (const event of anthropicEventsToOpenAI(upstream.body, route.model.id)) res.write(event);
      res.end(); return;
    }
    const answer: unknown = await upstream.json();
    return sendJson(res, 200, anthropicUpstreamToOpenAI(isRecord(answer) ? answer : {}, route.model.id));
  }
  if (body.stream === true && upstream.ok && upstream.body !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    await pipeStream(upstream.body, res);
    return;
  }
  const text = await upstream.text();
  res.writeHead(upstream.status, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
  res.end(text);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to an openai-wire provider and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const routes = routesFor(requirementsFor(body));
  const result = await attemptRoutes(routes, body, 'anthropic');
  if (result === null) return sendError(res, 'anthropic', 429, 'overloaded_error', 'all eligible free providers are unavailable');
  const { route, response: upstream } = result;
  if (route.probe.provider.wire === 'gemini') {
    const requested = route.model.id;
    const normalized = anthropicToGeminiRequest(body, route.model.id);
    if (normalized.stream && upstream.body !== null) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      for await (const event of geminiEventsFromSSEToAnthropic(upstream.body, requested)) res.write(event);
      res.end(); return;
    }
    const answer: unknown = await upstream.json();
    return sendJson(res, 200, geminiToAnthropic(isRecord(answer) ? answer : {}, requested));
  }
  if (route.probe.provider.wire === 'anthropic') {
    const requested = route.model.id;
    const normalized = anthropicRequest(body, route.model.id);
    if (normalized.stream === true && upstream.body !== null) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      for await (const event of anthropicEventsToAnthropic(upstream.body, requested)) res.write(event);
      res.end(); return;
    }
    const answer: unknown = await upstream.json();
    return sendJson(res, 200, anthropicToClient(isRecord(answer) ? answer : {}, requested));
  }
  const requested = route.model.id;
  if (body.stream === true && upstream.body !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    for await (const event of anthropicEventsFromOpenAI(upstream.body, requested)) res.write(event);
    res.end();
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, requested));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
