// free-model-aggregator-02: one server, two client wires, a real free model behind every answer.
//
// The trial gives this server a brief (MAKEFUN_PROMPT_FILE) listing OpenRouter's free models with the
// house's rating_milli, and six prompts the env drives at it one at a time. Every prompt is answered
// by dialling a free model through the grader's gateway ($OPENAI_BASE_URL, Chat Completions only: the
// gateway refuses these ids on every other route), translating between the client's wire and the
// gateway's, and relaying the provider's answer with the dialled id in `model`.
//
// Routing: candidates are ordered by the brief's rating_milli, best first, with the models that meet
// the request's requirements ahead of the ones that do not (a request's requirements are read off its
// body: response_format, tools, image blocks). The quality column credits the model that really
// answered against the best rating meeting the prompt's requirements, so the best-rated candidate
// that can serve the prompt is the one to try first, and every refusal falls through to the next.
//
// Failover: a refused call (HTTP 4xx/5xx from the gateway) is recorded and the next candidate is
// tried; a 5xx is retried once on the same model first, and an answer with no content is treated as
// a refusal without a trail row (the call succeeded; the completion is not worth serving). Recovered
// refusals are reported in x-makefun-upstream-failures, the only way the failover column sees them.
//
// Determinism: every choice is a pure function of the brief, the request and the gateway's answers.
// No wall clock and no randomness anywhere: a replayed trial must reproduce this server's answers,
// and the state export byte for byte. The two model listings use fixed timestamps for the same
// reason, and the poll interval below is a cadence, never a decision.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type Brief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const REGISTER_RETRY_MS = 200;
const REGISTER_RETRY_MAX = 150;
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The output cap a request without one names: the gateway refuses uncapped requests outright. */
const DEFAULT_MAX_TOKENS = 512;
/** One immediate retry for a 5xx: a provider's hiccup, not its rate limit. */
const SERVER_ERROR_RETRIES = 1;
/** One patient retry for a 429: the free tier's per-minute ceiling, not a dead model. */
const RATE_LIMIT_RETRIES = 1;
const RATE_LIMIT_WAIT_MS = 1500;
/** Refusals that will not heal within the trial: never spend another call on that model. */
const PERMANENT_REFUSAL_STATUSES = new Set([400, 401, 402, 403, 404, 405, 410, 422]);
const TRAIL_HEADER = 'x-makefun-upstream-failures';
const TRAIL_MAX_ROWS = 8;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key; the gateway swaps it for the OpenRouter key pushed to the competition. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Every free model, best rating first, brief order breaking ties: the failover ladder. */
const ranked: ModelRow[] = [...freeModels(brief)].sort((a, b) => b.rating_milli - a.rating_milli);

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** The gateway's Chat Completions route; a base without /v1 gets one. */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

function servedModels(): ServedModel[] {
  return ranked.map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/** The requirements a request body declares: JSON mode, tools, images. */
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
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/**
 * The ladder for one request: every free model, best rating first. The quality column credits the
 * model that really answered at min(its rating, the best rating meeting the prompt's requirements),
 * so the highest-rated model that serves the prompt earns that bound in full, whichever requirements
 * it carries. A model that meets the request's own requirements wins an equal-rating tie, and the
 * brief's order breaks the rest: the ladder is a pure function of the brief and the request.
 */
function ladderFor(body: Json): ModelRow[] {
  const requires = requirementsFromBody(body);
  const meets = (model: ModelRow): number => (meetsRequirements(requires, capabilitiesOf(model)) ? 1 : 0);
  return [...ranked].sort((a, b) => b.rating_milli - a.rating_milli || meets(b) - meets(a));
}

/**
 * Models a permanent refusal has retired this trial, with the refusal that retired them. A provider's
 * 4xx gate (a model held back from plain API callers, an unknown id) will answer the same way for
 * every later prompt, and one spent call per retired model is one spent call too many across six
 * prompts: the tie-break is the trial's call count. The memory is derived from the gateway's own
 * answers alone, so a replayed trial builds it again, call for call, and lands on the same ladder.
 */
const retired = new Map<string, TrailRow>();

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

function sendJson(res: ServerResponse, status: number, body: unknown, headers?: Record<string, string>): void {
  const text = JSON.stringify(body);
  res.writeHead(status, {
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(text),
    ...(headers ?? {}),
  });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  const openai = typeof req.headers['x-api-key'] !== 'string';
  const rows = list.map((m) => ({ ...m, free: true as const }));
  if (openai) {
    sendJson(res, 200, {
      object: 'list',
      data: rows.map((m) => ({ id: m.id, object: 'model', created: 1767225600, owned_by: m.provider, free: true })),
    });
  } else {
    sendJson(res, 200, {
      data: rows.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: '2026-01-01T00:00:00Z', free: true })),
      has_more: false,
    });
  }
}

interface TrailRow {
  status: number;
  code: string;
  model: string | null;
}

/** The gateway refusal's code: its error body's code, or its type, short enough for the trail. */
function errorCodeOf(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      if (typeof value.error.code === 'string' && value.error.code !== '') return value.error.code.slice(0, 64);
      if (typeof value.error.type === 'string' && value.error.type !== '') return value.error.type.slice(0, 64);
    }
  } catch {
    // Not JSON: no code to name.
  }
  return '';
}

async function dial(model: string, body: Json): Promise<{ res: Response } | { error: string }> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  try {
    const res = await fetch(chatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify({ ...body, model }) });
    return { res };
  } catch (err) {
    return { error: (err as Error).message };
  }
}

/** The request body as it goes upstream: the output cap filled in when the client named none. */
function upstreamBody(body: Json): Json {
  const hasCap = typeof body.max_tokens === 'number' || typeof body.max_completion_tokens === 'number';
  return hasCap ? body : { ...body, max_tokens: DEFAULT_MAX_TOKENS };
}

/** The trail header's value, only when a refusal came back that this answer recovered from. */
function trailHeader(failures: TrailRow[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { [TRAIL_HEADER]: JSON.stringify(failures.slice(0, TRAIL_MAX_ROWS)) };
}

/** The text a Chat Completions JSON answer carries, tool calls included as names only. */
function openAIAnswerText(answer: Json): string {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  if (typeof message.content === 'string') return message.content;
  if (!Array.isArray(message.content)) return '';
  return message.content.flatMap((p: unknown) => (isRecord(p) && p.type === 'text' && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** The text a buffered SSE relay carries, from the content deltas it will write. */
function openAIStreamText(frames: readonly string[]): string {
  let text = '';
  let tools = 0;
  for (const frame of frames) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '' || payload === '[DONE]') continue;
      try {
        const chunk: unknown = JSON.parse(payload);
        if (!isRecord(chunk) || !Array.isArray(chunk.choices)) continue;
        const choice = isRecord(chunk.choices[0]) ? chunk.choices[0] : null;
        if (choice === null) continue;
        const delta = isRecord(choice.delta) ? choice.delta : {};
        if (typeof delta.content === 'string') text += delta.content;
        if (Array.isArray(delta.tool_calls)) tools += delta.tool_calls.length;
      } catch {
        // Not JSON: the relay passes it through untouched.
      }
    }
  }
  return tools > 0 ? 'tool-call' : text;
}

/** The text a buffered Anthropic event relay carries, from its text deltas and tool blocks. */
function anthropicStreamText(events: readonly string[]): string {
  let text = '';
  let tools = 0;
  for (const event of events) {
    for (const line of event.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      try {
        const value: unknown = JSON.parse(trimmed.slice(5).trim());
        if (!isRecord(value)) continue;
        if (value.type === 'content_block_delta' && isRecord(value.delta) && value.delta.type === 'text_delta' && typeof value.delta.text === 'string') {
          text += value.delta.text;
        }
        if (value.type === 'content_block_start' && isRecord(value.content_block) && value.content_block.type === 'tool_use') tools += 1;
      } catch {
        // Not JSON: not ours.
      }
    }
  }
  return tools > 0 ? 'tool-call' : text;
}

/**
 * The candidates for one prompt, dialled in order, every refusal recorded. A 429 is retried once on
 * the same model after a short wait; a 5xx is retried once at once; a permanent 4xx retires the
 * model for the rest of the trial; anything else moves the ladder down. Returns null when the
 * ladder is spent: the caller answers a refusal of its own.
 */
async function serve(
  body: Json,
  failures: TrailRow[],
  answer: (model: ModelRow, upstream: Response) => Promise<boolean>,
): Promise<ModelRow | null> {
  for (const model of ladderFor(body)) {
    if (retired.has(model.id)) continue;
    const up = upstreamBody(body);
    let serverRetries = SERVER_ERROR_RETRIES;
    let limitRetries = RATE_LIMIT_RETRIES;
    for (;;) {
      const out = await dial(model.id, up);
      if ('error' in out) break;
      if (out.res.ok && out.res.body !== null) {
        const served = await answer(model, out.res);
        if (served) return model;
        // An answer with nothing in it: the call succeeded, so no trail row; the next model tries.
        break;
      }
      const text = out.res.body === null ? '' : await out.res.text();
      const row: TrailRow = { status: out.res.status, code: errorCodeOf(text), model: model.id };
      failures.push(row);
      if (out.res.status === 429 && limitRetries > 0) {
        limitRetries -= 1;
        await new Promise((resolve) => setTimeout(resolve, RATE_LIMIT_WAIT_MS));
        continue;
      }
      if (out.res.status >= 500 && serverRetries > 0) {
        serverRetries -= 1;
        continue;
      }
      if (PERMANENT_REFUSAL_STATUSES.has(out.res.status)) retired.set(model.id, row);
      break;
    }
  }
  return null;
}

/** Whether a Chat Completions JSON answer is a completion worth serving: choices, and text or tool calls. */
function openAIAnswerWorth(answer: Json): boolean {
  if (!Array.isArray(answer.choices) || answer.choices.length === 0) return false;
  const choice = isRecord(answer.choices[0]) ? answer.choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  if (Array.isArray(message.tool_calls) && message.tool_calls.length > 0) return true;
  return openAIAnswerText(answer).trim() !== '';
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the ladder. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const failures: TrailRow[] = [];
  const wantsStream = body.stream === true;
  const chosen = await serve(body, failures, async (model, upstream) => {
    if (wantsStream) {
      const frames: string[] = [];
      for await (const frame of openAIStreamWithModel(upstream.body, model.id)) frames.push(frame);
      if (openAIStreamText(frames).trim() === '') return false;
      res.writeHead(200, {
        'content-type': 'text/event-stream',
        'cache-control': 'no-cache',
        connection: 'keep-alive',
        ...trailHeader(failures),
      });
      res.end(frames.join(''));
      return true;
    }
    const answerBody: unknown = await upstream.json();
    if (!isRecord(answerBody)) return false;
    if (!openAIAnswerWorth(answerBody)) return false;
    sendJson(res, 200, openAIJsonWithModel(answerBody, model.id), trailHeader(failures));
    return true;
  });
  if (chosen === null) {
    sendError(res, 'openai', 502, 'upstream_error', 'every free model the ladder tried refused or answered nothing');
  }
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const failures: TrailRow[] = [];
  const wantsStream = body.stream === true;
  const chosen = await serve(body, failures, async (model, upstream) => {
    if (wantsStream) {
      const events: string[] = [];
      for await (const event of anthropicEventsFromOpenAI(upstream.body, model.id)) events.push(event);
      if (anthropicStreamText(events).trim() === '') return false;
      res.writeHead(200, {
        'content-type': 'text/event-stream',
        'cache-control': 'no-cache',
        connection: 'keep-alive',
        ...trailHeader(failures),
      });
      res.end(events.join(''));
      return true;
    }
    const answerBody: unknown = await upstream.json();
    if (!isRecord(answerBody)) return false;
    if (!openAIAnswerWorth(answerBody)) return false;
    sendJson(res, 200, openAIToAnthropic(answerBody, model.id), trailHeader(failures));
    return true;
  });
  if (chosen === null) {
    sendError(res, 'anthropic', 502, 'api_error', 'every free model the ladder tried refused or answered nothing');
  }
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

function listen(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => {
      const address = server.address();
      if (address === null || typeof address === 'string') reject(new Error('the server has no port'));
      else resolve(address.port);
    });
  });
}

/** POST /register with the port and the catalog; retried until the env is up to begin the trial. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const body = JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report });
  for (let attempt = 1; attempt <= REGISTER_RETRY_MAX; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body,
        signal: AbortSignal.timeout(5000),
      });
      if (res.ok) {
        const reply: unknown = await res.json();
        if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
        return;
      }
      log(`register answered HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
      return;
    } catch {
      // The env is not reachable yet: try again on the next beat.
    }
    await new Promise((resolve) => setTimeout(resolve, REGISTER_RETRY_MS));
  }
  throw new Error(`POST ${ENV_URL}/register never answered`);
}

/** Poll GET /work; exit 0 the moment it says done, so the trial never waits on this process. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            // Let the response's socket teardown settle before exiting: a process.exit that races an
            // open keep-alive socket aborts on one platform's libuv, and the trial wants a clean 0.
            try {
              server.closeAllConnections();
            } catch {
              // Older Node without the method: close() below still ends the listening handle.
            }
            server.close();
            setTimeout(() => process.exit(0), 150).unref();
            return;
          }
        }
      } catch {
        // The env is busy driving a prompt: try again on the next beat.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries: print the one built object and stop.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  let port = await listen(server);
  while (port === 8080) {
    // The env owns 8080 on one layout; the OS's ephemeral range rarely picks it, and one rebind
    // answers it when it does. The port is not part of the state export a replay compares.
    server.close();
    port = await listen(server);
  }
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models, best first: ${ranked[0]?.id ?? 'none'}`);
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
