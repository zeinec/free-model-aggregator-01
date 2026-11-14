// Anthropic upstream request and response translation, kept deterministic for replay.
import type { Json, ProviderDirectoryEntry } from './providers.ts';
import { authHeaders, isRecord, joinBase } from './providers.ts';

function dataImage(url: string): Json | null {
  const match = /^data:([^;,]+);base64,([\s\S]*)$/.exec(url);
  return match ? { type: 'image', source: { type: 'base64', media_type: match[1], data: match[2] } } : null;
}

/** OpenAI Chat Completions request into the Anthropic Messages request shape. */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system: string[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system' || raw.role === 'developer') {
      if (typeof raw.content === 'string') system.push(raw.content);
      continue;
    }
    const role = raw.role === 'assistant' ? 'assistant' : 'user';
    const content: Json[] = [];
    if (typeof raw.content === 'string') { if (raw.content !== '') content.push({ type: 'text', text: raw.content }); }
    else if (Array.isArray(raw.content)) for (const part of raw.content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') content.push({ type: 'text', text: part.text });
      if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
        const image = dataImage(part.image_url.url); if (image) content.push(image);
      }
    }
    if (role === 'assistant' && Array.isArray(raw.tool_calls)) for (const call of raw.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      let input: Json = {};
      try { const value: unknown = JSON.parse(typeof call.function.arguments === 'string' ? call.function.arguments : '{}'); if (isRecord(value)) input = value; } catch { /* malformed tool arguments become an empty object */ }
      content.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : '', name: typeof call.function.name === 'string' ? call.function.name : '', input });
    }
    if (content.length) messages.push({ role, content });
  }
  const out: Json = { model, max_tokens: typeof body.max_tokens === 'number' ? body.max_tokens : 1024, messages };
  if (system.length) out.system = system.join('\n');
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = Array.isArray(body.stop) ? body.stop.filter((s): s is string => typeof s === 'string') : typeof body.stop === 'string' ? [body.stop] : [];
  if (stops.length) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw)) return [];
      const fn = isRecord(raw.function) ? raw.function : raw;
      return typeof fn.name === 'string' ? [{ name: fn.name, description: typeof fn.description === 'string' ? fn.description : '', input_schema: isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} } }] : [];
    });
    if (tools.length) out.tools = tools;
  }
  if (body.stream === true) out.stream = true;
  // Anthropic has no JSON mode; supported routing excludes this wire for JSON requirements.
  return out;
}

/** Native Anthropic request with the selected provider model. */
export function anthropicRequest(body: Json, model: string): Json {
  return { ...body, model, max_tokens: typeof body.max_tokens === 'number' ? body.max_tokens : 1024 };
}

export async function callAnthropic(provider: ProviderDirectoryEntry, body: Json, stream: boolean, timeoutMs = 30000): Promise<Response> {
  const base = joinBase(process.env.MAKEFUN_ENV_URL ?? '', provider.base);
  return fetch(`${base}/v1/messages`, { method: 'POST', headers: { 'content-type': 'application/json', accept: stream ? 'text/event-stream' : 'application/json', 'anthropic-version': '2023-06-01', ...authHeaders(provider.auth, provider.key) }, body: JSON.stringify(body), signal: AbortSignal.timeout(timeoutMs) });
}

function contentText(content: unknown): string {
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []).join('');
}
function usageOf(value: unknown): { input: number; output: number } {
  const usage = isRecord(value) ? value : {};
  return { input: typeof usage.input_tokens === 'number' ? usage.input_tokens : 0, output: typeof usage.output_tokens === 'number' ? usage.output_tokens : 0 };
}

export function anthropicToOpenAI(answer: Json, model: string): Json {
  const usage = usageOf(answer.usage);
  return { id: typeof answer.id === 'string' ? answer.id : 'chatcmpl-anthropic', object: 'chat.completion', created: 1767225600, model, choices: [{ index: 0, message: { role: 'assistant', content: contentText(answer.content) }, finish_reason: 'stop' }], usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output } };
}

export function anthropicToClient(answer: Json, model: string): Json {
  const usage = usageOf(answer.usage);
  return { id: typeof answer.id === 'string' ? answer.id : 'msg_anthropic', type: 'message', role: 'assistant', model, content: Array.isArray(answer.content) ? answer.content : [], stop_reason: typeof answer.stop_reason === 'string' ? answer.stop_reason : 'end_turn', stop_sequence: answer.stop_sequence ?? null, usage: { input_tokens: usage.input, output_tokens: usage.output } };
}

async function* frames(stream: ReadableStream<Uint8Array>): AsyncGenerator<Json> {
  const reader = stream.getReader(); const decoder = new TextDecoder(); let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read(); buffer += decoder.decode(value, { stream: !done });
      let split: number;
      while ((split = buffer.indexOf('\n\n')) >= 0) {
        const frame = buffer.slice(0, split); buffer = buffer.slice(split + 2);
        for (const line of frame.split(/\r?\n/)) if (line.startsWith('data:')) { try { const v: unknown = JSON.parse(line.slice(5).trim()); if (isRecord(v)) yield v; } catch { /* ignore malformed events */ } }
      }
      if (done) break;
    }
  } finally { reader.releaseLock(); }
}
const openaiChunk = (model: string, delta: Json, finish: unknown = null): string => `data: ${JSON.stringify({ id: 'chatcmpl-anthropic', object: 'chat.completion.chunk', created: 1767225600, model, choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;
const anthropicEvent = (name: string, data: Json): string => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;

export async function* anthropicEventsToOpenAI(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let input = 0; let output = 0;
  yield openaiChunk(model, { role: 'assistant' });
  for await (const event of frames(stream)) {
    if (event.type === 'content_block_delta' && isRecord(event.delta) && typeof event.delta.text === 'string') yield openaiChunk(model, { content: event.delta.text });
    if (event.type === 'message_start' && isRecord(event.message) && isRecord(event.message.usage)) input = typeof event.message.usage.input_tokens === 'number' ? event.message.usage.input_tokens : 0;
    if (event.type === 'message_delta' && isRecord(event.usage)) output = typeof event.usage.output_tokens === 'number' ? event.usage.output_tokens : 0;
  }
  yield openaiChunk(model, {}, 'stop');
  yield `data: ${JSON.stringify({ id: 'chatcmpl-anthropic', object: 'chat.completion.chunk', created: 1767225600, model, choices: [], usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output } })}\n\n`;
  yield 'data: [DONE]\n\n';
}

export async function* anthropicEventsToAnthropic(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let input = 0; let output = 0; let id = 'msg_anthropic'; let stop = 'end_turn'; const pieces: string[] = [];
  for await (const event of frames(stream)) {
    if (event.type === 'message_start' && isRecord(event.message)) {
      id = typeof event.message.id === 'string' ? event.message.id : id;
      if (isRecord(event.message.usage)) input = typeof event.message.usage.input_tokens === 'number' ? event.message.usage.input_tokens : 0;
    }
    if (event.type === 'content_block_delta' && isRecord(event.delta) && typeof event.delta.text === 'string') pieces.push(event.delta.text);
    if (event.type === 'message_delta') {
      if (isRecord(event.delta) && typeof event.delta.stop_reason === 'string') stop = event.delta.stop_reason;
      if (isRecord(event.usage) && typeof event.usage.output_tokens === 'number') output = event.usage.output_tokens;
    }
  }
  // Rebuild the envelope in the client model name while retaining provider text and usage.
  yield anthropicEvent('message_start', { type: 'message_start', message: { id, type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: input, output_tokens: 0 } } });
  yield anthropicEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  for (const text of pieces) yield anthropicEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } });
  yield anthropicEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  yield anthropicEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: output } });
  yield anthropicEvent('message_stop', { type: 'message_stop' });
}
