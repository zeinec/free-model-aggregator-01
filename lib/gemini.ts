import type { Json, ProviderDirectoryEntry } from './providers.ts';
import { authHeaders, isRecord, joinBase } from './providers.ts';

export type GeminiPart = { type: 'text'; text: string } | { type: 'image'; mimeType: string; data: string } | { type: 'functionCall'; name: string; args: Json } | { type: 'functionResponse'; name: string; response: Json };
export interface GeminiMessage { role: 'user' | 'assistant'; parts: GeminiPart[] }
export interface GeminiTool { name: string; description?: string; parameters: Json }
export interface GeminiRequest {
  system: string;
  messages: GeminiMessage[];
  tools: GeminiTool[];
  maxTokens: number;
  temperature?: number;
  topP?: number;
  stopSequences: string[];
  jsonMode: boolean;
  stream: boolean;
  model: string;
}

function dataImage(url: string): { mimeType: string; data: string } | null {
  const match = /^data:([^;,]+);base64,([\s\S]*)$/.exec(url);
  return match ? { mimeType: match[1], data: match[2] } : null;
}

export function buildGeminiBody(request: GeminiRequest): Json {
  const contents = request.messages.map((message) => ({
    role: message.role === 'assistant' ? 'model' : 'user',
    parts: message.parts.map((part) => {
      if (part.type === 'text') return { text: part.text };
      if (part.type === 'image') return { inlineData: { mimeType: part.mimeType, data: part.data } };
      if (part.type === 'functionCall') return { functionCall: { name: part.name, args: part.args } };
      return { functionResponse: { name: part.name, response: part.response } };
    }),
  }));
  const generationConfig: Json = { maxOutputTokens: request.maxTokens };
  if (request.temperature !== undefined) generationConfig.temperature = request.temperature;
  if (request.topP !== undefined) generationConfig.topP = request.topP;
  if (request.stopSequences.length) generationConfig.stopSequences = request.stopSequences;
  if (request.jsonMode) generationConfig.responseMimeType = 'application/json';
  return {
    ...(request.system !== '' ? { systemInstruction: { parts: [{ text: request.system }] } } : {}),
    contents,
    ...(request.tools.length ? { tools: [{ functionDeclarations: request.tools.map((tool) => ({ name: tool.name, ...(tool.description ? { description: tool.description } : {}), parameters: tool.parameters })) }] } : {}),
    generationConfig,
  };
}

export function openAIToGeminiRequest(body: Json, model: string): GeminiRequest {
  const source = Array.isArray(body.messages) ? body.messages : [];
  let system = '';
  const messages: GeminiMessage[] = [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'assistant' : 'user';
    if (raw.role === 'system' || raw.role === 'developer') { if (typeof raw.content === 'string') system += (system ? '\n' : '') + raw.content; continue; }
    const parts: GeminiPart[] = [];
    if (typeof raw.content === 'string') { if (raw.content !== '') parts.push({ type: 'text', text: raw.content }); }
    else if (Array.isArray(raw.content)) for (const block of raw.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
      if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
        const image = dataImage(block.image_url.url); if (image) parts.push({ type: 'image', ...image });
      }
    }
    if (Array.isArray(raw.tool_calls)) for (const call of raw.tool_calls) if (isRecord(call) && isRecord(call.function) && typeof call.function.name === 'string') {
      let args: Json = {};
      try { const parsed: unknown = JSON.parse(typeof call.function.arguments === 'string' ? call.function.arguments : '{}'); if (isRecord(parsed)) args = parsed; } catch { /* malformed arguments become an empty object */ }
      parts.push({ type: 'functionCall', name: call.function.name, args });
    }
    if (raw.role === 'tool' && typeof raw.name === 'string') parts.push({ type: 'functionResponse', name: raw.name, response: { result: raw.content ?? '' } });
    if (parts.length) messages.push({ role, parts });
  }
  const tools: GeminiTool[] = [];
  if (Array.isArray(body.tools)) for (const raw of body.tools) {
    if (!isRecord(raw)) continue;
    const fn = isRecord(raw.function) ? raw.function : raw;
    if (typeof fn.name === 'string') tools.push({ name: fn.name, description: typeof fn.description === 'string' ? fn.description : undefined, parameters: isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} } });
  }
  const format = isRecord(body.response_format) ? body.response_format : {};
  return { system, messages, tools, maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : 0, temperature: typeof body.temperature === 'number' ? body.temperature : undefined, topP: typeof body.top_p === 'number' ? body.top_p : undefined, stopSequences: Array.isArray(body.stop) ? body.stop.filter((s): s is string => typeof s === 'string') : typeof body.stop === 'string' ? [body.stop] : [], jsonMode: format.type === 'json_object' || format.type === 'json_schema', stream: body.stream === true, model };
}

function textOfBlocks(content: unknown): string {
  if (!isRecord(content) || !Array.isArray(content.parts)) return '';
  return content.parts.flatMap((p) => isRecord(p) && typeof p.text === 'string' ? [p.text] : []).join('');
}
function usageOf(metadata: unknown): { prompt: number; completion: number; total: number } {
  const usage = isRecord(metadata) ? metadata : {};
  const prompt = typeof usage.promptTokenCount === 'number' ? usage.promptTokenCount : 0;
  const completion = typeof usage.candidatesTokenCount === 'number' ? usage.candidatesTokenCount : 0;
  return { prompt, completion, total: typeof usage.totalTokenCount === 'number' ? usage.totalTokenCount : prompt + completion };
}

export function geminiToOpenAI(answer: Json, model: string): Json {
  const candidate = Array.isArray(answer.candidates) && isRecord(answer.candidates[0]) ? answer.candidates[0] : {};
  const usage = usageOf(answer.usageMetadata);
  return { id: 'chatcmpl-gemini', object: 'chat.completion', created: 1767225600, model, choices: [{ index: 0, message: { role: 'assistant', content: textOfBlocks(candidate.content) }, finish_reason: 'stop' }], usage: { prompt_tokens: usage.prompt, completion_tokens: usage.completion, total_tokens: usage.total } };
}

export function geminiToAnthropic(answer: Json, model: string): Json {
  const candidate = Array.isArray(answer.candidates) && isRecord(answer.candidates[0]) ? answer.candidates[0] : {};
  const usage = usageOf(answer.usageMetadata);
  return { id: 'msg_gemini', type: 'message', role: 'assistant', model, content: [{ type: 'text', text: textOfBlocks(candidate.content) }], stop_reason: 'end_turn', stop_sequence: null, usage: { input_tokens: usage.prompt, output_tokens: usage.completion } };
}

export async function callGemini(provider: ProviderDirectoryEntry, model: string, body: Json, stream: boolean, timeoutMs = 30000): Promise<Response> {
  const base = joinBase(process.env.MAKEFUN_ENV_URL ?? '', provider.base);
  const action = stream ? 'streamGenerateContent' : 'generateContent';
  const headers = { 'content-type': 'application/json', ...(stream ? { accept: 'text/event-stream' } : {}), ...authHeaders(provider.auth, provider.key) };
  return fetch(`${base}/v1beta/models/${encodeURIComponent(model)}:${action}${stream ? '?alt=sse' : ''}`, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(timeoutMs) });
}

async function* frames(upstream: ReadableStream<Uint8Array>): AsyncGenerator<Json> {
  const reader = upstream.getReader(); const decoder = new TextDecoder(); let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read(); buffer += decoder.decode(value, { stream: !done });
      let split: number;
      while ((split = buffer.indexOf('\n\n')) >= 0) {
        const frame = buffer.slice(0, split); buffer = buffer.slice(split + 2);
        for (const line of frame.split(/\r?\n/)) if (line.startsWith('data:')) {
          const data = line.slice(5).trim(); if (!data) continue;
          try { const value: unknown = JSON.parse(data); if (isRecord(value)) yield value; } catch { /* ignore malformed SSE line */ }
        }
      }
      if (done) break;
    }
    if (buffer.trim()) for (const line of buffer.split(/\r?\n/)) if (line.startsWith('data:')) { try { const value: unknown = JSON.parse(line.slice(5).trim()); if (isRecord(value)) yield value; } catch { /* ignore */ } }
  } finally { reader.releaseLock(); }
}

export async function* geminiEventsFromSSE(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let index = 0;
  yield `data: ${JSON.stringify({ id: 'chatcmpl-gemini', object: 'chat.completion.chunk', created: 1767225600, model, choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }] })}\n\n`;
  for await (const frame of frames(upstream)) {
    const candidate = Array.isArray(frame.candidates) && isRecord(frame.candidates[0]) ? frame.candidates[0] : {};
    const text = textOfBlocks(candidate.content);
    if (text !== '') { index++; yield `data: ${JSON.stringify({ id: 'chatcmpl-gemini', object: 'chat.completion.chunk', created: 1767225600, model, choices: [{ index: 0, delta: { content: text }, finish_reason: null }] })}\n\n`; }
    if (candidate.finishReason !== undefined) {
      const usage = usageOf(frame.usageMetadata);
      yield `data: ${JSON.stringify({ id: 'chatcmpl-gemini', object: 'chat.completion.chunk', created: 1767225600, model, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] })}\n\n`;
      yield `data: ${JSON.stringify({ id: 'chatcmpl-gemini', object: 'chat.completion.chunk', created: 1767225600, model, choices: [], usage: { prompt_tokens: usage.prompt, completion_tokens: usage.completion, total_tokens: usage.total } })}\n\n`;
    }
  }
  yield 'data: [DONE]\n\n';
}

export async function* geminiEventsFromSSEToAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const collected: Json[] = [];
  for await (const frame of frames(upstream)) collected.push(frame);
  let promptTokens = 0; let outputTokens = 0;
  for (const frame of collected) {
    const usage = usageOf(frame.usageMetadata); promptTokens = usage.prompt; outputTokens = usage.completion;
  }
  yield `event: message_start\ndata: ${JSON.stringify({ type: 'message_start', message: { id: 'msg_gemini', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: promptTokens, output_tokens: 0 } } })}\n\n`;
  yield `event: content_block_start\ndata: ${JSON.stringify({ type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } })}\n\n`;
  for (const frame of collected) {
    const candidate = Array.isArray(frame.candidates) && isRecord(frame.candidates[0]) ? frame.candidates[0] : {};
    const text = textOfBlocks(candidate.content);
    if (text !== '') yield `event: content_block_delta\ndata: ${JSON.stringify({ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } })}\n\n`;
  }
  yield `event: content_block_stop\ndata: ${JSON.stringify({ type: 'content_block_stop', index: 0 })}\n\n`;
  yield `event: message_delta\ndata: ${JSON.stringify({ type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: outputTokens } })}\n\n`;
  yield `event: message_stop\ndata: ${JSON.stringify({ type: 'message_stop' })}\n\n`;
}

export function anthropicToGeminiRequest(body: Json, model: string): GeminiRequest {
  const messages: GeminiMessage[] = [];
  if (Array.isArray(body.messages)) for (const raw of body.messages) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'assistant' : 'user'; const parts: GeminiPart[] = [];
    const blocks = typeof raw.content === 'string' ? [{ type: 'text', text: raw.content }] : Array.isArray(raw.content) ? raw.content : [];
    for (const block of blocks) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
      else if (block.type === 'image' && isRecord(block.source) && block.source.type === 'base64' && typeof block.source.data === 'string') parts.push({ type: 'image', mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data });
      else if (block.type === 'tool_use' && typeof block.name === 'string') parts.push({ type: 'functionCall', name: block.name, args: isRecord(block.input) ? block.input : {} });
      else if (block.type === 'tool_result' && typeof block.tool_use_id === 'string') parts.push({ type: 'functionResponse', name: block.tool_use_id, response: { result: block.content ?? '' } });
    }
    if (parts.length) messages.push({ role, parts });
  }
  const tools: GeminiTool[] = Array.isArray(body.tools) ? body.tools.flatMap((raw) => isRecord(raw) && typeof raw.name === 'string' ? [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : undefined, parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }] : []) : [];
  return { system: typeof body.system === 'string' ? body.system : Array.isArray(body.system) ? body.system.flatMap((b) => isRecord(b) && typeof b.text === 'string' ? [b.text] : []).join('\n') : '', messages, tools, maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : 0, temperature: typeof body.temperature === 'number' ? body.temperature : undefined, topP: typeof body.top_p === 'number' ? body.top_p : undefined, stopSequences: Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === 'string') : [], jsonMode: false, stream: body.stream === true, model };
}
