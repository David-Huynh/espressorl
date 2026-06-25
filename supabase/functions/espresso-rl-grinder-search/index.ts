import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';
import { serve } from 'https://deno.land/std@0.224.0/http/server.ts';

const MIN_QUERY_LENGTH = 2;
const MAX_QUERY_LENGTH = 80;
const DEFAULT_LIMIT = 8;
const MAX_LIMIT = 10;
const IP_MINUTE_LIMIT = 60;

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'GET, OPTIONS',
};

type JsonRecord = Record<string, unknown>;
type SupabaseRpcClient = {
  rpc: (name: string, args: JsonRecord) => Promise<{ data: unknown; error: unknown }>;
};

serve(async request => {
  if (request.method === 'OPTIONS') {
    return new Response(null, { status: 204, headers: CORS_HEADERS });
  }
  if (request.method !== 'GET') {
    return jsonResponse(405, { error: 'method not allowed' });
  }

  const supabaseUrl = Deno.env.get('SUPABASE_URL') || '';
  const serviceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') || '';
  if (!supabaseUrl || !serviceRoleKey) {
    return jsonResponse(500, { error: 'grinder search service is not configured' });
  }
  const supabase = createClient(supabaseUrl, serviceRoleKey, {
    auth: { persistSession: false },
  });

  const url = new URL(request.url);
  const query = normalizeSearch(url.searchParams.get('q') || '');
  if (query.length < MIN_QUERY_LENGTH) {
    return jsonResponse(200, { suggestions: [] });
  }
  if (query.length > MAX_QUERY_LENGTH) {
    return jsonResponse(400, { error: 'query too long' });
  }

  const sourceIpHash = await sourceIpDigest(request);
  if (!(await consumeSearchRateLimit(supabase, sourceIpHash))) {
    return jsonResponse(429, { error: 'rate limited' }, { 'Retry-After': '60' });
  }

  const limit = safeLimit(url.searchParams.get('limit'));
  const { data, error } = await supabase
    .from('espressorl_grinder_aliases')
    .select(
      'alias_name, normalized_alias, confidence, grinder:espressorl_grinder_catalog(grinder_id, canonical_name, manufacturer, model, microns_per_step, max_steps, step_direction, confidence)',
    )
    .ilike('normalized_alias', `%${query}%`)
    .order('confidence', { ascending: false })
    .limit(limit);

  if (error) {
    return jsonResponse(500, { error: 'search failed' });
  }

  const suggestions = (data || []).map(toSuggestion).filter(Boolean).slice(0, limit);
  return jsonResponse(200, { suggestions });
});

function toSuggestion(row: JsonRecord): JsonRecord | null {
  const grinder = Array.isArray(row.grinder) ? row.grinder[0] : row.grinder;
  if (!isRecord(grinder)) {
    return null;
  }
  const name = stringValue(grinder.canonical_name);
  if (!name) {
    return null;
  }
  return {
    grinder_id: stringValue(grinder.grinder_id),
    name,
    alias: stringValue(row.alias_name),
    manufacturer: stringValue(grinder.manufacturer),
    model: stringValue(grinder.model),
    microns_per_step: numberValue(grinder.microns_per_step),
    max_steps: numberValue(grinder.max_steps),
    step_direction: stepDirectionValue(grinder.step_direction),
    confidence: Math.max(numberValue(row.confidence) || 0, numberValue(grinder.confidence) || 0),
  };
}

function normalizeSearch(value: string): string {
  return value
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9._ -]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, MAX_QUERY_LENGTH + 1);
}

function safeLimit(value: string | null): number {
  const parsed = Number(value || DEFAULT_LIMIT);
  if (!Number.isInteger(parsed)) {
    return DEFAULT_LIMIT;
  }
  return Math.max(1, Math.min(MAX_LIMIT, parsed));
}

async function consumeSearchRateLimit(supabase: SupabaseRpcClient, sourceIpHash: string) {
  const now = Date.now();
  const bucketStart = new Date(Math.floor(now / 60000) * 60000).toISOString();
  const { data, error } = await supabase.rpc('espressorl_consume_rate_limit', {
    p_scope: `grinder-search-ip:${sourceIpHash}`,
    p_bucket_start: bucketStart,
    p_limit: IP_MINUTE_LIMIT,
  });
  return !error && data === true;
}

async function sourceIpDigest(request: Request): Promise<string> {
  const raw =
    header(request, 'x-forwarded-for').split(',')[0].trim() ||
    header(request, 'cf-connecting-ip') ||
    'unknown';
  return sha256Hex(`espresso-rl-grinder-search:${raw}`);
}

function header(request: Request, key: string): string {
  return request.headers.get(key) || '';
}

async function sha256Hex(value: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value));
  return Array.from(new Uint8Array(digest))
    .map(byte => byte.toString(16).padStart(2, '0'))
    .join('');
}

function jsonResponse(status: number, body: JsonRecord, extraHeaders: Record<string, string> = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      ...CORS_HEADERS,
      ...extraHeaders,
      'Content-Type': 'application/json',
    },
  });
}

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function stringValue(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined;
}

function numberValue(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function stepDirectionValue(value: unknown): string | undefined {
  return value === 'higher_is_finer' || value === 'higher_is_coarser' ? value : undefined;
}
