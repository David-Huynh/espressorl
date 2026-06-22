import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';
import { serve } from 'https://deno.land/std@0.224.0/http/server.ts';

const MAX_PAYLOAD_BYTES = 2_000_000;
const MAX_CLOCK_SKEW_S = 15 * 60;
const INSTALL_MINUTE_LIMIT = 30;
const INSTALL_DAY_LIMIT = 500;
const IP_MINUTE_LIMIT = 30;

type JsonRecord = Record<string, unknown>;

serve(async request => {
  if (request.method !== 'POST') {
    return jsonResponse(405, { error: 'method not allowed' });
  }

  const supabaseUrl = Deno.env.get('SUPABASE_URL') || '';
  const serviceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') || '';
  if (!supabaseUrl || !serviceRoleKey) {
    return jsonResponse(500, { error: 'ingestion service is not configured' });
  }
  const supabase = createClient(supabaseUrl, serviceRoleKey, {
    auth: { persistSession: false },
  });

  const installId = header(request, 'x-espressorl-install-id');
  const tokenId = header(request, 'x-espressorl-token-id');
  const uploadId = header(request, 'x-espressorl-upload-id');
  const timestampText = header(request, 'x-espressorl-timestamp');
  const signature = header(request, 'x-espressorl-signature');
  const payloadHash = header(request, 'x-espressorl-payload-hash');
  if (!installId || !uploadId || !timestampText || !signature || !payloadHash) {
    await logAbuse(supabase, { installId, uploadId, payloadHash, reason: 'missing_headers' });
    return jsonResponse(400, { error: 'missing required EspressoRL upload headers' });
  }

  const body = await request.text();
  if (new TextEncoder().encode(body).length > MAX_PAYLOAD_BYTES) {
    await logAbuse(supabase, { installId, uploadId, payloadHash, reason: 'payload_too_large' });
    return jsonResponse(413, { error: 'payload too large' });
  }

  const timestamp = Number(timestampText);
  const now = Math.floor(Date.now() / 1000);
  if (!Number.isInteger(timestamp) || Math.abs(now - timestamp) > MAX_CLOCK_SKEW_S) {
    await logAbuse(supabase, { installId, uploadId, payloadHash, reason: 'timestamp_out_of_range' });
    return jsonResponse(400, { error: 'timestamp out of range' });
  }

  const actualHash = await sha256Hex(body);
  if (actualHash !== payloadHash) {
    await logAbuse(supabase, { installId, uploadId, payloadHash, reason: 'payload_hash_mismatch' });
    return jsonResponse(400, { error: 'payload hash mismatch' });
  }

  const credential = await lookupCredential(supabase, installId, tokenId);
  if (!credential) {
    await logAbuse(supabase, { installId, uploadId, payloadHash, reason: 'unknown_or_revoked_install' });
    return jsonResponse(403, { error: 'unknown or revoked upload credential' });
  }

  const expectedSignature = await hmacSha256Hex(credential.upload_secret, `${timestampText}.${body}`);
  if (!constantTimeEqual(signature, expectedSignature)) {
    await logAbuse(supabase, { installId, uploadId, payloadHash, reason: 'invalid_signature' });
    return jsonResponse(403, { error: 'invalid signature' });
  }

  // Dedup before spending rate-limit budget: a re-send of an already-queued
  // upload is acknowledged without consuming the install's quota.
  if (await uploadAlreadyQueued(supabase, installId, uploadId)) {
    return jsonResponse(202, { status: 'duplicate' });
  }

  const sourceIpHash = await sourceIpDigest(request);
  const rate = await consumeRateLimits(supabase, installId, sourceIpHash);
  if (!rate.ok) {
    await logAbuse(supabase, {
      installId,
      uploadId,
      payloadHash,
      sourceIpHash,
      reason: 'rate_limited',
      detail: { retry_after: rate.retryAfter },
    });
    return jsonResponse(429, { error: 'rate limit exceeded' }, { 'Retry-After': String(rate.retryAfter) });
  }

  let payload: JsonRecord;
  try {
    payload = JSON.parse(body);
  } catch (_error) {
    await logAbuse(supabase, { installId, uploadId, payloadHash, sourceIpHash, reason: 'invalid_json' });
    return jsonResponse(400, { error: 'invalid JSON payload' });
  }

  const validation = validatePayload(payload);
  if (!validation.ok) {
    await logAbuse(supabase, {
      installId,
      uploadId,
      payloadHash,
      sourceIpHash,
      reason: 'schema_validation_failed',
      detail: { errors: validation.errors },
    });
    return jsonResponse(400, { error: validation.errors.join('; ') });
  }
  if (payload.install_id !== undefined && payload.install_id !== installId) {
    await logAbuse(supabase, {
      installId,
      uploadId,
      payloadHash,
      sourceIpHash,
      reason: 'install_id_mismatch',
      detail: { payload_install_id: String(payload.install_id) },
    });
    return jsonResponse(403, { error: 'payload install_id does not match upload credential' });
  }
  payload.install_id = installId;

  const { error } = await supabase.from('raw_upload_queue').insert({
    install_id: installId,
    upload_id: uploadId,
    upload_token_id: tokenId,
    payload_hash: payloadHash,
    local_record_type: validation.localRecordType,
    local_record_id: validation.localRecordId,
    event_type: payload.event_type,
    payload_json: payload,
    client_timestamp: timestamp,
    status: 'queued',
    source_ip_hash: sourceIpHash,
    validation_summary: { initial_validation: 'accepted' },
  });
  if (error) {
    if (error.code === '23505') {
      return jsonResponse(202, { status: 'duplicate' });
    }
    await logAbuse(supabase, {
      installId,
      uploadId,
      payloadHash,
      sourceIpHash,
      reason: 'queue_insert_failed',
      detail: { code: error.code, message: error.message },
    });
    return jsonResponse(500, { error: 'failed to queue upload' });
  }

  return jsonResponse(202, { status: 'queued' });
});

function validatePayload(payload: JsonRecord): {
  ok: boolean;
  errors: string[];
  localRecordType?: string;
  localRecordId?: string;
} {
  const errors: string[] = [];
  if (payload.event_type === 'shot_record') {
    validateShotRecord(payload, errors);
    return {
      ok: errors.length === 0,
      errors,
      localRecordType: 'shot',
      localRecordId: String(payload.shot_id || ''),
    };
  }
  if (payload.event_type === 'recommendation_record') {
    validateRecommendationRecord(payload, errors);
    return {
      ok: errors.length === 0,
      errors,
      localRecordType: 'recommendation',
      localRecordId: String(payload.recommendation_id || ''),
    };
  }
  return {
    ok: false,
    errors: ['unsupported event_type'],
  };
}

function validateShotRecord(payload: JsonRecord, errors: string[]) {
  requireString(payload, 'shot_id', errors);
  requireString(payload, 'install_id', errors);
  requireString(payload, 'machine_id', errors);
  requireNumberRange(payload, 'timestamp', 0, Number.MAX_SAFE_INTEGER, errors);
  requireNumberRange(payload, 'dose_in_g', 5, 30, errors);
  optionalNumberRange(payload, 'beverage_out_g', 5, 100, errors);
  requireNumberRange(payload, 'target_yield_g', 5, 100, errors);
  optionalNumberRange(payload, 'shot_time_s', 5, 90, errors);
  optionalNumberRange(payload, 'human_rating', 1, 5, errors);
  optionalBoolean(payload, 'feedback_recorded', errors);
  optionalEnum(payload, 'shot_type', ['espresso', 'utility_flush', 'cleaning', 'calibration', 'unknown'], errors);
  optionalBoolean(payload, 'exclude_from_local_optimization', errors);
  optionalBoolean(payload, 'rating_prompt_allowed', errors);
  optionalNumberRange(payload, 'optimization_weight', 0, 1, errors);
  optionalBoolean(payload, 'grind_followed', errors);
  optionalBoolean(payload, 'dose_followed', errors);
  optionalBoolean(payload, 'yield_followed', errors);
  optionalBoolean(payload, 'pump_flow_calibration_required', errors);
  optionalBoolean(payload, 'profile_flow_valid', errors);
  optionalBoolean(payload, 'profile_flow_masked', errors);
  optionalString(payload, 'weight_source', 80, errors);
  optionalString(payload, 'flow_source', 80, errors);
  optionalString(payload, 'flow_units', 40, errors);
  optionalString(payload, 'pump_flow_source', 80, errors);
  optionalString(payload, 'pump_flow_units', 40, errors);
  optionalNumberRange(payload, 'grind_recommendation_trust', 0, 1, errors);
  optionalNumberRange(payload, 'dose_recommendation_trust', 0, 1, errors);
  optionalNumberRange(payload, 'yield_recommendation_trust', 0, 1, errors);
  const profile = payload.profile_resampled;
  if (profile !== undefined) {
    if (!Array.isArray(profile) || profile.length !== 5) {
      errors.push('profile_resampled must have 5 channels');
    } else {
      validateProfileResampled(profile, payload.beverage_out_g, errors);
    }
  }
}

function validateRecommendationRecord(payload: JsonRecord, errors: string[]) {
  requireString(payload, 'recommendation_id', errors);
  requireString(payload, 'install_id', errors);
  requireString(payload, 'machine_id', errors);
  requireNumberRange(payload, 'next_dose_g', 5, 30, errors);
  requireNumberRange(payload, 'target_yield_g', 5, 100, errors);
  requireNumberRange(payload, 'target_ratio', 1.2, 3.5, errors);
}

function requireString(payload: JsonRecord, key: string, errors: string[]) {
  if (typeof payload[key] !== 'string' || String(payload[key]).length === 0) {
    errors.push(`${key} is required`);
  }
}

function requireNumberRange(payload: JsonRecord, key: string, min: number, max: number, errors: string[]) {
  const value = payload[key];
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) {
    errors.push(`${key} out of range`);
  }
}

function optionalNumberRange(payload: JsonRecord, key: string, min: number, max: number, errors: string[]) {
  if (payload[key] === undefined || payload[key] === null) {
    return;
  }
  requireNumberRange(payload, key, min, max, errors);
}

function optionalBoolean(payload: JsonRecord, key: string, errors: string[]) {
  if (payload[key] === undefined || payload[key] === null) {
    return;
  }
  if (typeof payload[key] !== 'boolean') {
    errors.push(`${key} must be boolean`);
  }
}

function optionalString(payload: JsonRecord, key: string, maxLength: number, errors: string[]) {
  if (payload[key] === undefined || payload[key] === null) {
    return;
  }
  if (typeof payload[key] !== 'string' || String(payload[key]).length > maxLength) {
    errors.push(`${key} must be a short string`);
  }
}

function optionalEnum(payload: JsonRecord, key: string, allowed: string[], errors: string[]) {
  if (payload[key] === undefined || payload[key] === null) {
    return;
  }
  if (typeof payload[key] !== 'string' || !allowed.includes(String(payload[key]))) {
    errors.push(`${key} is invalid`);
  }
}

function validateProfileResampled(profile: unknown[], beverageOutG: unknown, errors: string[]) {
  const ranges: Array<[number, number, string]> = [
    [0, 15, 'pressure'],
    [0, 15, 'target_pressure'],
    [0, 20, 'flow'],
    [0, 20, 'target_flow'],
    [-1, 100, 'weight'],
  ];
  const targetFlowActive = channelActive(profile[3]);
  for (let channelIndex = 0; channelIndex < 5; channelIndex += 1) {
    const channel = profile[channelIndex];
    const [min, max, label] = ranges[channelIndex];
    if (!Array.isArray(channel) || channel.length !== 100) {
      errors.push(`profile_resampled ${label} channel must have exactly 100 samples`);
      continue;
    }
    if (!channelInRange(channel, min, max)) {
      if (label === 'flow' && !targetFlowActive) {
        continue;
      }
      errors.push(`profile_resampled ${label} out of range`);
    }
  }
  if (typeof beverageOutG === 'number' && Number.isFinite(beverageOutG)) {
    const weight = profile[4];
    if (Array.isArray(weight) && weight.length === 100) {
      const finalWeight = weight[99];
      if (typeof finalWeight === 'number' && Number.isFinite(finalWeight) && Math.abs(finalWeight - beverageOutG) > 5) {
        errors.push('final profile weight does not match beverage_out_g');
      }
    }
  }
}

function channelActive(channel: unknown): boolean {
  if (!Array.isArray(channel) || channel.length !== 100) {
    return false;
  }
  return channel.some(value => typeof value === 'number' && Number.isFinite(value) && Math.abs(value) > 1e-6);
}

function channelInRange(channel: unknown[], min: number, max: number): boolean {
  return channel.every(
    value => typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max,
  );
}

async function lookupCredential(
  supabase: ReturnType<typeof createClient>,
  installId: string,
  tokenId: string,
) {
  const { data, error } = await supabase
    .from('espressorl_upload_credentials')
    .select('upload_secret, community_upload_enabled, revoked_at')
    .eq('install_id', installId)
    .eq('upload_token_id', tokenId)
    .maybeSingle();
  if (error || !data || !data.community_upload_enabled || data.revoked_at) {
    return null;
  }
  return data as { upload_secret: string };
}

async function consumeRateLimits(
  supabase: ReturnType<typeof createClient>,
  installId: string,
  sourceIpHash: string,
): Promise<{ ok: boolean; retryAfter: number }> {
  const now = new Date();
  const minuteBucket = new Date(Math.floor(now.getTime() / 60_000) * 60_000).toISOString();
  const dayBucket = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate())).toISOString();
  const minuteReset = secondsToNextMinute(now);
  const dayReset = secondsToNextUtcDay(now);
  const checks: Array<[string, string, number, number]> = [
    [`install:${installId}:minute`, minuteBucket, INSTALL_MINUTE_LIMIT, minuteReset],
    [`install:${installId}:day`, dayBucket, INSTALL_DAY_LIMIT, dayReset],
    [`ip:${sourceIpHash}:minute`, minuteBucket, IP_MINUTE_LIMIT, minuteReset],
  ];
  for (const [scope, bucket, limit, retryAfter] of checks) {
    const { data, error } = await supabase.rpc('espressorl_consume_rate_limit', {
      p_scope: scope,
      p_bucket_start: bucket,
      p_limit: limit,
    });
    if (error || data !== true) {
      return { ok: false, retryAfter };
    }
  }
  return { ok: true, retryAfter: 0 };
}

async function uploadAlreadyQueued(
  supabase: ReturnType<typeof createClient>,
  installId: string,
  uploadId: string,
): Promise<boolean> {
  const { data, error } = await supabase
    .from('raw_upload_queue')
    .select('upload_id')
    .eq('install_id', installId)
    .eq('upload_id', uploadId)
    .maybeSingle();
  // Fail open on error: the insert's unique constraint still dedups (23505 -> 202).
  return !error && data !== null;
}

function secondsToNextMinute(now: Date): number {
  return Math.max(1, 60 - now.getUTCSeconds());
}

function secondsToNextUtcDay(now: Date): number {
  const next = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() + 1);
  return Math.max(1, Math.ceil((next - now.getTime()) / 1000));
}

async function logAbuse(
  supabase: ReturnType<typeof createClient>,
  event: {
    installId?: string;
    uploadId?: string;
    payloadHash?: string;
    sourceIpHash?: string;
    reason: string;
    detail?: JsonRecord;
  },
) {
  await supabase.from('espressorl_abuse_events').insert({
    install_id: event.installId || null,
    upload_id: event.uploadId || null,
    payload_hash: event.payloadHash || null,
    source_ip_hash: event.sourceIpHash || null,
    reason: event.reason,
    detail: event.detail || {},
  });
}

async function sourceIpDigest(request: Request): Promise<string> {
  const raw =
    request.headers.get('x-forwarded-for') ||
    request.headers.get('cf-connecting-ip') ||
    'unknown';
  return sha256Hex(raw.split(',')[0].trim());
}

async function sha256Hex(value: string): Promise<string> {
  const data = new TextEncoder().encode(value);
  const hash = await crypto.subtle.digest('SHA-256', data);
  return hex(hash);
}

async function hmacSha256Hex(secret: string, value: string): Promise<string> {
  const key = await crypto.subtle.importKey(
    'raw',
    new TextEncoder().encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign'],
  );
  const signature = await crypto.subtle.sign('HMAC', key, new TextEncoder().encode(value));
  return hex(signature);
}

function constantTimeEqual(left: string, right: string): boolean {
  if (left.length !== right.length) {
    return false;
  }
  let diff = 0;
  for (let i = 0; i < left.length; i += 1) {
    diff |= left.charCodeAt(i) ^ right.charCodeAt(i);
  }
  return diff === 0;
}

function hex(buffer: ArrayBuffer): string {
  return Array.from(new Uint8Array(buffer))
    .map(byte => byte.toString(16).padStart(2, '0'))
    .join('');
}

function header(request: Request, name: string): string {
  return request.headers.get(name)?.trim() || '';
}

function jsonResponse(status: number, body: JsonRecord, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json', ...headers },
  });
}
