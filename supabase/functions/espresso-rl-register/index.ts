import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';
import { serve } from 'https://deno.land/std@0.224.0/http/server.ts';

const REGISTRATION_IP_MINUTE_LIMIT = 5;
const REGISTRATION_IP_DAY_LIMIT = 25;
const MAX_CLOCK_SKEW_S = 15 * 60;

type JsonRecord = Record<string, unknown>;

serve(async request => {
  if (request.method !== 'POST') {
    return jsonResponse(405, { error: 'method not allowed' });
  }

  const supabaseUrl = Deno.env.get('SUPABASE_URL') || '';
  const serviceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') || '';
  if (!supabaseUrl || !serviceRoleKey) {
    return jsonResponse(500, { error: 'registration service is not configured' });
  }
  const supabase = createClient(supabaseUrl, serviceRoleKey, {
    auth: { persistSession: false },
  });

  let body: JsonRecord;
  try {
    body = await request.json();
  } catch (_error) {
    return jsonResponse(400, { error: 'invalid JSON payload' });
  }

  const action = String(body.action || '');
  if (action === 'register') {
    return registerInstall(supabase, request);
  }
  if (action === 'rotate') {
    return rotateCredentials(supabase, request, JSON.stringify(body));
  }
  if (action === 'revoke') {
    return revokeCredentials(supabase, request, JSON.stringify(body));
  }
  return jsonResponse(400, { error: 'unsupported action' });
});

async function registerInstall(
  supabase: ReturnType<typeof createClient>,
  request: Request,
): Promise<Response> {
  const sourceIpHash = await sourceIpDigest(request);
  const rateOk = await consumeRegistrationRateLimits(supabase, sourceIpHash);
  if (!rateOk) {
    await logAbuse(supabase, { sourceIpHash, reason: 'registration_rate_limited' });
    return jsonResponse(429, { error: 'rate limit exceeded' });
  }

  const credentials = newCredentials();
  const { error } = await supabase.from('espressorl_upload_credentials').insert({
    install_id: credentials.install_id,
    upload_token_id: credentials.upload_token_id,
    upload_secret: credentials.upload_secret,
    community_upload_enabled: true,
  });
  if (error) {
    await logAbuse(supabase, {
      sourceIpHash,
      reason: 'registration_insert_failed',
      detail: { code: error.code, message: error.message },
    });
    return jsonResponse(500, { error: 'failed to register install' });
  }

  return jsonResponse(201, credentials);
}

async function rotateCredentials(
  supabase: ReturnType<typeof createClient>,
  request: Request,
  body: string,
): Promise<Response> {
  const current = await requireSignedCredential(supabase, request, body);
  if (!current.ok) {
    return current.response;
  }

  const credentials = newCredentials(current.installId);
  const { error: insertError } = await supabase.from('espressorl_upload_credentials').insert({
    install_id: credentials.install_id,
    upload_token_id: credentials.upload_token_id,
    upload_secret: credentials.upload_secret,
    community_upload_enabled: true,
  });
  if (insertError) {
    return jsonResponse(500, { error: 'failed to rotate upload credential' });
  }

  await supabase
    .from('espressorl_upload_credentials')
    .update({ revoked_at: new Date().toISOString(), community_upload_enabled: false })
    .eq('install_id', current.installId)
    .eq('upload_token_id', current.tokenId);

  return jsonResponse(200, credentials);
}

async function revokeCredentials(
  supabase: ReturnType<typeof createClient>,
  request: Request,
  body: string,
): Promise<Response> {
  const current = await requireSignedCredential(supabase, request, body);
  if (!current.ok) {
    return current.response;
  }

  const { error } = await supabase
    .from('espressorl_upload_credentials')
    .update({ revoked_at: new Date().toISOString(), community_upload_enabled: false })
    .eq('install_id', current.installId)
    .eq('upload_token_id', current.tokenId);
  if (error) {
    return jsonResponse(500, { error: 'failed to revoke upload credential' });
  }
  return jsonResponse(200, { status: 'revoked' });
}

async function requireSignedCredential(
  supabase: ReturnType<typeof createClient>,
  request: Request,
  body: string,
): Promise<
  | { ok: true; installId: string; tokenId: string }
  | { ok: false; response: Response }
> {
  const installId = header(request, 'x-espressorl-install-id');
  const tokenId = header(request, 'x-espressorl-token-id');
  const timestampText = header(request, 'x-espressorl-timestamp');
  const signature = header(request, 'x-espressorl-signature');
  if (!installId || !timestampText || !signature) {
    await logAbuse(supabase, { installId, reason: 'credential_action_missing_headers' });
    return { ok: false, response: jsonResponse(400, { error: 'missing signed credential headers' }) };
  }

  const timestamp = Number(timestampText);
  const now = Math.floor(Date.now() / 1000);
  if (!Number.isInteger(timestamp) || Math.abs(now - timestamp) > MAX_CLOCK_SKEW_S) {
    await logAbuse(supabase, { installId, reason: 'credential_action_timestamp_out_of_range' });
    return { ok: false, response: jsonResponse(400, { error: 'timestamp out of range' }) };
  }

  const { data, error } = await supabase
    .from('espressorl_upload_credentials')
    .select('upload_secret, community_upload_enabled, revoked_at')
    .eq('install_id', installId)
    .eq('upload_token_id', tokenId)
    .maybeSingle();
  if (error || !data || !data.community_upload_enabled || data.revoked_at) {
    await logAbuse(supabase, { installId, reason: 'credential_action_unknown_or_revoked' });
    return { ok: false, response: jsonResponse(403, { error: 'unknown or revoked upload credential' }) };
  }

  const expectedSignature = await hmacSha256Hex(data.upload_secret, `${timestampText}.${body}`);
  if (!constantTimeEqual(signature, expectedSignature)) {
    await logAbuse(supabase, { installId, reason: 'credential_action_invalid_signature' });
    return { ok: false, response: jsonResponse(403, { error: 'invalid signature' }) };
  }
  return { ok: true, installId, tokenId };
}

async function consumeRegistrationRateLimits(
  supabase: ReturnType<typeof createClient>,
  sourceIpHash: string,
): Promise<boolean> {
  const now = new Date();
  const minuteBucket = new Date(Math.floor(now.getTime() / 60_000) * 60_000).toISOString();
  const dayBucket = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate())).toISOString();
  const checks = [
    [`registration-ip:${sourceIpHash}:minute`, minuteBucket, REGISTRATION_IP_MINUTE_LIMIT],
    [`registration-ip:${sourceIpHash}:day`, dayBucket, REGISTRATION_IP_DAY_LIMIT],
  ];
  for (const [scope, bucket, limit] of checks) {
    const { data, error } = await supabase.rpc('espressorl_consume_rate_limit', {
      p_scope: scope,
      p_bucket_start: bucket,
      p_limit: limit,
    });
    if (error || data !== true) {
      return false;
    }
  }
  return true;
}

function newCredentials(installId: string = crypto.randomUUID()) {
  return {
    install_id: installId,
    upload_token_id: crypto.randomUUID(),
    upload_secret: randomHex(32),
  };
}

function randomHex(byteCount: number): string {
  const bytes = new Uint8Array(byteCount);
  crypto.getRandomValues(bytes);
  return Array.from(bytes)
    .map(byte => byte.toString(16).padStart(2, '0'))
    .join('');
}

async function logAbuse(
  supabase: ReturnType<typeof createClient>,
  event: {
    installId?: string;
    sourceIpHash?: string;
    reason: string;
    detail?: JsonRecord;
  },
) {
  await supabase.from('espressorl_abuse_events').insert({
    install_id: event.installId || null,
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

function jsonResponse(status: number, body: JsonRecord): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}
