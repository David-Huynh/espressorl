import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';
import { serve } from 'https://deno.land/std@0.224.0/http/server.ts';

const MAX_PAYLOAD_BYTES = 2_000_000;
const MAX_CLOCK_SKEW_S = 15 * 60;
const INSTALL_MINUTE_LIMIT = 30;
const INSTALL_DAY_LIMIT = 500;
const IP_MINUTE_LIMIT = 30;
const SUPPORTED_SCHEMA_VERSION = 1;
const SAFE_IDENTIFIER = /^[A-Za-z0-9_.:@-]{1,160}$/;
const SHA256_HEX = /^[0-9a-f]{64}$/i;
const UNSAFE_TEXT = /[\x00-\x08\x0b\x0c\x0e-\x1f\x7f<>]/;
const tasteGoalAttributes = new Set([
  'fruity', 'citrus', 'floral', 'sweet', 'nutty_cocoa', 'roasted', 'spice', 'fermented',
  'sour', 'green_vegetative', 'bitter', 'astringent_harsh', 'papery_stale', 'salty',
]);
const tasteGoalLevels = new Set(['low', 'medium', 'high']);

type JsonRecord = Record<string, unknown>;

const allowedShotFields = new Set([
  'event_type', 'schema_version', 'shot_id', 'timestamp', 'install_id', 'machine_id',
  'machine_adapter', 'bean_context_id', 'grinder_context_id', 'profile_resampled',
  'taste_goal',
  'raw_profile_available', 'raw_profile_hash', 'grinder_calibration_mode',
  'grinder_adjustment_mode', 'microns_per_step', 'step_direction', 'reference_label',
  'relative_grind_steps_from_reference',
  'relative_grind_um_from_reference', 'current_absolute_step', 'absolute_reference_step',
  'action_observed',
  'dose_in_g', 'dose_target_g', 'dose_observed', 'dose_target_confirmed',
  'beverage_out_g', 'beverage_out_observation', 'predicted_final_beverage_out_g',
  'predictive_stop_applied', 'predictive_stop_delay_ms', 'predictive_stop_rate_g_per_s',
  'predictive_stop_lead_g', 'brew_ratio', 'target_yield_g', 'target_ratio',
  'shot_time_s', 'recommendation_id', 'recommended_grind_delta_steps_from_current',
  'recommended_grind_delta_um_from_current', 'recommended_projected_relative_step_from_reference',
  'recommended_dose_g', 'recommended_target_yield_g', 'recommended_target_ratio',
  'recommendation_decision', 'recommendation_followed',
  'shot_type', 'exclude_from_local_optimization',
  'grind_followed', 'dose_followed', 'yield_followed',
  'weight_source', 'flow_source', 'flow_units',
  'pump_flow_source', 'pump_flow_units', 'pump_flow_calibration_required',
  'profile_flow_valid', 'profile_flow_masked', 'profile_id', 'profile_label',
  'profile_type', 'profile_phase_count', 'final_phase_index', 'final_phase_name',
  'final_phase_type', 'final_phase_elapsed_s', 'final_pump_target', 'final_target_pressure',
  'final_target_flow', 'final_valve_open', 'profile_temperature_c',
  'final_phase_temperature_c', 'beverage_flow_profile', 'temperature_profile', 'target_temperature_profile',
  'pump_target_mode_profile', 'fixed_cadence_sequence', 'shot_end_state', 'created_at', 'updated_at',
]);

const allowedRecommendationFields = new Set([
  'event_type', 'schema_version', 'recommendation_id', 'created_at', 'updated_at',
  'expires_at', 'install_id', 'machine_id', 'bean_context_id', 'grinder_context_id',
  'profile_id', 'raw_profile_hash',
  'taste_goal',
  'grind_delta_steps_from_current', 'grind_delta_um_from_current',
  'projected_relative_step_from_reference', 'projected_relative_grind_um_from_reference',
  'grinder_calibration_mode', 'grinder_adjustment_mode', 'microns_per_step',
  'step_direction', 'reference_label',
  'current_absolute_step', 'absolute_reference_step', 'projected_absolute_step',
  'next_dose_g', 'target_yield_g', 'target_ratio', 'mode', 'confidence', 'reason',
  'status', 'shown_count', 'accepted_at', 'ignored_at', 'edited_at', 'used_at',
  'superseded_at', 'source_shot_id', 'apply_status', 'apply_acknowledged_at',
  'optimization_run_id', 'comparison_anchor_shot_id', 'comparison_mode',
  'preference_feedback_required',
  'applied_fields', 'manual_fields', 'apply_error',
]);

const allowedComparisonFields = new Set([
  'event_type', 'schema_version', 'comparison_id', 'optimization_run_id',
  'new_shot_id', 'anchor_shot_id', 'label', 'comparison_mode', 'created_at',
  'install_id', 'machine_id', 'machine_adapter', 'recommendation_id',
  'bean_context_id', 'grinder_context_id', 'profile_id', 'raw_profile_hash',
  'taste_goal',
]);

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
  if (!safeIdentifier(installId) || !safeIdentifier(uploadId) || !SHA256_HEX.test(signature) || !SHA256_HEX.test(payloadHash)) {
    await logAbuse(supabase, { installId, uploadId, payloadHash, reason: 'invalid_headers' });
    return jsonResponse(400, { error: 'invalid EspressoRL upload headers' });
  }

  const contentLength = request.headers.get('content-length');
  if (contentLength && Number(contentLength) > MAX_PAYLOAD_BYTES) {
    await logAbuse(supabase, { installId, uploadId, payloadHash, reason: 'payload_too_large' });
    return jsonResponse(413, { error: 'payload too large' });
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
  if (await uploadAlreadyQueued(supabase, installId, uploadId, payloadHash)) {
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
  if (!payload || Array.isArray(payload) || typeof payload !== 'object') {
    await logAbuse(supabase, { installId, uploadId, payloadHash, sourceIpHash, reason: 'invalid_json_shape' });
    return jsonResponse(400, { error: 'payload must be an object' });
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
  const sanitizedPayload = sanitizePayload(payload);
  sanitizedPayload.install_id = installId;

  const queueRow = {
    install_id: installId,
    upload_id: uploadId,
    upload_token_id: tokenId,
    payload_hash: payloadHash,
    local_record_type: validation.localRecordType,
    local_record_id: validation.localRecordId,
    event_type: sanitizedPayload.event_type,
    payload_json: sanitizedPayload,
    client_timestamp: timestamp,
    status: 'queued',
    mirror_error: null,
    mirror_claimed_by: null,
    mirror_claimed_at: null,
    mirror_claim_expires_at: null,
    mirror_completed_at: null,
    mirror_attempt_count: 0,
    source_ip_hash: sourceIpHash,
    validation_summary: { initial_validation: 'accepted' },
  };
  const queued = validation.localRecordType === 'recommendation'
    ? await supabase.from('raw_upload_queue').upsert(queueRow, { onConflict: 'install_id,upload_id' })
    : await supabase.from('raw_upload_queue').insert(queueRow);
  const { error } = queued;
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
    rejectUnknownFields(payload, allowedShotFields, errors);
    validateShotRecord(payload, errors);
    return {
      ok: errors.length === 0,
      errors,
      localRecordType: 'shot',
      localRecordId: String(payload.shot_id || ''),
    };
  }
  if (payload.event_type === 'recommendation_record') {
    rejectUnknownFields(payload, allowedRecommendationFields, errors);
    validateRecommendationRecord(payload, errors);
    return {
      ok: errors.length === 0,
      errors,
      localRecordType: 'recommendation',
      localRecordId: String(payload.recommendation_id || ''),
    };
  }
  if (payload.event_type === 'comparison_record') {
    rejectUnknownFields(payload, allowedComparisonFields, errors);
    validateComparisonRecord(payload, errors);
    return {
      ok: errors.length === 0,
      errors,
      localRecordType: 'comparison',
      localRecordId: String(payload.comparison_id || ''),
    };
  }
  return {
    ok: false,
    errors: ['unsupported event_type'],
  };
}

function validateShotRecord(payload: JsonRecord, errors: string[]) {
  requireNumberRange(payload, 'schema_version', SUPPORTED_SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSION, errors);
  requireIdentifier(payload, 'shot_id', errors);
  requireIdentifier(payload, 'install_id', errors);
  requireIdentifier(payload, 'machine_id', errors);
  optionalIdentifier(payload, 'machine_adapter', errors);
  optionalIdentifier(payload, 'bean_context_id', errors);
  optionalString(payload, 'grinder_context_id', 120, errors);
  requireTasteGoal(payload, errors);
  requireNumberRange(payload, 'timestamp', 0, Number.MAX_SAFE_INTEGER, errors);
  optionalNumberRange(payload, 'dose_in_g', 5, 30, errors);
  requireNumberRange(payload, 'dose_target_g', 5, 30, errors);
  optionalBoolean(payload, 'dose_observed', errors);
  optionalBoolean(payload, 'dose_target_confirmed', errors);
  optionalNumberRange(payload, 'beverage_out_g', 0, 120, errors);
  optionalString(payload, 'beverage_out_observation', 40, errors);
  optionalNumberRange(payload, 'predicted_final_beverage_out_g', 0, 120, errors);
  optionalBoolean(payload, 'predictive_stop_applied', errors);
  optionalNumberRange(payload, 'predictive_stop_delay_ms', 0, 10000, errors);
  optionalNumberRange(payload, 'predictive_stop_rate_g_per_s', 0, 25, errors);
  optionalNumberRange(payload, 'predictive_stop_lead_g', 0, 20, errors);
  optionalNumberRange(payload, 'brew_ratio', 0.1, 10, errors);
  requireNumberRange(payload, 'target_yield_g', 5, 100, errors);
  optionalNumberRange(payload, 'target_ratio', 1.2, 3.5, errors);
  optionalNumberRange(payload, 'shot_time_s', 0, 180, errors);
  optionalBoolean(payload, 'raw_profile_available', errors);
  optionalSha256(payload, 'raw_profile_hash', errors);
  optionalEnum(payload, 'grinder_calibration_mode', ['uncalibrated', 'relative_calibrated', 'absolute_display_calibrated'], errors);
  optionalEnum(payload, 'grinder_adjustment_mode', ['stepped', 'stepless'], errors);
  optionalEnum(payload, 'step_direction', ['higher_is_finer', 'higher_is_coarser'], errors);
  optionalString(payload, 'reference_label', 80, errors);
  optionalNumberRange(payload, 'microns_per_step', 0.1, 100, errors);
  optionalNumberRange(payload, 'relative_grind_steps_from_reference', -10_000, 10_000, errors);
  optionalNumberRange(payload, 'relative_grind_um_from_reference', -1_000_000, 1_000_000, errors);
  optionalNumberRange(payload, 'current_absolute_step', -10_000, 10_000, errors);
  optionalNumberRange(payload, 'absolute_reference_step', -10_000, 10_000, errors);
  optionalActionObserved(payload, errors);
  optionalIdentifier(payload, 'recommendation_id', errors);
  optionalNumberRange(payload, 'recommended_grind_delta_steps_from_current', -1000, 1000, errors);
  optionalNumberRange(payload, 'recommended_grind_delta_um_from_current', -100_000, 100_000, errors);
  optionalNumberRange(payload, 'recommended_projected_relative_step_from_reference', -10_000, 10_000, errors);
  optionalNumberRange(payload, 'recommended_dose_g', 5, 30, errors);
  optionalNumberRange(payload, 'recommended_target_yield_g', 5, 100, errors);
  optionalNumberRange(payload, 'recommended_target_ratio', 1.2, 3.5, errors);
  optionalEnum(payload, 'recommendation_decision', ['accepted', 'edited', 'ignored', 'dismissed', 'unknown'], errors);
  optionalEnum(payload, 'recommendation_followed', ['followed', 'partially_followed', 'not_followed', 'unknown'], errors);
  optionalEnum(payload, 'shot_type', ['espresso', 'utility_flush', 'cleaning', 'calibration', 'unknown'], errors);
  if (payload.shot_type !== undefined && payload.shot_type !== null && payload.shot_type !== 'espresso') {
    errors.push('non-espresso shot uploads are not trusted training shots');
    errors.push('shot_type must be espresso');
  }
  optionalBoolean(payload, 'exclude_from_local_optimization', errors);
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
  optionalString(payload, 'profile_id', 120, errors);
  optionalString(payload, 'profile_label', 120, errors);
  optionalString(payload, 'profile_type', 80, errors);
  optionalIntegerRange(payload, 'profile_phase_count', 0, 100, errors);
  optionalIntegerRange(payload, 'final_phase_index', 0, 100, errors);
  optionalString(payload, 'final_phase_name', 120, errors);
  optionalEnum(payload, 'final_phase_type', ['preinfusion', 'brew'], errors);
  optionalNumberRange(payload, 'final_phase_elapsed_s', 0, 600, errors);
  optionalEnum(payload, 'final_pump_target', ['simple', 'pressure', 'flow'], errors);
  optionalNumberRange(payload, 'final_target_pressure', 0, 15, errors);
  optionalNumberRange(payload, 'final_target_flow', 0, 25, errors);
  optionalBoolean(payload, 'final_valve_open', errors);
  requireNumberRange(payload, 'profile_temperature_c', 0, 160, errors);
  requireNumberRange(payload, 'final_phase_temperature_c', 0, 160, errors);
  optionalNumericProfileVector(payload, 'beverage_flow_profile', 0, 20, errors);
  optionalNumericProfileVector(payload, 'temperature_profile', 0, 160, errors);
  optionalNumericProfileVector(payload, 'target_temperature_profile', 0, 160, errors);
  optionalPumpTargetModeProfile(payload, 'pump_target_mode_profile', errors);
  optionalFixedCadenceSequence(payload.fixed_cadence_sequence, errors);
  optionalEnum(payload, 'shot_end_state', ['finished', 'manual_or_interrupted', 'unknown'], errors);
  optionalNumberRange(payload, 'created_at', 0, Number.MAX_SAFE_INTEGER, errors);
  optionalNumberRange(payload, 'updated_at', 0, Number.MAX_SAFE_INTEGER, errors);
  const profile = payload.profile_resampled;
  if (profile !== undefined) {
    if (!Array.isArray(profile) || profile.length !== 5) {
      errors.push('profile_resampled must have 5 channels');
    } else {
      validateProfileResampled(profile, errors);
    }
  }
}

function validateRecommendationRecord(payload: JsonRecord, errors: string[]) {
  requireNumberRange(payload, 'schema_version', SUPPORTED_SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSION, errors);
  requireIdentifier(payload, 'recommendation_id', errors);
  requireIdentifier(payload, 'install_id', errors);
  requireIdentifier(payload, 'machine_id', errors);
  optionalIdentifier(payload, 'bean_context_id', errors);
  optionalString(payload, 'grinder_context_id', 120, errors);
  optionalString(payload, 'profile_id', 120, errors);
  optionalSha256(payload, 'raw_profile_hash', errors);
  requireTasteGoal(payload, errors);
  optionalNumberRange(payload, 'created_at', 0, Number.MAX_SAFE_INTEGER, errors);
  optionalNumberRange(payload, 'updated_at', 0, Number.MAX_SAFE_INTEGER, errors);
  optionalNumberRange(payload, 'expires_at', 0, Number.MAX_SAFE_INTEGER, errors);
  optionalNumberRange(payload, 'grind_delta_steps_from_current', -1000, 1000, errors);
  optionalNumberRange(payload, 'grind_delta_um_from_current', -100_000, 100_000, errors);
  optionalNumberRange(payload, 'projected_relative_step_from_reference', -10_000, 10_000, errors);
  optionalNumberRange(payload, 'projected_relative_grind_um_from_reference', -1_000_000, 1_000_000, errors);
  optionalEnum(payload, 'grinder_calibration_mode', ['uncalibrated', 'relative_calibrated', 'absolute_display_calibrated'], errors);
  optionalEnum(payload, 'grinder_adjustment_mode', ['stepped', 'stepless'], errors);
  optionalEnum(payload, 'step_direction', ['higher_is_finer', 'higher_is_coarser'], errors);
  optionalString(payload, 'reference_label', 80, errors);
  optionalNumberRange(payload, 'microns_per_step', 0.1, 100, errors);
  optionalNumberRange(payload, 'current_absolute_step', -10_000, 10_000, errors);
  optionalNumberRange(payload, 'absolute_reference_step', -10_000, 10_000, errors);
  optionalNumberRange(payload, 'projected_absolute_step', -10_000, 10_000, errors);
  requireNumberRange(payload, 'next_dose_g', 5, 30, errors);
  requireNumberRange(payload, 'target_yield_g', 5, 100, errors);
  requireNumberRange(payload, 'target_ratio', 1.2, 3.5, errors);
  optionalEnum(payload, 'mode', ['cpbo_global_previous', 'cpbo_best_incumbent'], errors);
  optionalNumberRange(payload, 'confidence', 0, 1, errors);
  optionalString(payload, 'reason', 500, errors);
  optionalEnum(payload, 'status', ['pending', 'shown', 'accepted', 'edited', 'ignored', 'expired', 'used', 'superseded'], errors);
  optionalIntegerRange(payload, 'shown_count', 0, 1_000_000, errors);
  optionalNumberRange(payload, 'accepted_at', 0, Number.MAX_SAFE_INTEGER, errors);
  optionalNumberRange(payload, 'ignored_at', 0, Number.MAX_SAFE_INTEGER, errors);
  optionalNumberRange(payload, 'edited_at', 0, Number.MAX_SAFE_INTEGER, errors);
  optionalNumberRange(payload, 'used_at', 0, Number.MAX_SAFE_INTEGER, errors);
  optionalNumberRange(payload, 'superseded_at', 0, Number.MAX_SAFE_INTEGER, errors);
  optionalIdentifier(payload, 'source_shot_id', errors);
  optionalIdentifier(payload, 'optimization_run_id', errors);
  optionalIdentifier(payload, 'comparison_anchor_shot_id', errors);
  optionalEnum(payload, 'comparison_mode', ['global_previous', 'best_incumbent'], errors);
  optionalBoolean(payload, 'preference_feedback_required', errors);
  if (payload.mode === 'cpbo_global_previous' || payload.mode === 'cpbo_best_incumbent') {
    if (!payload.optimization_run_id) errors.push('CPBO recommendation requires optimization_run_id');
    if (!payload.comparison_anchor_shot_id) errors.push('CPBO recommendation requires comparison_anchor_shot_id');
    if (payload.preference_feedback_required !== true) {
      errors.push('CPBO recommendation requires preference_feedback_required=true');
    }
    const expectedMode = payload.mode === 'cpbo_global_previous'
      ? 'global_previous'
      : 'best_incumbent';
    if (payload.comparison_mode !== expectedMode) {
      errors.push('CPBO recommendation comparison_mode does not match mode');
    }
  }
  optionalEnum(payload, 'apply_status', ['unknown', 'applied', 'partially_applied', 'manual_required', 'failed'], errors);
  optionalNumberRange(payload, 'apply_acknowledged_at', 0, Number.MAX_SAFE_INTEGER, errors);
  optionalObject(payload, 'applied_fields', errors);
  optionalStringList(payload, 'manual_fields', errors);
  optionalString(payload, 'apply_error', 500, errors);
}

function validateComparisonRecord(payload: JsonRecord, errors: string[]) {
  requireNumberRange(payload, 'schema_version', SUPPORTED_SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSION, errors);
  requireIdentifier(payload, 'comparison_id', errors);
  requireIdentifier(payload, 'optimization_run_id', errors);
  requireIdentifier(payload, 'new_shot_id', errors);
  requireIdentifier(payload, 'anchor_shot_id', errors);
  requireIdentifier(payload, 'install_id', errors);
  requireIdentifier(payload, 'machine_id', errors);
  optionalIdentifier(payload, 'machine_adapter', errors);
  optionalIdentifier(payload, 'recommendation_id', errors);
  optionalIdentifier(payload, 'bean_context_id', errors);
  optionalString(payload, 'grinder_context_id', 120, errors);
  optionalString(payload, 'profile_id', 120, errors);
  optionalSha256(payload, 'raw_profile_hash', errors);
  requireTasteGoal(payload, errors);
  requireNumberRange(payload, 'created_at', 0, Number.MAX_SAFE_INTEGER, errors);
  optionalEnum(payload, 'label', ['new_better', 'anchor_better', 'tie'], errors);
  if (payload.label === undefined || payload.label === null) errors.push('label is required');
  optionalEnum(payload, 'comparison_mode', ['global_previous', 'best_incumbent'], errors);
  if (payload.comparison_mode === undefined || payload.comparison_mode === null) {
    errors.push('comparison_mode is required');
  }
  if (payload.new_shot_id === payload.anchor_shot_id) {
    errors.push('comparison requires distinct physical shots');
  }
}

function requireTasteGoal(payload: JsonRecord, errors: string[]) {
  const value = payload.taste_goal;
  if (!value || Array.isArray(value) || typeof value !== 'object') {
    errors.push('taste_goal must be an object');
    return;
  }
  const goal = value as JsonRecord;
  const allowed = new Set(['schema_version', 'mode', 'targets']);
  const unknown = Object.keys(goal).filter(key => !allowed.has(key));
  const missing = [...allowed].filter(key => !(key in goal));
  if (unknown.length > 0 || missing.length > 0) {
    errors.push('taste_goal fields are invalid');
    return;
  }
  if (goal.schema_version !== 1) {
    errors.push('taste_goal schema_version is unsupported');
  }
  if (goal.mode !== 'balanced' && goal.mode !== 'custom') {
    errors.push('taste_goal mode is invalid');
  }
  if (!goal.targets || Array.isArray(goal.targets) || typeof goal.targets !== 'object') {
    errors.push('taste_goal targets must be an object');
    return;
  }
  const targets = goal.targets as JsonRecord;
  const targetEntries = Object.entries(targets);
  if (targetEntries.some(([attribute, level]) => !tasteGoalAttributes.has(attribute) || !tasteGoalLevels.has(String(level)))) {
    errors.push('taste_goal targets contain invalid values');
  }
  if (goal.mode === 'balanced' && targetEntries.length > 0) {
    errors.push('balanced taste_goal cannot contain targets');
  }
  if (goal.mode === 'custom' && targetEntries.length === 0) {
    errors.push('custom taste_goal requires at least one target');
  }
}

function requireString(payload: JsonRecord, key: string, errors: string[]) {
  const value = payload[key];
  if (typeof value !== 'string' || value.trim().length === 0) {
    errors.push(`${key} is required`);
    return;
  }
  if (value.length > 160 || UNSAFE_TEXT.test(value)) {
    errors.push(`${key} contains unsafe characters`);
  }
}

function requireIdentifier(payload: JsonRecord, key: string, errors: string[]) {
  const value = payload[key];
  if (typeof value !== 'string' || value.trim().length === 0) {
    errors.push(`${key} is required`);
    return;
  }
  if (!safeIdentifier(value)) {
    errors.push(`${key} contains unsafe characters`);
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
  if (typeof payload[key] !== 'string' || String(payload[key]).length > maxLength || UNSAFE_TEXT.test(String(payload[key]))) {
    errors.push(`${key} must be a short string`);
  }
}

function optionalIdentifier(payload: JsonRecord, key: string, errors: string[]) {
  if (payload[key] === undefined || payload[key] === null) {
    return;
  }
  if (typeof payload[key] !== 'string' || !safeIdentifier(String(payload[key]))) {
    errors.push(`${key} contains unsafe characters`);
  }
}

function optionalSha256(payload: JsonRecord, key: string, errors: string[]) {
  if (payload[key] === undefined || payload[key] === null) {
    return;
  }
  if (typeof payload[key] !== 'string' || !SHA256_HEX.test(String(payload[key]))) {
    errors.push(`${key} must be a sha256 hex digest`);
  }
}

function optionalIntegerRange(payload: JsonRecord, key: string, min: number, max: number, errors: string[]) {
  if (payload[key] === undefined || payload[key] === null) {
    return;
  }
  const value = payload[key];
  if (typeof value !== 'number' || !Number.isInteger(value) || value < min || value > max) {
    errors.push(`${key} out of range`);
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

function optionalStringList(payload: JsonRecord, key: string, errors: string[]) {
  if (payload[key] === undefined || payload[key] === null) {
    return;
  }
  const value = payload[key];
  if (!Array.isArray(value)) {
    errors.push(`${key} must be a list`);
    return;
  }
  if (value.some(item => typeof item !== 'string' || item.length > 80 || UNSAFE_TEXT.test(item))) {
    errors.push(`${key} contains invalid values`);
  }
}

function optionalStringListEnum(payload: JsonRecord, key: string, allowed: string[], errors: string[]) {
  if (payload[key] === undefined || payload[key] === null) {
    return;
  }
  const value = payload[key];
  if (!Array.isArray(value)) {
    errors.push(`${key} must be a list`);
    return;
  }
  if (value.some(item => typeof item !== 'string' || !allowed.includes(item))) {
    errors.push(`${key} contains invalid values`);
  }
}

function optionalObject(payload: JsonRecord, key: string, errors: string[]) {
  if (payload[key] === undefined || payload[key] === null) {
    return;
  }
  const value = payload[key];
  if (!value || Array.isArray(value) || typeof value !== 'object') {
    errors.push(`${key} must be an object`);
    return;
  }
  const entries = Object.entries(value as JsonRecord);
  if (entries.length > 25) {
    errors.push(`${key} must be an object`);
    return;
  }
  for (const [entryKey, entryValue] of entries) {
    if (entryKey.length > 80 || UNSAFE_TEXT.test(entryKey)) {
      errors.push(`${key} contains unsafe keys`);
      return;
    }
    if (
      entryValue !== null &&
      typeof entryValue !== 'string' &&
      typeof entryValue !== 'number' &&
      typeof entryValue !== 'boolean'
    ) {
      errors.push(`${key} contains unsafe values`);
      return;
    }
    if (typeof entryValue === 'string' && (entryValue.length > 160 || UNSAFE_TEXT.test(entryValue))) {
      errors.push(`${key} contains unsafe values`);
      return;
    }
    if (typeof entryValue === 'number' && !Number.isFinite(entryValue)) {
      errors.push(`${key} contains unsafe values`);
      return;
    }
  }
}

function optionalActionObserved(payload: JsonRecord, errors: string[]) {
  const value = payload.action_observed;
  if (value === undefined || value === null) {
    return;
  }
  if (!value || Array.isArray(value) || typeof value !== 'object') {
    errors.push('action_observed must be an object');
    return;
  }
  const observed = value as JsonRecord;
  const allowed = new Set(['grind', 'dose', 'target_yield']);
  const unknown = Object.keys(observed).filter(key => !allowed.has(key));
  if (unknown.length > 0) {
    errors.push(`action_observed contains unknown fields: ${unknown.slice(0, 10).join(', ')}`);
  }
  for (const field of allowed) {
    if (typeof observed[field] !== 'boolean') {
      errors.push(`action_observed.${field} must be boolean`);
    }
  }
  if (observed.grind === true) {
    const hasRelativeGrind = payload.relative_grind_steps_from_reference !== undefined &&
      payload.relative_grind_steps_from_reference !== null;
    const hasAbsolutePair = payload.current_absolute_step !== undefined &&
      payload.current_absolute_step !== null &&
      payload.absolute_reference_step !== undefined &&
      payload.absolute_reference_step !== null;
    if (!hasRelativeGrind && !hasAbsolutePair) {
      errors.push('action_observed.grind cannot be true without a grind measurement');
    }
  }
  if (observed.dose === true) {
    const measured = payload.dose_observed === true && payload.dose_in_g !== undefined && payload.dose_in_g !== null;
    const confirmed = payload.dose_target_confirmed === true;
    if (!measured && !confirmed) {
      errors.push('action_observed.dose cannot be true without a measured or confirmed dose');
    }
  }
}

function validateProfileResampled(profile: unknown[], errors: string[]) {
  const ranges: Array<[number, number, string]> = [
    [0, 20, 'pressure'],
    [0, 15, 'target_pressure'],
    [0, 20, 'pump_flow'],
    [0, 20, 'target_flow'],
    [-1, 120, 'weight'],
  ];
  for (let channelIndex = 0; channelIndex < 5; channelIndex += 1) {
    const channel = profile[channelIndex];
    const [min, max, label] = ranges[channelIndex];
    if (!Array.isArray(channel) || channel.length !== 100) {
      errors.push(`profile_resampled ${label} channel must have exactly 100 samples`);
      continue;
    }
    if (!channelNumericFinite(channel)) {
      errors.push(`profile_resampled ${label} contains non-finite or nonnumeric values`);
      continue;
    }
    if (!channelInRange(channel, min, max)) {
      if (label === 'pump_flow' || label === 'target_flow') {
        continue;
      }
      errors.push(`profile_resampled ${label} out of range`);
    }
  }
}

function optionalNumericProfileVector(payload: JsonRecord, key: string, min: number, max: number, errors: string[]) {
  if (payload[key] === undefined || payload[key] === null) {
    return;
  }
  const channel = payload[key];
  if (!Array.isArray(channel) || channel.length !== 100) {
    errors.push(`${key} must have exactly 100 samples`);
    return;
  }
  if (!channelNumericFinite(channel)) {
    errors.push(`${key} contains non-finite or nonnumeric values`);
    return;
  }
  if (!channelInRange(channel, min, max)) {
    errors.push(`${key} out of range`);
  }
}

function optionalPumpTargetModeProfile(payload: JsonRecord, key: string, errors: string[]) {
  if (payload[key] === undefined || payload[key] === null) {
    return;
  }
  const channel = payload[key];
  if (!Array.isArray(channel) || channel.length !== 100) {
    errors.push(`${key} must have exactly 100 samples`);
    return;
  }
  if (channel.some(value => typeof value !== 'number' || !Number.isInteger(value) || value < 0 || value > 2)) {
    errors.push(`${key} contains invalid pump target mode values`);
  }
}

function optionalFixedCadenceSequence(value: unknown, errors: string[]) {
  if (value === undefined || value === null) {
    return;
  }
  if (!value || Array.isArray(value) || typeof value !== 'object') {
    errors.push('fixed_cadence_sequence must be an object');
    return;
  }
  const sequence = value as JsonRecord;
  const numericRanges: Record<string, [number, number]> = {
    pressure_bar: [0, 15],
    pressure_target_bar: [0, 15],
    pump_flow_ml_s: [0, 20],
    pump_flow_target_ml_s: [0, 20],
    beverage_flow_g_s: [0, 20],
    weight_g: [-1, 120],
    temperature_c: [0, 160],
    temperature_target_c: [0, 160],
  };
  const allowed = new Set(['sample_interval_ms', 'pump_target_mode', 'valve_open', ...Object.keys(numericRanges)]);
  const unknown = Object.keys(sequence).filter(key => !allowed.has(key)).sort();
  if (unknown.length > 0) {
    errors.push(`fixed_cadence_sequence contains unknown fields: ${unknown.slice(0, 10).join(', ')}`);
  }
  if (sequence.sample_interval_ms !== 250) {
    errors.push('fixed_cadence_sequence.sample_interval_ms must be 250');
  }

  const lengths = new Set<number>();
  for (const [key, [min, max]] of Object.entries(numericRanges)) {
    const channel = sequence[key];
    if (!Array.isArray(channel)) {
      errors.push(`fixed_cadence_sequence.${key} must be a list`);
      continue;
    }
    lengths.add(channel.length);
    if (!channelNumericFinite(channel)) {
      errors.push(`fixed_cadence_sequence.${key} contains non-finite or nonnumeric values`);
    } else if (!channelInRange(channel, min, max)) {
      errors.push(`fixed_cadence_sequence.${key} out of range`);
    }
  }

  const pumpModes = sequence.pump_target_mode;
  if (!Array.isArray(pumpModes)) {
    errors.push('fixed_cadence_sequence.pump_target_mode must be a list');
  } else {
    lengths.add(pumpModes.length);
    if (pumpModes.some(item => typeof item !== 'number' || !Number.isInteger(item) || item < 0 || item > 2)) {
      errors.push('fixed_cadence_sequence.pump_target_mode contains invalid values');
    }
  }

  const valveOpen = sequence.valve_open;
  if (!Array.isArray(valveOpen)) {
    errors.push('fixed_cadence_sequence.valve_open must be a list');
  } else {
    lengths.add(valveOpen.length);
    if (valveOpen.some(item => typeof item !== 'boolean')) {
      errors.push('fixed_cadence_sequence.valve_open contains invalid values');
    }
  }

  if (lengths.size !== 1) {
    errors.push('fixed_cadence_sequence channels must have matching lengths');
  } else {
    const stepCount = lengths.values().next().value as number;
    if (stepCount < 2 || stepCount > 500) {
      errors.push('fixed_cadence_sequence must contain 2..500 steps');
    }
  }
}

function channelInRange(channel: unknown[], min: number, max: number): boolean {
  return channel.every(
    value => typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max,
  );
}

function channelNumericFinite(channel: unknown[]): boolean {
  return channel.every(value => typeof value === 'number' && Number.isFinite(value));
}

function rejectUnknownFields(payload: JsonRecord, allowed: Set<string>, errors: string[]) {
  const unknown = Object.keys(payload).filter(key => !allowed.has(key)).sort();
  if (unknown.length > 0) {
    errors.push(`unknown fields: ${unknown.slice(0, 10).join(', ')}`);
  }
}

function sanitizePayload(payload: JsonRecord): JsonRecord {
  const allowed = payload.event_type === 'shot_record'
    ? allowedShotFields
    : payload.event_type === 'recommendation_record'
      ? allowedRecommendationFields
      : allowedComparisonFields;
  const sanitized: JsonRecord = {};
  for (const key of allowed) {
    if (Object.prototype.hasOwnProperty.call(payload, key)) {
      sanitized[key] = payload[key];
    }
  }
  return sanitized;
}

function safeIdentifier(value: string): boolean {
  return SAFE_IDENTIFIER.test(value);
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
  payloadHash: string,
): Promise<boolean> {
  const { data, error } = await supabase
    .from('raw_upload_queue')
    .select('upload_id')
    .eq('install_id', installId)
    .eq('upload_id', uploadId)
    .eq('payload_hash', payloadHash)
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
