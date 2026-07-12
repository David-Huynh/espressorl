from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from typing import Any

import paho.mqtt.client as mqtt

from espresso_rl.config import Config
from espresso_rl.domain.events import (
    LocalResetEvent,
    MachineStateEvent,
    OptimizerSettingsEvent,
    PreferenceFeedbackEvent,
    RecommendationApplyEvent,
    RecommendationDecisionEvent,
    ShotCorrectionEvent,
    ShotProfileEvent,
    UploadQueueMaintenanceEvent,
)
from espresso_rl.domain.models import Recommendation
from espresso_rl.domain.models import new_id

logger = logging.getLogger(__name__)

SHOT_TOPIC = "gaggimate/+/shot/profile"
PREFERENCE_TOPIC = "gaggimate/+/rl/preference"
CORRECTION_TOPIC = "gaggimate/+/rl/shot/correction"
UPLOAD_REQUEUE_TOPIC = "gaggimate/+/rl/upload/requeue"
DECISION_TOPIC = "gaggimate/+/rl/recommendation/decision"
APPLY_TOPIC = "gaggimate/+/rl/recommendation/apply"
MACHINE_STATE_TOPIC = "gaggimate/+/machine/state"
OPTIMIZER_SETTINGS_TOPIC = "gaggimate/+/rl/settings"
LOCAL_RESET_TOPIC = "gaggimate/+/rl/local/reset"
_SHOT_FINISH_SETTLE_SAMPLES = 2
_SHOT_FINISH_SETTLE_MAX_DELTA_G = 1.0
_PREFERENCE_FIELDS = frozenset(
    {
        "event_type",
        "schema_version",
        "optimization_run_id",
        "new_shot_id",
        "anchor_shot_id",
        "label",
        "comparison_mode",
        "install_id",
        "machine_id",
        "timestamp",
        "source",
    }
)

class GaggimateMQTTClient:
    """Gaggimate MQTT adapter. Machine-specific topics stay out of core."""

    def __init__(
        self,
        config: Config,
        on_shot: Callable[[ShotProfileEvent], None],
        on_correction: Callable[[ShotCorrectionEvent], None],
        on_upload_maintenance: Callable[[UploadQueueMaintenanceEvent], None],
        on_decision: Callable[[RecommendationDecisionEvent], None],
        on_apply: Callable[[RecommendationApplyEvent], None],
        on_machine_state: Callable[[MachineStateEvent], None],
        on_preference: Callable[[PreferenceFeedbackEvent], None] | None = None,
        on_optimizer_settings: Callable[[OptimizerSettingsEvent], None] | None = None,
        on_local_reset: Callable[[LocalResetEvent], None] | None = None,
    ) -> None:
        self._config = config
        self._on_shot = on_shot
        self._on_correction = on_correction
        self._on_upload_maintenance = on_upload_maintenance
        self._on_decision = on_decision
        self._on_apply = on_apply
        self._on_machine_state = on_machine_state
        self._on_preference = on_preference or (lambda event: None)
        self._on_optimizer_settings = on_optimizer_settings or (lambda event: None)
        self._on_local_reset = on_local_reset or (lambda event: None)
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if config.mqtt_user:
            self._client.username_pw_set(config.mqtt_user, config.mqtt_password)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

    def start(self) -> None:
        self._client.connect(self._config.mqtt_host, self._config.mqtt_port, keepalive=60)
        self._client.loop_start()

    def stop(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()

    def publish_recommendation(self, recommendation: Recommendation) -> None:
        machine_topic_id = _machine_topic_id(recommendation.machine_id)
        topic = f"gaggimate/{machine_topic_id}/rl/recommendation"
        payload = {
            "event_type": "recommendation",
            "schema_version": 1,
            "shot_id": recommendation.source_shot_id,
            "recommendation_id": recommendation.recommendation_id,
            "optimization_run_id": recommendation.optimization_run_id,
            "comparison_anchor_shot_id": recommendation.comparison_anchor_shot_id,
            "comparison_mode": recommendation.comparison_mode,
            "preference_feedback_required": recommendation.preference_feedback_required,
            "install_id": recommendation.install_id,
            "machine_id": recommendation.machine_id,
            "bean_context_id": recommendation.bean_context_id,
            "grinder_context_id": recommendation.grinder_context_id,
            "grind_delta_steps_from_current": recommendation.grind_delta_steps_from_current,
            "grind_delta_um_from_current": recommendation.grind_delta_um_from_current,
            "projected_relative_step_from_reference": recommendation.projected_relative_step_from_reference,
            "projected_relative_grind_um_from_reference": recommendation.projected_relative_grind_um_from_reference,
            "next_dose_g": recommendation.next_dose_g,
            "target_yield_g": recommendation.target_yield_g,
            "target_ratio": recommendation.target_ratio,
            "mode": recommendation.mode.value,
            "confidence": recommendation.confidence,
            "reason": recommendation.reason,
            "status": recommendation.status.value,
            "grinder_calibration_mode": recommendation.grinder_calibration_mode.value,
            "step_direction": recommendation.grinder_step_direction.value,
            "grinder_adjustment_mode": recommendation.grinder_adjustment_mode.value,
            "reference_label": recommendation.grinder_reference_label,
            "current_absolute_step": recommendation.current_absolute_step,
            "absolute_reference_step": recommendation.absolute_reference_step,
            "projected_absolute_step": recommendation.projected_absolute_step,
            "profile_id": recommendation.profile_id,
            "raw_profile_hash": recommendation.raw_profile_hash,
            "expires_at": recommendation.expires_at,
        }
        self._client.publish(topic, json.dumps(payload), qos=1, retain=True)
        logger.info("Published recommendation %s to %s", recommendation.recommendation_id, topic)

    def clear_recommendation(self, machine_id: str) -> None:
        for machine_topic_id in _machine_topic_id_variants(machine_id):
            topic = f"gaggimate/{machine_topic_id}/rl/recommendation"
            self._client.publish(topic, "", qos=1, retain=True)
            logger.info("Cleared retained recommendation on %s", topic)

    def publish_status(self, machine_id: str, status: dict[str, Any]) -> None:
        machine_topic_id = _machine_topic_id(machine_id)
        topic = f"gaggimate/{machine_topic_id}/rl/status"
        payload = {
            "event_type": "espresso_rl_status",
            "schema_version": 1,
            "machine_id": machine_id,
            **status,
        }
        self._client.publish(topic, json.dumps(payload), qos=1, retain=True)
        logger.info("Published EspressoRL status to %s", topic)

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: object,
        flags: mqtt.ConnectFlags,
        reason_code: mqtt.ReasonCode,
        properties: mqtt.Properties | None,
    ) -> None:
        if reason_code == 0:
            client.subscribe(SHOT_TOPIC)
            client.subscribe(PREFERENCE_TOPIC)
            client.subscribe(CORRECTION_TOPIC)
            client.subscribe(UPLOAD_REQUEUE_TOPIC)
            client.subscribe(DECISION_TOPIC)
            client.subscribe(APPLY_TOPIC)
            client.subscribe(MACHINE_STATE_TOPIC)
            client.subscribe(OPTIMIZER_SETTINGS_TOPIC)
            client.subscribe(LOCAL_RESET_TOPIC)
            logger.info(
                "Subscribed to %s, %s, %s, %s, %s, %s, %s, %s, %s",
                SHOT_TOPIC,
                PREFERENCE_TOPIC,
                CORRECTION_TOPIC,
                UPLOAD_REQUEUE_TOPIC,
                DECISION_TOPIC,
                APPLY_TOPIC,
                MACHINE_STATE_TOPIC,
                OPTIMIZER_SETTINGS_TOPIC,
                LOCAL_RESET_TOPIC,
            )
        else:
            logger.error("MQTT connection refused: %s", reason_code)

    def _on_message(
        self,
        client: mqtt.Client,
        userdata: object,
        msg: mqtt.MQTTMessage,
    ) -> None:
        try:
            parts = msg.topic.split("/")
            mac = parts[1] if len(parts) > 1 else "unknown"
            payload_text = msg.payload.decode().strip()
            if not payload_text:
                logger.debug("Ignoring empty MQTT payload on %s", msg.topic)
                return
            payload = json.loads(payload_text)
            if msg.topic.endswith("/shot/profile"):
                self._on_shot(self.translate_shot_payload(payload, mac))
            elif msg.topic.endswith("/rl/preference"):
                self._on_preference(self.translate_preference_payload(payload, mac))
            elif msg.topic.endswith("/rl/shot/correction"):
                self._on_correction(self.translate_correction_payload(payload, mac))
            elif msg.topic.endswith("/rl/upload/requeue"):
                self._on_upload_maintenance(self.translate_upload_maintenance_payload(payload, mac))
            elif msg.topic.endswith("/rl/recommendation/decision"):
                self._on_decision(self.translate_decision_payload(payload, mac))
            elif msg.topic.endswith("/rl/recommendation/apply"):
                self._on_apply(self.translate_apply_payload(payload, mac))
            elif msg.topic.endswith("/machine/state"):
                self._on_machine_state(self.translate_machine_state_payload(payload, mac))
            elif msg.topic.endswith("/rl/settings"):
                self._on_optimizer_settings(self.translate_optimizer_settings_payload(payload, mac))
            elif msg.topic.endswith("/rl/local/reset"):
                self._on_local_reset(self.translate_local_reset_payload(payload, mac))
        except Exception:
            logger.exception("Error handling message on %s", msg.topic)

    def translate_shot_payload(self, payload: dict[str, Any], mac: str) -> ShotProfileEvent:
        install_id = str(payload.get("install_id") or self._config.install_id)
        machine_id = str(payload.get("machine_id") or f"gaggimate:{mac}")
        payload = dict(payload)
        shot_time_s = payload.get("shot_time_s")
        time_ms, trimmed_to_shot_time = _trim_profile_payload_to_shot_time(
            payload,
            payload.get("time_ms") or payload.get("time") or [],
            shot_time_s,
        )
        weight = payload.get("weight") or payload.get("weight_g") or []
        n = len(time_ms)
        pressure = _channel(payload, "pressure", n)
        target_pressure = _channel(payload, "target_pressure", n)
        beverage_flow = _channel(payload, "flow", n)
        pump_flow = _channel(payload, "pump_flow", n)
        target_flow = _channel(payload, "target_flow", n)
        temperature = _optional_channel(payload, "temperature", n)
        target_temperature = _optional_channel(payload, "target_temperature", n)
        pump_target_mode = _optional_int_channel(payload, "pump_target_mode", n)
        valve_open = _optional_bool_channel(payload, "valve_open", n)
        weight = _channel({"weight": weight}, "weight", n)
        payload_target_yield_g = _positive_optional_float(payload.get("target_yield_g"))
        target_yield_observed = payload_target_yield_g is not None
        target_yield_g = payload_target_yield_g or self._config.initial_target_yield_g
        payload_dose_in_g = _positive_optional_float(payload.get("dose_in_g"))
        payload_dose_target_g = _positive_optional_float(payload.get("dose_target_g"))
        declared_dose_observed = _optional_bool(payload.get("dose_observed"))
        dose_observed = payload_dose_in_g is not None and declared_dose_observed is not False
        dose_in_g = payload_dose_in_g or payload_dose_target_g or self._config.initial_dose_g
        relative_grind_steps = _relative_grind_steps_from_payload(
            payload,
            self._config.initial_relative_grind_steps_from_reference,
        )
        inferred_grind_observed = (
            _optional_float(payload.get("relative_grind_steps_from_reference")) is not None
            or (
                _optional_float(payload.get("current_absolute_step")) is not None
                and _optional_float(payload.get("absolute_reference_step")) is not None
            )
        )
        declared_grind_observed = _optional_bool(payload.get("grind_observed"))
        grind_observed = inferred_grind_observed and declared_grind_observed is not False
        beverage_out_g = payload.get("beverage_out_g")
        if (beverage_out_g is None or trimmed_to_shot_time) and weight:
            beverage_out_g = float(weight[-1])
        beverage_out_g = _positive_optional_float(beverage_out_g)
        if time_ms and (shot_time_s is None or trimmed_to_shot_time):
            shot_time_s = float(time_ms[-1]) / 1000.0
        return ShotProfileEvent(
            shot_id=str(payload.get("shot_id") or payload.get("id") or new_id("shot")),
            install_id=install_id,
            machine_id=machine_id,
            machine_adapter="gaggimate",
            timestamp=int(payload.get("timestamp", self._config.now())),
            time_ms=list(time_ms),
            pressure=pressure,
            target_pressure=target_pressure,
            pump_flow=pump_flow,
            target_flow=target_flow,
            beverage_flow=beverage_flow,
            temperature=temperature,
            target_temperature=target_temperature,
            pump_target_mode=pump_target_mode,
            valve_open=valve_open,
            weight=weight,
            microns_per_step=float(payload.get("microns_per_step", self._config.microns_per_step)),
            relative_grind_steps_from_reference=relative_grind_steps,
            grind_observed=grind_observed,
            dose_in_g=dose_in_g,
            dose_observed=dose_observed,
            target_yield_g=target_yield_g,
            target_yield_observed=target_yield_observed,
            beverage_out_g=beverage_out_g,
            shot_time_s=_optional_float(shot_time_s),
            bean_context_id=payload.get("bean_context_id", self._config.bean_context_id),
            bean_context_name=_optional_string(payload.get("bean_context_name")),
            grinder_context_id=payload.get("grinder_context_id", self._config.grinder_context_id),
            grinder_calibration_mode=payload.get("grinder_calibration_mode", "relative_calibrated"),
            grinder_step_direction=payload.get("step_direction", "higher_is_finer"),
            grinder_adjustment_mode=payload.get("grinder_adjustment_mode", "stepped"),
            grinder_reference_label=payload.get("reference_label", "reference"),
            current_absolute_step=_optional_float(payload.get("current_absolute_step")),
            absolute_reference_step=_optional_float(payload.get("absolute_reference_step")),
            recommendation_id=payload.get("recommendation_id"),
            shot_type=payload.get("shot_type", "espresso"),
            utility=bool(payload.get("utility", False)),
            exclude_from_local_optimization=bool(payload.get("exclude_from_local_optimization", False)),
            local_optimization_enabled=bool(payload.get("local_optimization_enabled", True)),
            community_upload_enabled=_optional_bool(payload.get("community_upload_enabled")),
            optimization_weight=_optional_float(payload.get("optimization_weight")),
            weight_source=_optional_string(payload.get("weight_source")),
            flow_source=_optional_string(payload.get("flow_source")),
            flow_units=_optional_string(payload.get("flow_units")),
            pump_flow_source=_optional_string(payload.get("pump_flow_source")),
            pump_flow_units=_optional_string(payload.get("pump_flow_units")),
            pump_flow_calibration_required=bool(payload.get("pump_flow_calibration_required", False)),
            profile_id=_optional_string(payload.get("profile_id")),
            profile_label=_optional_string(payload.get("profile_label")),
            raw_profile_hash=_optional_string(payload.get("raw_profile_hash")),
            profile_type=_optional_string(payload.get("profile_type")),
            profile_phase_count=_optional_int(payload.get("profile_phase_count")),
            final_phase_index=_optional_int(payload.get("final_phase_index")),
            final_phase_name=_optional_string(payload.get("final_phase_name")),
            final_phase_type=_optional_string(payload.get("final_phase_type")),
            final_phase_elapsed_s=_optional_float(payload.get("final_phase_elapsed_s")),
            final_pump_target=_optional_string(payload.get("final_pump_target")),
            final_target_pressure=_optional_float(payload.get("final_target_pressure")),
            final_target_flow=_optional_float(payload.get("final_target_flow")),
            final_valve_open=_optional_bool(payload.get("final_valve_open")),
            profile_temperature_c=_optional_float(payload.get("profile_temperature_c")),
            final_phase_temperature_c=_optional_float(payload.get("final_phase_temperature_c")),
            shot_end_state=_optional_string(payload.get("shot_end_state")),
        )

    def translate_preference_payload(
        self,
        payload: dict[str, Any],
        mac: str,
    ) -> PreferenceFeedbackEvent:
        _require_exact_object_fields(payload, _PREFERENCE_FIELDS, "preference feedback")
        if payload.get("event_type") != "preference_feedback":
            raise ValueError("preference feedback event_type is invalid")
        schema_version = _strict_non_negative_int(payload.get("schema_version"), "schema_version")
        if schema_version != 1:
            raise ValueError("preference feedback schema_version is unsupported")
        topic_machine_id = f"gaggimate:{mac}"
        machine_id = _required_bounded_string(payload.get("machine_id"), "machine_id", maximum=160)
        if not _same_gaggimate_machine_id(topic_machine_id, machine_id):
            raise ValueError("preference feedback machine_id does not match topic")
        return PreferenceFeedbackEvent(
            optimization_run_id=_required_bounded_string(
                payload.get("optimization_run_id"),
                "optimization_run_id",
                maximum=256,
            ),
            new_shot_id=_required_bounded_string(payload.get("new_shot_id"), "new_shot_id", maximum=256),
            anchor_shot_id=_required_bounded_string(
                payload.get("anchor_shot_id"),
                "anchor_shot_id",
                maximum=256,
            ),
            label=_required_bounded_string(payload.get("label"), "label", maximum=32),
            comparison_mode=_required_bounded_string(
                payload.get("comparison_mode"),
                "comparison_mode",
                maximum=32,
            ),
            install_id=_required_bounded_string(payload.get("install_id"), "install_id", maximum=160),
            machine_id=machine_id,
            timestamp=_strict_non_negative_int(payload.get("timestamp"), "timestamp"),
            source=_required_bounded_string(payload.get("source"), "source", maximum=80),
            schema_version=schema_version,
        )

    def translate_correction_payload(self, payload: dict[str, Any], mac: str) -> ShotCorrectionEvent:
        return ShotCorrectionEvent(
            shot_id=str(payload.get("shot_id") or ""),
            install_id=str(payload.get("install_id") or self._config.install_id),
            machine_id=str(payload.get("machine_id") or f"gaggimate:{mac}"),
            timestamp=int(payload.get("timestamp", self._config.now())),
            exclude_from_local_optimization=_optional_bool(payload.get("exclude_from_local_optimization")),
            shot_type=payload.get("shot_type"),
            grind_followed=_optional_bool(payload.get("grind_followed")),
            dose_followed=_optional_bool(payload.get("dose_followed")),
            yield_followed=_optional_bool(payload.get("yield_followed")),
            correction_tags=list(payload.get("correction_tags", [])),
            source=payload.get("source", "gaggimate_mqtt"),
        )

    def translate_upload_maintenance_payload(
        self,
        payload: dict[str, Any],
        mac: str,
    ) -> UploadQueueMaintenanceEvent:
        return UploadQueueMaintenanceEvent(
            install_id=str(payload.get("install_id") or self._config.install_id),
            machine_id=str(payload.get("machine_id") or f"gaggimate:{mac}"),
            timestamp=int(payload.get("timestamp", self._config.now())),
            action=payload.get("action", "requeue_valid_rejected"),
            limit=int(payload.get("limit", 25)),
            bean_context_id=payload.get("bean_context_id", self._config.bean_context_id),
            grinder_context_id=payload.get("grinder_context_id", self._config.grinder_context_id),
            local_record_id=_optional_string(payload.get("local_record_id")),
            source=payload.get("source", "gaggimate_mqtt"),
        )

    def translate_decision_payload(self, payload: dict[str, Any], mac: str) -> RecommendationDecisionEvent:
        return RecommendationDecisionEvent(
            recommendation_id=str(payload.get("recommendation_id") or ""),
            decision=payload.get("decision", "unknown"),
            timestamp=int(payload.get("timestamp", self._config.now())),
            install_id=payload.get("install_id", self._config.install_id),
            machine_id=payload.get("machine_id", f"gaggimate:{mac}"),
            edited_fields=dict(payload.get("edited_fields", {})),
            source=payload.get("source", "gaggimate_mqtt"),
        )

    def translate_apply_payload(self, payload: dict[str, Any], mac: str) -> RecommendationApplyEvent:
        return RecommendationApplyEvent(
            recommendation_id=str(payload.get("recommendation_id") or ""),
            status=payload.get("status", "unknown"),
            timestamp=int(payload.get("timestamp", self._config.now())),
            install_id=payload.get("install_id", self._config.install_id),
            machine_id=payload.get("machine_id", f"gaggimate:{mac}"),
            applied_fields=dict(payload.get("applied_fields") or {}),
            manual_fields=list(payload.get("manual_fields") or []),
            failed_fields=dict(payload.get("failed_fields") or {}),
            message=payload.get("message"),
            source=payload.get("source", "gaggimate_mqtt"),
        )

    def translate_machine_state_payload(self, payload: dict[str, Any], mac: str) -> MachineStateEvent:
        return MachineStateEvent(
            install_id=str(payload.get("install_id") or self._config.install_id),
            machine_id=str(payload.get("machine_id") or f"gaggimate:{mac}"),
            machine_adapter="gaggimate",
            timestamp=int(payload.get("timestamp", self._config.now())),
            state=payload.get("state", "unknown"),
            bean_context_id=payload.get("bean_context_id", self._config.bean_context_id),
            bean_context_name=_optional_string(payload.get("bean_context_name")),
            grinder_context_id=payload.get("grinder_context_id", self._config.grinder_context_id),
            grinder_calibration_mode=payload.get("grinder_calibration_mode", "relative_calibrated"),
            grinder_step_direction=payload.get("step_direction", "higher_is_finer"),
            grinder_adjustment_mode=payload.get("grinder_adjustment_mode", "stepped"),
            grinder_reference_label=payload.get("reference_label", "reference"),
            relative_grind_steps_from_reference=_relative_grind_steps_from_payload(
                payload,
                None,
            ),
            microns_per_step=_optional_float(payload.get("microns_per_step")),
            current_absolute_step=_optional_float(payload.get("current_absolute_step")),
            absolute_reference_step=_optional_float(payload.get("absolute_reference_step")),
            dose_in_g=_optional_float(payload.get("dose_in_g")),
            target_yield_g=_optional_float(payload.get("target_yield_g")),
            profile_id=_optional_string(payload.get("profile_id")),
            profile_label=_optional_string(payload.get("profile_label")),
            raw_profile_hash=_optional_string(payload.get("raw_profile_hash")),
            community_upload_enabled=_optional_bool(payload.get("community_upload_enabled")),
            source=payload.get("source", "gaggimate_mqtt"),
        )

    def translate_optimizer_settings_payload(self, payload: dict[str, Any], mac: str) -> OptimizerSettingsEvent:
        return OptimizerSettingsEvent(
            install_id=str(payload.get("install_id") or self._config.install_id),
            machine_id=str(payload.get("machine_id") or f"gaggimate:{mac}"),
            timestamp=int(payload.get("timestamp") or self._config.now()),
            schema_version=int(payload.get("schema_version", 1)),
            optimizer_mode=str(payload.get("optimizer_mode") or payload.get("mode") or self._config.optimizer_mode),
            bean_context_id=_optional_string(payload.get("bean_context_id")),
            grinder_context_id=_optional_string(payload.get("grinder_context_id")),
            source=payload.get("source", "gaggimate_mqtt"),
        )

    def translate_local_reset_payload(self, payload: dict[str, Any], mac: str) -> LocalResetEvent:
        return LocalResetEvent(
            install_id=str(payload.get("install_id") or self._config.install_id),
            machine_id=str(payload.get("machine_id") or f"gaggimate:{mac}"),
            timestamp=int(payload.get("timestamp") or self._config.now()),
            scope=str(payload.get("scope") or payload.get("reset_scope") or "all"),
            dry_run=bool(payload.get("dry_run", False)),
            source=payload.get("source", "gaggimate_mqtt"),
        )


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("optional numeric fields cannot be boolean")
    return float(value)


def _trim_profile_payload_to_shot_time(
    payload: dict[str, Any],
    time_values: Any,
    shot_time_s: Any,
) -> tuple[list[Any], bool]:
    time_ms = list(time_values or [])
    shot_time = _optional_float(shot_time_s)
    trimmed = False
    if shot_time is None or shot_time <= 0 or len(time_ms) < 2:
        return _trim_profile_payload_to_inactive_control_tail(payload, time_ms, trimmed)

    cutoff_ms = shot_time * 1000.0
    if float(time_ms[-1]) > cutoff_ms:
        keep_count = sum(1 for value in time_ms if float(value) <= cutoff_ms)
        keep_count = max(2, min(keep_count, len(time_ms)))
        if keep_count < len(time_ms):
            time_ms = _trim_profile_payload_samples(payload, time_ms, keep_count)
            trimmed = True

    return _trim_profile_payload_to_inactive_control_tail(payload, time_ms, trimmed)


def _trim_profile_payload_to_inactive_control_tail(
    payload: dict[str, Any],
    time_ms: list[Any],
    already_trimmed: bool,
) -> tuple[list[Any], bool]:
    keep_count = _inactive_control_tail_keep_count(payload, len(time_ms))
    if keep_count is None or keep_count >= len(time_ms):
        return time_ms, already_trimmed
    return _trim_profile_payload_samples(payload, time_ms, keep_count), True


def _trim_profile_payload_samples(payload: dict[str, Any], time_ms: list[Any], keep_count: int) -> list[Any]:
    for key in (
        "time_ms",
        "time",
        "pressure",
        "target_pressure",
        "flow",
        "pump_flow",
        "target_flow",
        "temperature",
        "target_temperature",
        "pump_target_mode",
        "valve_open",
        "weight",
        "weight_g",
    ):
        values = payload.get(key)
        if isinstance(values, (list, tuple)) and len(values) == len(time_ms):
            payload[key] = list(values)[:keep_count]
    return time_ms[:keep_count]


def _inactive_control_tail_keep_count(payload: dict[str, Any], n: int) -> int | None:
    if n < 3:
        return None
    target_pressure = payload.get("target_pressure")
    target_flow = payload.get("target_flow")
    if not isinstance(target_pressure, (list, tuple)) or not isinstance(target_flow, (list, tuple)):
        return None
    if len(target_pressure) != n or len(target_flow) != n:
        return None

    pump_target_mode = payload.get("pump_target_mode")
    valve_open = payload.get("valve_open")
    mode_available = isinstance(pump_target_mode, (list, tuple)) and len(pump_target_mode) == n
    valve_available = isinstance(valve_open, (list, tuple)) and len(valve_open) == n
    if not mode_available and not valve_available:
        return None

    saw_active_target = False
    for index in range(n):
        target_active = _profile_target_active(target_pressure[index]) or _profile_target_active(target_flow[index])
        if target_active:
            saw_active_target = True
            continue
        if not saw_active_target:
            continue
        mode_inactive = True
        if mode_available:
            mode_inactive = _optional_int(pump_target_mode[index]) == 0
        valve_inactive = True
        if valve_available:
            valve_inactive = _optional_bool(valve_open[index]) is False
        if mode_inactive and valve_inactive:
            return _settled_inactive_tail_keep_count(payload, index, n)
    return None


def _settled_inactive_tail_keep_count(payload: dict[str, Any], index: int, n: int) -> int:
    keep = max(2, index)
    weights = payload.get("weight")
    if not isinstance(weights, (list, tuple)) or len(weights) != n:
        weights = payload.get("weight_g")
    if not isinstance(weights, (list, tuple)) or len(weights) != n or index <= 0:
        return keep

    previous_weight = _optional_float(weights[index - 1])
    if previous_weight is None:
        return keep

    settled_keep = keep
    settle_end = min(n, index + _SHOT_FINISH_SETTLE_SAMPLES)
    for cursor in range(index, settle_end):
        current_weight = _optional_float(weights[cursor])
        if current_weight is None:
            break
        if not math.isfinite(previous_weight) or not math.isfinite(current_weight):
            break
        if abs(current_weight - previous_weight) > _SHOT_FINISH_SETTLE_MAX_DELTA_G:
            break
        settled_keep = cursor + 1
        previous_weight = current_weight
    return max(keep, settled_keep)


def _profile_target_active(value: Any) -> bool:
    parsed = _optional_float(value)
    return parsed is not None and abs(parsed) > 1e-6


def _relative_grind_steps_from_payload(payload: dict[str, Any], fallback: float | None) -> float | None:
    current_absolute_step = _optional_float(payload.get("current_absolute_step"))
    absolute_reference_step = _optional_float(payload.get("absolute_reference_step"))
    if current_absolute_step is not None and absolute_reference_step is not None:
        return current_absolute_step - absolute_reference_step
    return _optional_float(payload.get("relative_grind_steps_from_reference", fallback))


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("optional integer fields cannot be boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    raise ValueError("optional integer field is invalid")


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    raise ValueError("optional boolean field is invalid")


def _optional_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _required_bounded_string(value: Any, field_name: str, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized or len(normalized) > maximum:
        raise ValueError(f"{field_name} is invalid")
    return normalized


def _strict_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _require_exact_object_fields(
    value: Any,
    expected: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    actual = set(value)
    unknown = sorted(str(key) for key in actual - expected)
    missing = sorted(expected - actual)
    if unknown or missing:
        details = []
        if unknown:
            details.append(f"unsupported fields: {', '.join(unknown[:5])}")
        if missing:
            details.append(f"missing fields: {', '.join(missing[:5])}")
        raise ValueError(f"{label} fields are invalid ({'; '.join(details)})")
    return value


def _same_gaggimate_machine_id(left: str, right: str) -> bool:
    return _machine_topic_id(left).casefold() == _machine_topic_id(right).casefold()


def _machine_topic_id(machine_id: str) -> str:
    return machine_id.removeprefix("gaggimate:")


def _machine_topic_id_variants(machine_id: str) -> tuple[str, ...]:
    topic_id = _machine_topic_id(machine_id)
    variants: list[str] = []
    for candidate in (topic_id, topic_id.upper(), topic_id.lower()):
        if candidate and candidate not in variants:
            variants.append(candidate)
    return tuple(variants)


def _positive_optional_float(value: Any) -> float | None:
    parsed = _optional_float(value)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def _channel(payload: dict[str, Any], key: str, n: int) -> list[float]:
    values = payload.get(key)
    if values is None:
        return [0.0] * n
    result = list(values)
    if len(result) != n:
        return [0.0] * n
    return result


def _optional_channel(payload: dict[str, Any], key: str, n: int) -> list[float] | None:
    values = payload.get(key)
    if values is None:
        return None
    result = list(values)
    if len(result) != n:
        return None
    return result


def _optional_int_channel(payload: dict[str, Any], key: str, n: int) -> list[int] | None:
    values = payload.get(key)
    if values is None:
        return None
    result = list(values)
    if len(result) != n:
        return None
    return [_optional_int(value) or 0 for value in result]


def _optional_bool_channel(payload: dict[str, Any], key: str, n: int) -> list[bool] | None:
    values = payload.get(key)
    if values is None:
        return None
    result = list(values)
    if len(result) != n:
        return None
    parsed = [_optional_bool(value) for value in result]
    if any(value is None for value in parsed):
        return None
    return [bool(value) for value in parsed]
