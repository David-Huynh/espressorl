from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import paho.mqtt.client as mqtt

from espresso_rl.config import Config
from espresso_rl.domain.events import (
    MachineStateEvent,
    RecommendationApplyEvent,
    RecommendationDecisionEvent,
    ShotCorrectionEvent,
    ShotFeedbackEvent,
    ShotProfileEvent,
    UploadQueueMaintenanceEvent,
)
from espresso_rl.domain.models import Recommendation
from espresso_rl.domain.models import new_id

logger = logging.getLogger(__name__)

SHOT_TOPIC = "gaggimate/+/shot/profile"
FEEDBACK_TOPIC = "gaggimate/+/rl/rating"
CORRECTION_TOPIC = "gaggimate/+/rl/shot/correction"
UPLOAD_REQUEUE_TOPIC = "gaggimate/+/rl/upload/requeue"
DECISION_TOPIC = "gaggimate/+/rl/recommendation/decision"
APPLY_TOPIC = "gaggimate/+/rl/recommendation/apply"
MACHINE_STATE_TOPIC = "gaggimate/+/machine/state"


class GaggimateMQTTClient:
    """Gaggimate MQTT adapter. Machine-specific topics stay out of core."""

    def __init__(
        self,
        config: Config,
        on_shot: Callable[[ShotProfileEvent], None],
        on_feedback: Callable[[ShotFeedbackEvent], None],
        on_correction: Callable[[ShotCorrectionEvent], None],
        on_upload_maintenance: Callable[[UploadQueueMaintenanceEvent], None],
        on_decision: Callable[[RecommendationDecisionEvent], None],
        on_apply: Callable[[RecommendationApplyEvent], None],
        on_machine_state: Callable[[MachineStateEvent], None],
    ) -> None:
        self._config = config
        self._on_shot = on_shot
        self._on_feedback = on_feedback
        self._on_correction = on_correction
        self._on_upload_maintenance = on_upload_maintenance
        self._on_decision = on_decision
        self._on_apply = on_apply
        self._on_machine_state = on_machine_state
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
        machine_topic_id = recommendation.machine_id.removeprefix("gaggimate:")
        topic = f"gaggimate/{machine_topic_id}/rl/recommendation"
        payload = {
            "event_type": "recommendation",
            "schema_version": 1,
            "shot_id": recommendation.source_shot_id,
            "recommendation_id": recommendation.recommendation_id,
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
            "reference_label": recommendation.grinder_reference_label,
            "current_absolute_step": recommendation.current_absolute_step,
            "absolute_reference_step": recommendation.absolute_reference_step,
            "projected_absolute_step": recommendation.projected_absolute_step,
            "expires_at": recommendation.expires_at,
        }
        self._client.publish(topic, json.dumps(payload), qos=1, retain=True)
        logger.info("Published recommendation %s to %s", recommendation.recommendation_id, topic)

    def publish_status(self, machine_id: str, status: dict[str, Any]) -> None:
        machine_topic_id = machine_id.removeprefix("gaggimate:")
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
            client.subscribe(FEEDBACK_TOPIC)
            client.subscribe(CORRECTION_TOPIC)
            client.subscribe(UPLOAD_REQUEUE_TOPIC)
            client.subscribe(DECISION_TOPIC)
            client.subscribe(APPLY_TOPIC)
            client.subscribe(MACHINE_STATE_TOPIC)
            logger.info(
                "Subscribed to %s, %s, %s, %s, %s, %s, %s",
                SHOT_TOPIC,
                FEEDBACK_TOPIC,
                CORRECTION_TOPIC,
                UPLOAD_REQUEUE_TOPIC,
                DECISION_TOPIC,
                APPLY_TOPIC,
                MACHINE_STATE_TOPIC,
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
            payload = json.loads(msg.payload.decode())
            if msg.topic.endswith("/shot/profile"):
                self._on_shot(self.translate_shot_payload(payload, mac))
            elif msg.topic.endswith("/rl/rating"):
                self._on_feedback(self.translate_feedback_payload(payload, mac))
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
        except Exception:
            logger.exception("Error handling message on %s", msg.topic)

    def translate_shot_payload(self, payload: dict[str, Any], mac: str) -> ShotProfileEvent:
        install_id = str(payload.get("install_id") or self._config.install_id)
        machine_id = str(payload.get("machine_id") or f"gaggimate:{mac}")
        time_ms = payload.get("time_ms") or payload.get("time") or []
        weight = payload.get("weight") or payload.get("weight_g") or []
        n = len(time_ms)
        pressure = _channel(payload, "pressure", n)
        target_pressure = _channel(payload, "target_pressure", n)
        flow = _channel(payload, "flow", n)
        target_flow = _channel(payload, "target_flow", n)
        weight = _channel({"weight": weight}, "weight", n)
        target_yield_g = float(payload.get("target_yield_g", self._config.initial_target_yield_g))
        beverage_out_g = payload.get("beverage_out_g")
        if beverage_out_g is None and weight:
            beverage_out_g = float(weight[-1])
        beverage_out_g = _positive_optional_float(beverage_out_g)
        shot_time_s = payload.get("shot_time_s")
        if shot_time_s is None and time_ms:
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
            flow=flow,
            target_flow=target_flow,
            weight=weight,
            microns_per_step=float(payload.get("microns_per_step", self._config.microns_per_step)),
            relative_grind_steps_from_reference=_relative_grind_steps_from_payload(
                payload,
                self._config.initial_relative_grind_steps_from_reference,
            ),
            dose_in_g=float(payload.get("dose_in_g", self._config.initial_dose_g)),
            target_yield_g=target_yield_g,
            beverage_out_g=beverage_out_g,
            shot_time_s=_optional_float(shot_time_s),
            bean_context_id=payload.get("bean_context_id", self._config.bean_context_id),
            bean_context_name=_optional_string(payload.get("bean_context_name")),
            grinder_context_id=payload.get("grinder_context_id", self._config.grinder_context_id),
            grinder_calibration_mode=payload.get("grinder_calibration_mode", "relative_calibrated"),
            grinder_step_direction=payload.get("step_direction", "higher_is_finer"),
            grinder_reference_label=payload.get("reference_label", "reference"),
            current_absolute_step=_optional_float(payload.get("current_absolute_step")),
            absolute_reference_step=_optional_float(payload.get("absolute_reference_step")),
            recommendation_id=payload.get("recommendation_id"),
            shot_type=payload.get("shot_type", "espresso"),
            utility=bool(payload.get("utility", False)),
            exclude_from_local_optimization=bool(payload.get("exclude_from_local_optimization", False)),
            local_optimization_enabled=bool(payload.get("local_optimization_enabled", True)),
            optimization_weight=_optional_float(payload.get("optimization_weight")),
            rating_prompt_allowed=bool(payload.get("rating_prompt_allowed", True)),
            weight_source=_optional_string(payload.get("weight_source")),
            flow_source=_optional_string(payload.get("flow_source")),
            flow_units=_optional_string(payload.get("flow_units")),
            pump_flow_source=_optional_string(payload.get("pump_flow_source")),
            pump_flow_units=_optional_string(payload.get("pump_flow_units")),
            pump_flow_calibration_required=bool(payload.get("pump_flow_calibration_required", False)),
            profile_id=_optional_string(payload.get("profile_id")),
            profile_label=_optional_string(payload.get("profile_label")),
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

    def translate_feedback_payload(self, payload: dict[str, Any], mac: str) -> ShotFeedbackEvent:
        rating = payload.get("rating")
        skipped = bool(payload.get("skipped", rating is None))
        return ShotFeedbackEvent(
            shot_id=str(payload.get("shot_id") or ""),
            install_id=str(payload.get("install_id") or self._config.install_id),
            machine_id=str(payload.get("machine_id") or f"gaggimate:{mac}"),
            timestamp=int(payload.get("timestamp", self._config.now())),
            recommendation_id=payload.get("recommendation_id"),
            rating=None if rating is None else int(rating),
            taste_tags=list(payload.get("taste_tags", [])),
            user_note=payload.get("user_note"),
            skipped=skipped,
            source=payload.get("source", "gaggimate_mqtt"),
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
            grinder_reference_label=payload.get("reference_label", "reference"),
            relative_grind_steps_from_reference=_relative_grind_steps_from_payload(
                payload,
                self._config.initial_relative_grind_steps_from_reference,
            ),
            microns_per_step=_optional_float(
                payload.get("microns_per_step", self._config.microns_per_step)
            ),
            current_absolute_step=_optional_float(payload.get("current_absolute_step")),
            absolute_reference_step=_optional_float(payload.get("absolute_reference_step")),
            dose_in_g=_optional_float(payload.get("dose_in_g", self._config.initial_dose_g)),
            target_yield_g=_optional_float(
                payload.get("target_yield_g", self._config.initial_target_yield_g)
            ),
            source=payload.get("source", "gaggimate_mqtt"),
        )


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("optional numeric fields cannot be boolean")
    return float(value)


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
