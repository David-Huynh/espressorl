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
    ShotFeedbackEvent,
    ShotProfileEvent,
)
from espresso_rl.domain.models import Recommendation
from espresso_rl.domain.models import new_id

logger = logging.getLogger(__name__)

SHOT_TOPIC = "gaggimate/+/shot/profile"
FEEDBACK_TOPIC = "gaggimate/+/rl/rating"
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
        on_decision: Callable[[RecommendationDecisionEvent], None],
        on_apply: Callable[[RecommendationApplyEvent], None],
        on_machine_state: Callable[[MachineStateEvent], None],
    ) -> None:
        self._config = config
        self._on_shot = on_shot
        self._on_feedback = on_feedback
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
            "grind_delta_steps": recommendation.grind_delta_steps,
            "grind_delta_um": recommendation.grind_delta_um,
            "next_grind_steps": recommendation.next_grind_steps,
            "next_grind_um": recommendation.next_grind_um,
            "next_dose_g": recommendation.next_dose_g,
            "target_yield_g": recommendation.target_yield_g,
            "target_ratio": recommendation.target_ratio,
            "mode": recommendation.mode.value,
            "confidence": recommendation.confidence,
            "reason": recommendation.reason,
            "status": recommendation.status.value,
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
            client.subscribe(DECISION_TOPIC)
            client.subscribe(APPLY_TOPIC)
            client.subscribe(MACHINE_STATE_TOPIC)
            logger.info(
                "Subscribed to %s, %s, %s, %s, %s",
                SHOT_TOPIC,
                FEEDBACK_TOPIC,
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
            grinder_step_size_um=float(payload.get("grinder_step_size_um", self._config.grinder_step_size_um)),
            grind_steps=_optional_float(payload.get("grind_steps", self._config.initial_grind_steps)),
            dose_in_g=float(payload.get("dose_in_g", self._config.initial_dose_g)),
            target_yield_g=target_yield_g,
            beverage_out_g=_optional_float(beverage_out_g),
            shot_time_s=_optional_float(shot_time_s),
            bean_context_id=payload.get("bean_context_id", self._config.bean_context_id),
            recommendation_id=payload.get("recommendation_id"),
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
            grind_steps=_optional_float(payload.get("grind_steps", self._config.initial_grind_steps)),
            grinder_step_size_um=_optional_float(
                payload.get("grinder_step_size_um", self._config.grinder_step_size_um)
            ),
            dose_in_g=_optional_float(payload.get("dose_in_g", self._config.initial_dose_g)),
            target_yield_g=_optional_float(
                payload.get("target_yield_g", self._config.initial_target_yield_g)
            ),
            source=payload.get("source", "gaggimate_mqtt"),
        )


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _channel(payload: dict[str, Any], key: str, n: int) -> list[float]:
    values = payload.get(key)
    if values is None:
        return [0.0] * n
    result = list(values)
    if len(result) != n:
        return [0.0] * n
    return result
