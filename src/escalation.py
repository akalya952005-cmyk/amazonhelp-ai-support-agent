"""
Escalation Detection Module.
Evaluates customer messages against safety, risk, and model confidence criteria.
Decides whether to AUTO_HANDLE or ESCALATE_TO_HUMAN with explicit reasoning.
"""

import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Trigger keywords and regexes for critical escalation categories
ESCALATION_RULES = {
    "legal_complaint": [
        r"\blawyer\b", r"\battorney\b", r"\bsue\b", r"\blawsuit\b", r"\blegal action\b",
        r"\bcourt\b", r"\bconsumer court\b", r"\bregulatory\b", r"\blitigation\b",
        r"\bsubpoena\b", r"\bpolice complaint\b", r"\bftc\b", r"\bbetter business bureau\b", r"\bbbb\b"
    ],
    "fraud": [
        r"\bfraud\b", r"\bunauthorized charge\b", r"\bstolen card\b", r"\bidentity theft\b",
        r"\bscam\b", r"\bscammed\b", r"\bcredit card fraud\b", r"\bunauthorized transaction\b",
        r"\bcompromised card\b", r"\bmoney stolen\b"
    ],
    "abuse": [
        r"\bfuck\b", r"\bshit\b", r"\bbastard\b", r"\basshole\b", r"\bbitch\b",
        r"\bidiot\b", r"\bkill\b", r"\bthreat\b", r"\bharass\b", r"\bscumbag\b",
        r"\bworthless garbage\b", r"\bhorrible humans\b"
    ],
    "account_compromise": [
        r"\bhacked\b", r"\baccount taken over\b", r"\bunauthorized login\b",
        r"\bemail changed without my permission\b", r"\baccount compromise\b",
        r"\bsomeone changed my password\b", r"\bstrange login from\b"
    ],
    "repeated_failed_attempts": [
        r"\b(tried|contacted|called|reached out|asked)\s+(3|4|5|several|multiple)\s+times\b",
        r"\b(3rd|4th|5th)\s+time\b",
        r"\bstill no response after\b",
        r"\bno one is helping\b",
        r"\balready complained\b",
        r"\bgetting nowhere\b",
        r"\btalked to (3|4|multiple) agents\b"
    ]
}


class EscalationDetector:
    """Production Escalation Detector evaluating rule-based triggers and model confidence."""
    def __init__(self, confidence_threshold: float = 0.55):
        self.confidence_threshold = confidence_threshold

    def evaluate(
        self,
        query: str,
        predicted_intent: Optional[str] = None,
        confidence: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a query and returns structured decision:
        {
          "decision": "AUTO_HANDLE" or "ESCALATE_TO_HUMAN",
          "reason": "..."
        }
        """
        text_lower = query.lower()

        # 1. Check Legal Complaint
        for pattern in ESCALATION_RULES["legal_complaint"]:
            if re.search(pattern, text_lower):
                return {
                    "decision": "ESCALATE_TO_HUMAN",
                    "reason": f"Legal risk detected: query matches legal complaint criteria ('{pattern.strip(r'\\b')}').",
                    "trigger_type": "legal_complaint",
                    "confidence": confidence
                }

        # 2. Check Fraud
        for pattern in ESCALATION_RULES["fraud"]:
            if re.search(pattern, text_lower):
                return {
                    "decision": "ESCALATE_TO_HUMAN",
                    "reason": f"Financial security risk: potential fraud or unauthorized activity detected ('{pattern.strip(r'\\b')}').",
                    "trigger_type": "fraud",
                    "confidence": confidence
                }

        # 3. Check Abuse / Harassment
        for pattern in ESCALATION_RULES["abuse"]:
            if re.search(pattern, text_lower):
                return {
                    "decision": "ESCALATE_TO_HUMAN",
                    "reason": "Customer communication contains abusive, profane, or threatening language requiring human specialist de-escalation.",
                    "trigger_type": "abuse",
                    "confidence": confidence
                }

        # 4. Check Account Compromise
        for pattern in ESCALATION_RULES["account_compromise"]:
            if re.search(pattern, text_lower):
                return {
                    "decision": "ESCALATE_TO_HUMAN",
                    "reason": "Security incident: account takeover or compromised credential indicators present.",
                    "trigger_type": "account_compromise",
                    "confidence": confidence
                }

        # 5. Check Repeated Failed Attempts
        for pattern in ESCALATION_RULES["repeated_failed_attempts"]:
            if re.search(pattern, text_lower):
                return {
                    "decision": "ESCALATE_TO_HUMAN",
                    "reason": "Customer expressed multiple unsuccessful prior contact attempts; requires priority human intervention.",
                    "trigger_type": "repeated_failed_attempts",
                    "confidence": confidence
                }

        # 6. Check Low Confidence Intent Classification
        if confidence is not None and confidence < self.confidence_threshold:
            return {
                "decision": "ESCALATE_TO_HUMAN",
                "reason": (
                    f"Low classification confidence ({confidence:.2f} < threshold {self.confidence_threshold:.2f}). "
                    f"Ambiguous intent '{predicted_intent}' safely escalated to human tier."
                ),
                "trigger_type": "low_confidence",
                "confidence": confidence
            }

        # Safe for automated resolution
        return {
            "decision": "AUTO_HANDLE",
            "reason": f"Standard support inquiry within operational bounds for intent '{predicted_intent}' (confidence: {confidence if confidence else 1.0:.2f}).",
            "trigger_type": "none",
            "confidence": confidence
        }
