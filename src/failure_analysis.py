"""
Failure Analysis Module.
Automatically inspects model predictions, retrieval results, generated replies,
and escalation decisions to identify and categorize failure modes:
1. Wrong intent
2. Ambiguous intent
3. Poor retrieval
4. Hallucinated reply
5. Incorrect escalation
"""

import os
import re
import json
import logging
import pandas as pd
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class FailureAnalyzer:
    """Detects and categorizes system failures across the customer support pipeline."""
    def __init__(self, ambiguity_margin: float = 0.15, retrieval_sim_threshold: float = 0.45):
        self.ambiguity_margin = ambiguity_margin
        self.retrieval_sim_threshold = retrieval_sim_threshold

    def analyze_sample(
        self,
        tweet: str,
        true_intent: str,
        pred_intent: str,
        intent_probs: Optional[Dict[str, float]],
        top_retrievals: List[Dict[str, Any]],
        generated_reply: str,
        true_escalation: str,
        pred_escalation: str,
        escalation_reason: str,
        judge_scores: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Identifies any failure modes present in a processed customer support transaction."""
        failures = []

        # 1. Wrong Intent
        if true_intent != pred_intent:
            failures.append({
                "type": "Wrong Intent",
                "severity": "HIGH",
                "details": f"Ground truth intent was '{true_intent}', but model predicted '{pred_intent}'."
            })

        # 2. Ambiguous Intent
        if intent_probs:
            sorted_probs = sorted(intent_probs.values(), reverse=True)
            if len(sorted_probs) >= 2:
                margin = sorted_probs[0] - sorted_probs[1]
                if margin < self.ambiguity_margin or sorted_probs[0] < 0.45:
                    failures.append({
                        "type": "Ambiguous Intent",
                        "severity": "MEDIUM",
                        "details": f"Top probability ({sorted_probs[0]:.2f}) close to runner-up ({sorted_probs[1]:.2f}), margin: {margin:.2f}."
                    })

        # 3. Poor Retrieval
        if not top_retrievals:
            failures.append({
                "type": "Poor Retrieval",
                "severity": "HIGH",
                "details": "No historical context was retrieved for the query."
            })
        else:
            top_sim = top_retrievals[0].get("similarity_score", 1.0)
            top_intent = top_retrievals[0].get("intent", "")
            if top_sim < self.retrieval_sim_threshold:
                failures.append({
                    "type": "Poor Retrieval",
                    "severity": "MEDIUM",
                    "details": f"Top retrieved similarity score ({top_sim:.2f}) fell below threshold ({self.retrieval_sim_threshold:.2f})."
                })
            elif top_intent != true_intent and top_sim < 0.60:
                failures.append({
                    "type": "Poor Retrieval",
                    "severity": "LOW",
                    "details": f"Retrieved top example had mismatched intent ('{top_intent}' vs '{true_intent}')."
                })

        # 4. Hallucinated Reply
        hallucination_triggers = [
            r"100% money back guaranteed within 10 minutes",
            r"free iphone",
            r"http://[a-zA-Z0-9.-]+(?<!amzn\.to)\b",  # Non-Amazon link
            r"send us your credit card number and cvv",
            r"call 1-800-[0-9]{3}-[0-9]{4}"
        ]
        for trig in hallucination_triggers:
            if re.search(trig, generated_reply, re.IGNORECASE):
                failures.append({
                    "type": "Hallucinated Reply",
                    "severity": "CRITICAL",
                    "details": f"Reply generated suspicious, out-of-policy, or hallucinated claims matching '{trig}'."
                })

        if judge_scores and judge_scores.get("correctness", 5) <= 2:
            failures.append({
                "type": "Hallucinated Reply",
                "severity": "HIGH",
                "details": f"Judge gave low correctness score ({judge_scores.get('correctness')}): {judge_scores.get('reasoning', '')}"
            })

        # 5. Incorrect Escalation
        if true_escalation != pred_escalation:
            if pred_escalation == "AUTO_HANDLE" and true_escalation == "ESCALATE_TO_HUMAN":
                failures.append({
                    "type": "Incorrect Escalation",
                    "severity": "CRITICAL",
                    "details": f"False Negative Escalation: high-risk inquiry was auto-handled! Reason given: {escalation_reason}"
                })
            else:
                failures.append({
                    "type": "Incorrect Escalation",
                    "severity": "LOW",
                    "details": f"False Positive Escalation: safe routine inquiry was unnecessarily escalated. Reason: {escalation_reason}"
                })

        return failures

    def save_failure_report(self, failure_records: List[Dict[str, Any]], output_path: str = "reports/failure_analysis.json"):
        """Exports detected failure cases to a formatted JSON report."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(failure_records, f, indent=2)
        logger.info(f"Saved {len(failure_records)} failure case records to {output_path}")
