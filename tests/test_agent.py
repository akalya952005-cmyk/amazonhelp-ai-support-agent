"""
Unit Test Suite for Hiver Support Agent.
Verifies preprocessing, intent classification, baselines, escalation triggers,
retrieval, generation, and golden dataset integrity.
"""

import os
import pandas as pd

from src.preprocess import clean_text, assign_intent, load_and_preprocess_data, INTENT_LABELS
from src.intent_classifier import (
    IntentClassifier,
    MajorityBaselineClassifier,
    KeywordBaselineClassifier,
    compute_metrics
)
from src.escalation import EscalationDetector
from src.retriever import HistoricalRetriever
from src.generator import ResponseGenerator
from src.golden_dataset_gen import build_golden_dataset
from src.failure_analysis import FailureAnalyzer


def test_clean_text():
    raw = "@AmazonHelp @john_doe My order https://amzn.to/xyz is delayed!!!   &amp; damaged"
    cleaned = clean_text(raw)
    assert "@AmazonHelp" not in cleaned
    assert "@john_doe" not in cleaned
    assert "https://" not in cleaned
    assert "&amp;" not in cleaned
    assert "&" in cleaned
    assert "delayed" in cleaned


def test_intent_mapping():
    assert assign_intent("Where is my delayed package?") == "Order Delay"
    assert assign_intent("I want my refund for the canceled item") == "Refund Request"
    assert assign_intent("Cannot login, locked out of account 2FA OTP") == "Account Access"
    assert assign_intent("Credit card payment failed and double charged") == "Payment Issue"
    assert assign_intent("The blender is defective and broken") == "Product Complaint"
    assert assign_intent("Delivered to wrong address driver missed bell") == "Delivery Problem"
    assert assign_intent("How do I return this item for a replacement pickup") == "Return Request"
    assert assign_intent("What time is Prime Day starting?") == "Other"


def test_escalation_detection_triggers():
    detector = EscalationDetector(confidence_threshold=0.55)

    # 1. Legal complaint
    legal_res = detector.evaluate("My lawyer is preparing a lawsuit in court against Amazon", "Order Delay", 0.90)
    assert legal_res["decision"] == "ESCALATE_TO_HUMAN"
    assert "Legal" in legal_res["reason"]

    # 2. Fraud
    fraud_res = detector.evaluate("There is unauthorized charge and credit card fraud on my account", "Payment Issue", 0.90)
    assert fraud_res["decision"] == "ESCALATE_TO_HUMAN"
    assert "fraud" in fraud_res["reason"].lower()

    # 3. Abuse
    abuse_res = detector.evaluate("You fucking idiots are worthless garbage", "Delivery Problem", 0.90)
    assert abuse_res["decision"] == "ESCALATE_TO_HUMAN"
    assert "abusive" in abuse_res["reason"].lower()

    # 4. Account compromise
    hack_res = detector.evaluate("Help! My account was hacked and primary email changed", "Account Access", 0.90)
    assert hack_res["decision"] == "ESCALATE_TO_HUMAN"
    assert "compromise" in hack_res["reason"].lower() or "takeover" in hack_res["reason"].lower()

    # 5. Repeated failed attempts
    repeat_res = detector.evaluate("I called 4 times and no one is helping me with my issue", "Refund Request", 0.90)
    assert repeat_res["decision"] == "ESCALATE_TO_HUMAN"
    assert "unsuccessful" in repeat_res["reason"].lower() or "multiple" in repeat_res["reason"].lower()

    # 6. Low confidence classification
    low_conf_res = detector.evaluate("Something seems strange today", "Other", 0.42)
    assert low_conf_res["decision"] == "ESCALATE_TO_HUMAN"
    assert "Low classification confidence" in low_conf_res["reason"]

    # 7. Safe auto-handle query
    safe_res = detector.evaluate("When will my order arrive? It says delayed.", "Order Delay", 0.92)
    assert safe_res["decision"] == "AUTO_HANDLE"


def test_intent_classifier_and_baselines():
    df = load_and_preprocess_data()
    assert len(df) >= 100
    assert "clean_text" in df.columns
    assert "intent" in df.columns

    # Train model
    clf = IntentClassifier()
    clf.fit(df["clean_text"], df["intent"])

    # Predict single
    pred_intent, conf = clf.predict_single("My order has been delayed for 3 days")
    assert pred_intent in INTENT_LABELS
    assert 0.0 <= conf <= 1.0

    # Baselines
    majority = MajorityBaselineClassifier()
    majority.fit(df["clean_text"], df["intent"])
    maj_preds = majority.predict(df["clean_text"].iloc[:5])
    assert len(maj_preds) == 5

    keyword = KeywordBaselineClassifier()
    keyword.fit(df["clean_text"], df["intent"])
    kw_preds = keyword.predict(df["clean_text"].iloc[:5])
    assert len(kw_preds) == 5


def test_retriever_and_generator():
    df = load_and_preprocess_data()
    retriever = HistoricalRetriever()
    retriever.build_index(df)

    results = retriever.retrieve_similar("Where is my package? It is late", top_k=5)
    assert len(results) == 5
    assert "customer_message" in results[0]
    assert "agent_reply" in results[0]
    assert "similarity_score" in results[0]

    formatted_context = retriever.format_examples_for_prompt(results)
    assert "Example 1" in formatted_context

    generator = ResponseGenerator()
    reply = generator.generate_reply("Where is my order?", formatted_context, intent="Order Delay")
    assert isinstance(reply, str)
    assert len(reply) > 10


def test_golden_dataset_structure():
    df = load_and_preprocess_data()
    golden_df = build_golden_dataset(df, output_path="golden_dataset/golden_dataset.csv", total_samples=200)

    assert len(golden_df) == 200
    expected_cols = ["tweet", "intent", "human_reply", "escalation"]
    for col in expected_cols:
        assert col in golden_df.columns

    # Verify both AUTO_HANDLE and ESCALATE_TO_HUMAN exist
    escalation_types = golden_df["escalation"].unique()
    assert "AUTO_HANDLE" in escalation_types
    assert "ESCALATE_TO_HUMAN" in escalation_types

    # Verify all 8 intents are represented
    for intent in INTENT_LABELS:
        assert (golden_df["intent"] == intent).sum() > 0


def test_failure_analyzer():
    analyzer = FailureAnalyzer()
    failures = analyzer.analyze_sample(
        tweet="Urgent lawyer lawsuit",
        true_intent="Payment Issue",
        pred_intent="Order Delay",  # Wrong intent
        intent_probs={"Order Delay": 0.50, "Payment Issue": 0.45},
        top_retrievals=[{"similarity_score": 0.20, "intent": "Other"}],  # Poor retrieval
        generated_reply="Everything is fine",
        true_escalation="ESCALATE_TO_HUMAN",
        pred_escalation="AUTO_HANDLE",  # False negative escalation
        escalation_reason="Standard routine"
    )

    failure_types = [f["type"] for f in failures]
    assert "Wrong Intent" in failure_types
    assert "Ambiguous Intent" in failure_types
    assert "Poor Retrieval" in failure_types
    assert "Incorrect Escalation" in failure_types
