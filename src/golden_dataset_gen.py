"""
Golden Dataset Generation Module.
Creates a balanced 200-sample high-quality review set covering all 8 intents
and critical escalation boundary cases (legal, fraud, abuse, account compromise, repeated failures).
Exports to golden_dataset/golden_dataset.csv.
"""

import os
import random
import logging
import pandas as pd
from typing import Optional
from src.preprocess import INTENT_LABELS
from src.escalation import EscalationDetector

logger = logging.getLogger(__name__)


# Specific targeted edge cases for human review calibration
CRITICAL_ESCALATION_CASES = [
    # Legal Complaints
    ("My lawyer advised me to file a formal lawsuit in consumer court against Amazon for gross negligence on my undelivered Mac.",
     "Order Delay",
     "We understand your concern regarding order delivery. Please DM us your order ID so we can investigate immediately. ^AB",
     "ESCALATE_TO_HUMAN"),
    ("I am submitting a complaint to the FTC and taking legal action regarding recurring charges after subscription cancellation.",
     "Payment Issue",
     "Please DM us the account email and last 4 digits of the card charged so our billing specialist can assist. ^RK",
     "ESCALATE_TO_HUMAN"),
    ("If my damaged goods are not compensated within 24 hours my attorney is serving a subpoena.",
     "Product Complaint",
     "We apologize for the damaged items. Please DM us your order number and photos so we can resolve this urgently. ^ST",
     "ESCALATE_TO_HUMAN"),
    # Fraud & Unauthorized Transactions
    ("URGENT: Someone stole my credit card and made $3000 unauthorized purchases on Amazon right now!",
     "Payment Issue",
     "Please visit amzn.to/security right away to report unauthorized charges or DM us your registered phone number. ^PK",
     "ESCALATE_TO_HUMAN"),
    ("I fell victim to a scam where a third party seller took my money and closed their storefront. Fraud!",
     "Payment Issue",
     "We take fraud allegations very seriously. Please DM us the seller name and order ID so our Trust & Safety team can review. ^AB",
     "ESCALATE_TO_HUMAN"),
    ("There is a strange charge on my bank statement from Amazon EU that I never authorized.",
     "Payment Issue",
     "We'd be glad to look into this charge. Please DM us the charge amount, date, and your billing postal code. ^MN",
     "ESCALATE_TO_HUMAN"),
    # Account Compromise
    ("My account has been hacked! Primary email was changed without my permission and I am locked out.",
     "Account Access",
     "Please contact our Account Security specialists immediately at amzn.to/security or DM us your registered contact number. ^AB",
     "ESCALATE_TO_HUMAN"),
    ("I am receiving two-factor authentication codes every 2 minutes. Someone is trying to break into my account.",
     "Account Access",
     "Security is our priority. Please change your password immediately and review active sessions at amzn.to/auth. ^SK",
     "ESCALATE_TO_HUMAN"),
    ("My seller account credentials were compromised and bank account details modified!",
     "Account Access",
     "Please reach out directly via Seller Central emergency escalation or DM us your merchant token so we can freeze updates. ^JD",
     "ESCALATE_TO_HUMAN"),
    # Abuse & Harassment
    ("You worthless pieces of shit ruined my daughter's birthday! I hope your whole company burns to the ground.",
     "Delivery Problem",
     "We are deeply sorry for the disappointment caused. Please DM us your order number so a senior lead can review this. ^AB",
     "ESCALATE_TO_HUMAN"),
    ("Fucking idiots! Your courier threw the fragile package over the gate and broke everything.",
     "Product Complaint",
     "We sincerely apologize for this courier behavior. Please DM us your tracking number and order ID so we can address this. ^RK",
     "ESCALATE_TO_HUMAN"),
    # Repeated Failed Attempts
    ("I have called your customer service 4 times and spoken to 3 different agents. No one is helping me with my lost package!",
     "Delivery Problem",
     "We sincerely apologize for your experience after multiple contacts. Please DM us your order ID and phone number for immediate supervisor review. ^AB",
     "ESCALATE_TO_HUMAN"),
    ("This is my 5th time reaching out about this refund. Every agent gives a different answer and closes the ticket!",
     "Refund Request",
     "We apologize for the frustration caused by contradictory updates. Please DM your return tracking details so we can escalate to a supervisor. ^SK",
     "ESCALATE_TO_HUMAN"),
    ("Tried 3 times to get a replacement for my broken monitor and courier keeps missing the pickup window.",
     "Return Request",
     "We understand how frustrating missed pickups are after repeated attempts. Please DM us your pickup ID so we can reschedule with courier leadership. ^JD",
     "ESCALATE_TO_HUMAN"),
    # Ambiguous / Low Confidence queries requiring human review
    ("Can somebody just help me? Nothing is working and I am lost.",
     "Other",
     "We're here to help! Could you please let us know what specific issue or order you're dealing with, or DM us? ^AB",
     "ESCALATE_TO_HUMAN"),
    ("The thing happened again like last Tuesday.",
     "Other",
     "We'd love to help out! Could you please DM us more context or your order ID so we can look up your history? ^AB",
     "ESCALATE_TO_HUMAN"),
]


def build_golden_dataset(
    processed_df: pd.DataFrame,
    output_path: str = "golden_dataset/golden_dataset.csv",
    total_samples: int = 200,
    seed: int = 42
) -> pd.DataFrame:
    """
    Constructs a calibrated 200-sample golden dataset.
    Ensures representation of all 8 intents and realistic escalation distribution.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    detector = EscalationDetector(confidence_threshold=0.55)
    random.seed(seed)

    golden_rows = []

    # 1. Add curated high-impact escalation test cases (16 cases)
    for tweet, intent, reply, esc in CRITICAL_ESCALATION_CASES:
        golden_rows.append({
            "tweet": tweet,
            "intent": intent,
            "human_reply": reply,
            "escalation": esc
        })

    remaining_slots = total_samples - len(golden_rows)
    samples_per_intent = remaining_slots // len(INTENT_LABELS)

    # 2. Sample proportionally from processed corpus for each intent
    for intent in INTENT_LABELS:
        intent_df = processed_df[processed_df["intent"] == intent]
        if intent_df.empty:
            continue

        n_sample = min(samples_per_intent, len(intent_df))
        sampled = intent_df.sample(n=n_sample, random_state=seed)

        for _, row in sampled.iterrows():
            text = str(row.get("clean_text", row.get("text", "")))
            reply = str(row.get("human_reply", row.get("clean_reply", "")))
            
            # Determine rule-based escalation
            esc_res = detector.evaluate(text, predicted_intent=intent, confidence=0.85)
            golden_rows.append({
                "tweet": text,
                "intent": intent,
                "human_reply": reply,
                "escalation": esc_res["decision"]
            })

    # Fill any remaining slots to reach exactly 200
    while len(golden_rows) < total_samples:
        random_row = processed_df.sample(n=1, random_state=seed + len(golden_rows)).iloc[0]
        text = str(random_row.get("clean_text", random_row.get("text", "")))
        intent = str(random_row.get("intent", "Other"))
        reply = str(random_row.get("human_reply", random_row.get("clean_reply", "")))
        esc_res = detector.evaluate(text, predicted_intent=intent, confidence=0.85)

        golden_rows.append({
            "tweet": text,
            "intent": intent,
            "human_reply": reply,
            "escalation": esc_res["decision"]
        })

    golden_df = pd.DataFrame(golden_rows).iloc[:total_samples]
    golden_df.to_csv(output_path, index=False)
    logger.info(f"Generated Golden Dataset with {len(golden_df)} samples at {output_path}")
    return golden_df


if __name__ == "__main__":
    from src.preprocess import load_and_preprocess_data
    df = load_and_preprocess_data()
    golden_df = build_golden_dataset(df)
    print("Golden Dataset Preview:")
    print(golden_df.head())
    print("\nIntent Distribution:")
    print(golden_df["intent"].value_counts())
    print("\nEscalation Distribution:")
    print(golden_df["escalation"].value_counts())
