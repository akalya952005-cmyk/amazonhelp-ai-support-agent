"""
Data Preprocessing Module for AmazonHelp Customer Support Tweets.
Handles raw Kaggle 'Customer Support on Twitter' (twcs.csv) extraction,
conversation pair joining, tweet cleaning, and intent mapping.
"""

import os
import re
import random
import logging
import pandas as pd
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

INTENT_LABELS = [
    "Order Delay",
    "Refund Request",
    "Account Access",
    "Payment Issue",
    "Product Complaint",
    "Delivery Problem",
    "Return Request",
    "Other",
]

# High precision heuristic keyword patterns for automatic intent annotation
INTENT_KEYWORDS = {
    "Order Delay": [
        r"\bdelayed\b", r"\blate\b", r"\bstill waiting\b", r"\bhasn'?t arrived\b",
        r"\bwhere is my order\b", r"\btracking not updating\b", r"\bexpected yesterday\b",
        r"\bpackage delayed\b", r"\bnot delivered yet\b", r"\border delay\b",
        r"\btaking too long\b", r"\btransit\b", r"\bestimated delivery\b"
    ],
    "Refund Request": [
        r"\brefund\b", r"\bmoney back\b", r"\bcharged me twice\b", r"\breimburse\b",
        r"\bcredit back\b", r"\bwhere is my refund\b", r"\bcancel and refund\b",
        r"\brefund status\b", r"\bdeducted\b", r"\bdeduction\b"
    ],
    "Account Access": [
        r"\blogin\b", r"\bpassword\b", r"\blocked out\b", r"\botp\b", r"\b2fa\b",
        r"\breset password\b", r"\bunauthorized access\b", r"\bhacked\b",
        r"\bcannot sign in\b", r"\baccount suspended\b", r"\bverification code\b",
        r"\bemail changed\b", r"\baccount access\b"
    ],
    "Payment Issue": [
        r"\bcredit card\b", r"\bdebit card\b", r"\bpayment failed\b", r"\bdeclined\b",
        r"\bdropped payment\b", r"\bdouble charged\b", r"\bbilling\b", r"\bupi\b",
        r"\bnetbanking\b", r"\bgift card not working\b", r"\bpromo code\b",
        r"\bwallet\b", r"\bpayment issue\b", r"\btransaction failed\b"
    ],
    "Product Complaint": [
        r"\bdefective\b", r"\bbroken\b", r"\bdamaged\b", r"\bcounterfeit\b",
        r"\bfake\b", r"\bpoor quality\b", r"\bexpired\b", r"\bused item\b",
        r"\bmissing parts\b", r"\bnot working\b", r"\bmalfunction\b",
        r"\bwrong specifications\b", r"\bterrible quality\b"
    ],
    "Delivery Problem": [
        r"\bdelivered to wrong address\b", r"\bmarked as delivered but not received\b",
        r"\bdelivery driver\b", r"\bcourier\b", r"\bwrong house\b", r"\bpackage lost\b",
        r"\bstolen package\b", r"\bmissed delivery\b", r"\bpackage missing\b",
        r"\bnot received\b", r"\bthrew package\b", r"\bporch pirate\b"
    ],
    "Return Request": [
        r"\breturn\b", r"\breplacement\b", r"\bexchange\b", r"\bpickup\b",
        r"\breturn label\b", r"\bdrop off\b", r"\bcourier pickup\b",
        r"\bhow to return\b", r"\breturn policy\b", r"\bsend back\b"
    ],
}


def clean_text(text: str) -> str:
    """Cleans tweet text by stripping handles, normalising whitespace, and stripping URLs."""
    if not isinstance(text, str):
        return ""
    # Strip user mentions e.g. @AmazonHelp @customer
    text = re.sub(r"@\w+", "", text)
    # Replace URLs with standard marker or remove
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove emojis / stray control chars
    text = re.sub(r"[\r\n\t]+", " ", text)
    # Remove HTML entities
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    # Strip excessive punctuation and whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def assign_intent(text: str) -> str:
    """Classifies a customer text into one of the 8 predefined intents based on pattern rules."""
    text_lower = text.lower()
    for intent, patterns in INTENT_KEYWORDS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return intent
    return "Other"


def generate_bootstrap_dataset(num_samples: int = 1200) -> pd.DataFrame:
    """
    Generates a realistic synthetic AmazonHelp conversation dataset adhering exactly
    to the Kaggle schema and style, covering all 8 intents evenly.
    Used when raw twcs.csv has not been placed in the data/ directory.
    """
    logger.info("Generating bootstrap dataset for AmazonHelp conversations...")
    templates = {
        "Order Delay": [
            ("My order #{} has been delayed for 4 days now. Still waiting for any update! What is going on?",
             "We apologize for the delay with your order! Please DM us your order ID and postal code at amzn.to/help so we can look into the status for you. ^AB"),
            ("Tracking says delayed in transit since Monday. When can I expect delivery?",
             "Sorry to hear your package is delayed in transit. Please send us a direct message with your tracking number so our logistics team can investigate. ^RK"),
            ("Amazon Prime delivery was promised by 8 PM yesterday, still hasn't arrived. Very disappointed.",
             "We understand how frustrating late deliveries are, especially with Prime. DM us your account email and order details so we can make this right. ^SK"),
            ("Where is my order? The tracking has not updated for three days straight.",
             "Thanks for reaching out. Tracking can occasionally pause between transit hubs. Please DM your order number and we'll check directly with the carrier. ^JD")
        ],
        "Refund Request": [
            ("I returned my shoes 2 weeks ago and still haven't received my refund. Where is my money?",
             "Refunds typically take 3-5 business days after the return is processed at our facility. Please DM us your return tracking details so we can expedite this. ^AB"),
            ("You charged me twice for my Prime membership renew! Please issue an immediate refund.",
             "We're sorry for the billing discrepancy! Please send us a DM with the charge date and the last 4 digits of the card used so we can reverse the duplicate charge. ^MS"),
            ("I canceled the order before it dispatched. When will the refund be credited back to my bank?",
             "Once an order is canceled, the pre-authorization hold is released within 3-7 business days depending on your bank. Send a DM if you need a transaction receipt. ^LK"),
            ("Can you please refund the money for the missing item in my package?",
             "We apologize for the missing item! Please DM us your order ID and confirm which item was missing so we can process a full refund or replacement immediately. ^TR")
        ],
        "Account Access": [
            ("I am locked out of my Amazon account because the 2FA OTP is sending to my old phone number.",
             "Account security is our top priority. Please visit amzn.to/account-recovery or send us a DM so our account verification team can assist you securely. ^PK"),
            ("Someone hacked my account and changed the primary email address. Help urgently!",
             "We treat security alerts with immediate urgency. Please contact our specialized fraud team right away via amzn.to/security or DM us your registered phone number. ^AB"),
            ("Cannot sign in to my Amazon Prime Video app, it keeps saying invalid password even after reset.",
             "Sorry for the login hassle! Please ensure your app is updated to the latest version. If it persists, DM us the device details and your account email. ^NR"),
            ("My account got suspended without any explanation. How do I appeal this decision?",
             "We understand your concern. Account suspensions are reviewed by our specialized Account Review team. Please check your inbox for an email from our team or DM us. ^KJ")
        ],
        "Payment Issue": [
            ("My debit card payment failed during checkout but the money was deducted from my account.",
             "When a transaction fails, debited funds are held by the bank and automatically reversed within 5-7 business days. DM us if you need payment failure logs. ^AB"),
            ("My Amazon Pay balance is not showing up when I try to pay for my grocery order.",
             "Sorry for the glitch! Try signing out and signing back in. If your balance still isn't available at checkout, please send us a DM with your account email. ^DV"),
            ("I tried to redeem a $50 gift card and it says code already redeemed by someone else.",
             "We'd like to check the redemption history of this gift card. Please DM us the 16-character claim code and proof of purchase so we can investigate. ^TC"),
            ("Credit card is declined only on Amazon website, works fine elsewhere. Please fix this.",
             "This may be caused by an expired billing address on file. Please verify your billing details at amzn.to/payments or DM us for further assistance. ^KL")
        ],
        "Product Complaint": [
            ("The laptop I received today has a cracked screen and doesn't power on. Terrible packaging!",
             "We are very sorry to hear your laptop arrived damaged! Please DM us your order number so we can arrange an immediate pickup and expedited replacement. ^AB"),
            ("The perfume I bought smells fake and the seal was clearly broken. How do you allow counterfeit items?",
             "Amazon strictly prohibits counterfeit items. We take this very seriously. Please DM us your order details so our investigation team can review the seller. ^RK"),
            ("Received the wrong color and size for the jacket I ordered. I ordered Medium Black not XL Yellow.",
             "Apologies for the mix-up! You can easily set up a free return and replacement via Your Orders or DM us and we'll process it for you right away. ^MP"),
            ("The blender stopped working after just 2 uses. Motor started smoking. What is the warranty process?",
             "Safety is our utmost priority. Please discontinue using the blender immediately. DM us your order ID so we can coordinate a return and manufacturer warranty support. ^HN")
        ],
        "Delivery Problem": [
            ("App says delivered to front porch at 2 PM, but I was home and nothing was delivered. Package missing!",
             "Sometimes carriers mark items delivered prematurely. Please check around your property or with neighbors. If it doesn't show up in 24 hours, DM us. ^AB"),
            ("Delivery driver threw my fragile package over the 6-foot fence! Contents are shattered.",
             "We hold our delivery partners to high service standards. Please DM us your tracking number and photos of the package so we can escalate this to the dispatch manager. ^ST"),
            ("Package delivered to wrong address down the street. Driver didn't even check the house number.",
             "We apologize for the delivery error. Please DM us your order number and full delivery address so we can retrieve or replace your package immediately. ^FG"),
            ("Courier marked customer unavailable, but I waited home all day. Nobody even rang the bell.",
             "We're sorry for the inconvenience! Please DM us your tracking details and best contact number so we can instruct the local delivery hub to re-attempt today. ^MN")
        ],
        "Return Request": [
            ("How do I return an item that doesn't fit? Where do I get the return shipping label?",
             "You can initiate a return by visiting Your Orders -> Return or Replace Items to print a prepaid label or generate a QR drop-off code. DM us if you need help! ^AB"),
            ("The courier didn't come to pick up the return package today as scheduled. Please reschedule.",
             "Sorry for the missed return pickup! Please DM us your pickup tracking ID and preferred date/time slot so we can rebook the courier for you. ^RT"),
            ("Can I drop off my Amazon return at Kohl's or Whole Foods without a box or label?",
             "Yes! Most returns can be dropped off packaging-free at participating Kohl's or Whole Foods with your return QR code. DM us if you'd like us to confirm eligibility. ^JH"),
            ("The return window closed yesterday by 1 day. Is there any way I can still return this unused item?",
             "We can check if an exception can be made for you. Please DM us your order number and reason for return so an agent can review your account. ^WE")
        ],
        "Other": [
            ("Are you having a sale on Echo Dot devices next week for Prime Day?",
             "We don't announce promotional deals in advance, but keep an eye on our Today's Deals page at amzn.to/deals for the latest announcements! ^AB"),
            ("Can I change the delivery address on an order that hasn't shipped yet?",
             "If the order hasn't entered the shipping process, you can update the address in Your Orders. If it has shipped, DM us your order ID to see if rerouting is possible. ^GB"),
            ("Does Amazon Prime membership include Kindle Unlimited books?",
             "Prime includes Prime Reading (a rotating catalog of books and magazines), while Kindle Unlimited is a separate subscription with a wider catalog. DM if you have questions! ^OP"),
            ("I need an invoice copy for my business expense reimbursement from last month.",
             "You can download official VAT/GST invoices directly from Your Orders -> Order Details -> Invoice. DM us your order number if you encounter any issue! ^KL")
        ]
    }

    rows = []
    rng = random.Random(42)
    sample_id = 100000

    while len(rows) < num_samples:
        for intent in INTENT_LABELS:
            pair = rng.choice(templates[intent])
            cust_text = pair[0]
            if "{}" in cust_text:
                order_id = f"{rng.randint(100, 999)}-{rng.randint(1000000, 9999999)}"
                cust_text = cust_text.format(order_id)
            agent_text = pair[1]

            # Add subtle variations
            rows.append({
                "tweet_id": sample_id,
                "author_id": f"customer_{sample_id}",
                "in_response_to_tweet_id": None,
                "created_at": "2023-10-15 12:00:00",
                "text": cust_text,
                "response_tweet_id": sample_id + 1,
                "inbound": True,
                "human_reply": agent_text,
                "intent": intent,
                "clean_text": clean_text(cust_text),
                "clean_reply": clean_text(agent_text),
            })
            sample_id += 2
            if len(rows) >= num_samples:
                break

    df = pd.DataFrame(rows)
    logger.info(f"Generated {len(df)} bootstrap sample conversations across {len(INTENT_LABELS)} intents.")
    return df


def load_and_preprocess_data(data_dir: str = "data", max_rows: Optional[int] = 50000) -> pd.DataFrame:
    """
    Loads Customer Support on Twitter data.
    If twcs.csv is present, extracts AmazonHelp conversations.
    If twcs.csv is absent, generates a high-quality bootstrap dataset.
    Returns clean DataFrame and saves to data/amazon_help_processed.csv.
    """
    os.makedirs(data_dir, exist_ok=True)
    processed_path = os.path.join(data_dir, "amazon_help_processed.csv")

    raw_csv_path = None
    for candidate in [os.path.join(data_dir, "twcs.csv"), "twcs.csv"]:
        if os.path.exists(candidate):
            raw_csv_path = candidate
            break

    if raw_csv_path:
        logger.info(f"Found raw Kaggle dataset at {raw_csv_path}. Processing AmazonHelp conversations...")
        try:
            # Read in chunks to manage memory efficiency on large Kaggle CSV (~700MB)
            chunk_size = 50000
            amazon_replies = []
            inbound_tweets = []

            for chunk in pd.read_csv(raw_csv_path, chunksize=chunk_size, low_memory=False):
                # AmazonHelp agent replies
                az_chunk = chunk[chunk["author_id"] == "AmazonHelp"].dropna(subset=["text", "in_response_to_tweet_id"])
                if not az_chunk.empty:
                    amazon_replies.append(az_chunk)

                # Inbound customer tweets
                inbound_chunk = chunk[chunk["inbound"] == True].dropna(subset=["text", "tweet_id"])
                if not inbound_chunk.empty:
                    inbound_tweets.append(inbound_chunk)

                if max_rows is not None and sum(len(c) for c in amazon_replies) >= max_rows:
                    break

            df_replies = pd.concat(amazon_replies, ignore_index=True)
            df_inbound = pd.concat(inbound_tweets, ignore_index=True)

            # Ensure numeric tweet_id types for reliable join
            df_replies["in_response_to_tweet_id"] = pd.to_numeric(df_replies["in_response_to_tweet_id"], errors="coerce")
            df_inbound["tweet_id"] = pd.to_numeric(df_inbound["tweet_id"], errors="coerce")

            # Join inbound customer tweet with AmazonHelp reply
            merged = pd.merge(
                df_inbound,
                df_replies[["tweet_id", "in_response_to_tweet_id", "text"]],
                left_on="tweet_id",
                right_on="in_response_to_tweet_id",
                suffixes=("_customer", "_agent")
            )

            # Drop missing values
            merged = merged.dropna(subset=["text_customer", "text_agent"])

            # Clean texts
            merged["clean_text"] = merged["text_customer"].apply(clean_text)
            merged["clean_reply"] = merged["text_agent"].apply(clean_text)
            merged = merged[(merged["clean_text"].str.len() > 10) & (merged["clean_reply"].str.len() > 10)]

            # Assign intents
            merged["intent"] = merged["clean_text"].apply(assign_intent)
            merged["human_reply"] = merged["text_agent"]
            merged["text"] = merged["text_customer"]

            final_df = merged[["tweet_id_customer", "clean_text", "text", "human_reply", "clean_reply", "intent"]].rename(
                columns={"tweet_id_customer": "tweet_id"}
            )
            final_df.to_csv(processed_path, index=False)
            logger.info(f"Processed {len(final_df)} AmazonHelp conversations from twcs.csv and saved to {processed_path}")
            return final_df
        except Exception as e:
            logger.error(f"Error processing {raw_csv_path}: {e}. Falling back to bootstrap dataset.", exc_info=True)

    # If processed dataset already exists and is non-empty, load it
    if os.path.exists(processed_path):
        existing_df = pd.read_csv(processed_path)
        if len(existing_df) >= 100:
            logger.info(f"Loaded existing processed dataset from {processed_path} ({len(existing_df)} rows).")
            return existing_df

    # Otherwise generate bootstrap dataset
    bootstrap_df = generate_bootstrap_dataset(num_samples=1600)
    bootstrap_df.to_csv(processed_path, index=False)
    logger.info(f"Saved processed dataset to {processed_path}")
    return bootstrap_df


if __name__ == "__main__":
    df = load_and_preprocess_data()
    print("Dataset Summary:")
    print(f"Total Rows: {len(df)}")
    print("Intent Distribution:")
    print(df["intent"].value_counts())
