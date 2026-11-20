import json
import re
import requests
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "aviation-slm-v3:latest"

# Dataset path can be configured through an environment variable.
# Default location: data/stage20_32_search_index.json

INDEX_PATH = Path(
    os.environ.get(
        "AVIATION_INDEX_PATH",
        Path(__file__).resolve().parent / "data" / "stage20_32_search_index.json"
    )
)

# --------------------------------------------------
# LOAD AVIATION DATASET
# --------------------------------------------------

print("Loading aviation records...")

if not INDEX_PATH.exists():
    raise FileNotFoundError(
        f"Aviation dataset index not found: {INDEX_PATH}\n"
        "Set the AVIATION_INDEX_PATH environment variable "
        "to the correct dataset index location."
    )

with open(INDEX_PATH, "r", encoding="utf-8") as f:
    aviation_index = json.load(f)

date_lookup = defaultdict(list)

date_pattern = re.compile(
    r'\b(?:Event\s+Date|Date)\s*:\s*(\d{2}/\d{2}/\d{4})',
    re.IGNORECASE
)

for item in aviation_index:
    record = item.get("record", "")

    for date in set(date_pattern.findall(record)):
        try:
            datetime.strptime(date, "%d/%m/%Y")
            date_lookup[date].append(record)
        except ValueError:
            continue

print(f"Loaded {len(aviation_index)} records.")
print(f"Found {len(date_lookup)} unique incident dates.")

# --------------------------------------------------
# FIELD EXTRACTION
# --------------------------------------------------

def extract_field(record, patterns):
    for pattern in patterns:
        match = re.search(
            pattern + r"\s*:\s*([^.;]+)",
            record,
            re.IGNORECASE
        )
        if match:
            return match.group(1).strip()
    return None


def get_category(question):
    q = question.lower()

    if any(x in q for x in ["registration", "registration number"]):
        return "registration"

    if any(x in q for x in ["fatalit", "deaths", "killed"]):
        return "fatalities"

    if any(x in q for x in ["operator", "airline"]):
        return "operator"

    if any(x in q for x in ["cause", "reason"]):
        return "cause"

    if any(x in q for x in ["on board", "onboard", "occupants"]):
        return "onboard"

    if "location" in q or "where" in q:
        return "location"

    if any(x in q for x in [
        "aircraft", "airplane", "aeroplane",
        "plane", "model", "type"
    ]):
        return "aircraft"

    if any(x in q for x in [
        "what happened", "incident details",
        "accident details", "summary", "describe the incident"
    ]):
        return "summary"

    return None


FIELD_PATTERNS = {
    "registration": [
        r"Aircraft Registration",
        r"Registration Number",
        r"Registration"
    ],
    "fatalities": [
        r"Total Fatalities",
        r"Total Fatal Injuries",
        r"Fatalities"
    ],
    "operator": [
        r"Aircraft Operator",
        r"Operator",
        r"Airline"
    ],
    "cause": [
        r"Incident Cause\(es\)",
        r"Cause"
    ],
    "onboard": [
        r"People on board",
        r"Persons on board",
        r"Onboard Total.*?Occupants"
    ],
    "location": [
        r"Incident Location",
        r"Location"
    ],
    "aircraft": [
        r"Aircraft Model",
        r"A/C Type",
        r"Aircraft Type",
        r"Model"
    ]
}

DISPLAY_NAMES = {
    "registration": "Aircraft registration",
    "fatalities": "Total fatalities",
    "operator": "Operator",
    "cause": "Incident cause",
    "onboard": "People on board",
    "location": "Incident location",
    "aircraft": "Aircraft type"
}

# --------------------------------------------------
# CHATBOT
# --------------------------------------------------

AVIATION_TERMS = [
    "aviation", "aircraft", "airplane", "aeroplane",
    "flight", "airline", "pilot", "plane",
    "crash", "accident", "incident", "registration",
    "fatalities", "fatality", "deaths", "killed",
    "operator", "cause", "aircraft type", "aircraft model",
    "location", "onboard", "on board", "survivors"
]


def ask_ollama(question, record, expected):

    prompt = f"""
Use only the aviation record below.

Return only the exact value requested.
Do not add explanations or outside information.
If the value is missing, return NOT_FOUND.

AVIATION RECORD:
{record}

QUESTION:
{question}

EXPECTED VALUE FORMAT:
{expected}

ANSWER:
"""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0}
        },
        timeout=120
    )

    response.raise_for_status()
    return response.json().get("response", "").strip()


def aviation_chatbot(question):

    question = question.strip()

    # Check aviation-related intent first
    if not any(term in question.lower() for term in AVIATION_TERMS):
        return "I can only answer aviation incident-related questions."

    # Check the required date format
    dates = re.findall(
        r'(?<!\d)\d{2}/\d{2}/\d{4}(?!\d)',
        question
    )

    if len(dates) != 1:
        return "Please provide exactly one full date in DD/MM/YYYY format."

    date = dates[0]

    try:
        datetime.strptime(date, "%d/%m/%Y")
    except ValueError:
        return "The supplied date is invalid."

    records = date_lookup.get(date, [])

    if not records:
        return "The information is not available in the provided aviation records."

    if len(records) > 1:
        return "Multiple records exist for this date. Please provide more details."

    record = records[0]
    category = get_category(question)

    if category is None:
        return (
            "Please ask about the aircraft, registration, operator, "
            "fatalities, location, cause, people on board, or incident details."
        )

    if category == "summary":

        operator = extract_field(record, FIELD_PATTERNS["operator"])
        aircraft = extract_field(record, FIELD_PATTERNS["aircraft"])
        location = extract_field(record, FIELD_PATTERNS["location"])
        fatalities = extract_field(record, FIELD_PATTERNS["fatalities"])
        cause = extract_field(record, FIELD_PATTERNS["cause"])

        details = [
            f"Operator: {operator}" if operator else None,
            f"Aircraft: {aircraft}" if aircraft else None,
            f"Location: {location}" if location else None,
            f"Total fatalities: {fatalities}" if fatalities else None,
            f"Cause: {cause}" if cause else None
        ]

        details = [x for x in details if x]

        if not details:
            return (
                "Incident details are not specified "
                "in the provided aviation records."
            )

        return f"Incident details for {date}:\n" + "\n".join(details)

    expected = extract_field(record, FIELD_PATTERNS[category])

    if not expected:
        return "The information is not specified in the provided aviation records."

    dataset_answer = f"{DISPLAY_NAMES[category]} on {date}: {expected}"

    try:

        ollama_answer = ask_ollama(question, record, expected)

        # Accept Ollama only when its complete answer matches
        # the verified dataset value.

        if ollama_answer.strip().casefold() == expected.strip().casefold():
            return f"Verified Ollama Answer: {ollama_answer}"

        # Save incorrect Ollama responses for later analysis

        with open(
            Path(__file__).resolve().parent / "ollama_mismatch_log.txt",
            "a",
            encoding="utf-8"
        ) as log:

            log.write(f"Question: {question}\n")
            log.write(f"Ollama response: {ollama_answer}\n")
            log.write(f"Verified value: {expected}\n")
            log.write("-" * 50 + "\n")

        return f"Answer: {dataset_answer}"

    except requests.exceptions.RequestException:

        return (
            f"Verified Dataset Answer: {dataset_answer}\n"
            "[Ollama connection unavailable. Dataset answer used.]"
        )


# --------------------------------------------------
# COMMAND PROMPT CHAT INTERFACE
# --------------------------------------------------

def start_chatbot():

    print("\n" + "=" * 55)
    print("       AVIATION OLLAMA RAG CHATBOT")
    print("=" * 55)
    print("Model:", OLLAMA_MODEL)
    print("Enter an aviation question with a full date.")
    print("Type 'exit' to close the chatbot.")
    print("=" * 55)

    while True:

        try:

            question = input("\nYou: ").strip()

            if question.lower() in ["exit", "quit"]:
                print("Chatbot closed.")
                break

            if not question:
                print("Please enter a question.")
                continue

            answer = aviation_chatbot(question)

            print("\nAviation Chatbot:", answer)

        except KeyboardInterrupt:

            print("\nChatbot closed.")
            break


if __name__ == "__main__":
    start_chatbot()