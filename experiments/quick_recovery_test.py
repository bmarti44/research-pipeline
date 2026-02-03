"""
Quick recovery test on silent failure trials.
"""
import json
import requests
import re
import time

OLLAMA_URL = "http://localhost:11434/api/generate"

EXTRACTION_PROMPT = """Extract any signal acknowledgment from this assistant response.

Rules:
1. If the assistant acknowledged frustration, urgency, or a blocking issue, output EXACTLY this format:
<signal type="TYPE">brief reason</signal>
Where TYPE is one of: frustration, urgency, blocking_issue

2. If the assistant did NOT acknowledge any such signal, output EXACTLY:
NO_SIGNAL

3. Output NOTHING else.

Assistant response:
\"\"\"
{response_text}
\"\"\""""


def parse_extraction(raw: str) -> dict:
    """Parse extraction output."""
    raw = raw.strip()
    raw = re.sub(r'^```\w*\n?', '', raw)
    raw = re.sub(r'\n?```$', '', raw)
    raw = raw.strip()

    match = re.search(
        r'<signal\s+type=["\']?(\w+)["\']?\s*>(.+?)</signal>',
        raw, re.IGNORECASE | re.DOTALL
    )
    if match:
        return {"extracted": True, "signal_type": match.group(1).lower(), "raw": raw}

    if re.search(r'no.?signal', raw, re.IGNORECASE):
        return {"extracted": False, "signal_type": None, "raw": raw}

    return {"extracted": False, "signal_type": None, "raw": raw, "parse_error": True}


def extract(response_text: str) -> dict:
    """Run extraction on a single response."""
    prompt = EXTRACTION_PROMPT.format(response_text=response_text)

    start = time.time()
    r = requests.post(OLLAMA_URL, json={
        "model": "qwen2.5:7b-instruct",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 150}
    }, timeout=120)
    latency = time.time() - start

    raw = r.json().get("response", "")
    result = parse_extraction(raw)
    result["latency_ms"] = int(latency * 1000)
    return result


def main():
    # Load data
    d = json.load(open("experiments/results/signal_detection_20260203_074411_judged.json"))
    results = d["results"]

    # Find silent failures in IMPLICIT with NL judge detection (recoverable)
    silent_failures = [
        x for x in results
        if x.get("ambiguity") == "IMPLICIT"
        and x.get("st_judge_detected") is True
        and x.get("st_regex_detected") is not True
        and x.get("nl_judge_detected") is True  # Only test recoverable ones
    ]

    print(f"Testing recovery on {len(silent_failures)} silent failures with NL detection")
    print()

    recovered = 0
    type_correct = 0
    parse_errors = 0
    total_latency = 0

    for i, trial in enumerate(silent_failures):
        nl_response = trial.get("nl_response_text", "")
        expected_type = trial.get("signal_type", "")

        result = extract(nl_response)
        total_latency += result.get("latency_ms", 0)

        if result.get("parse_error"):
            parse_errors += 1

        if result["extracted"]:
            recovered += 1
            if result["signal_type"] == expected_type:
                type_correct += 1

        # Progress
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(silent_failures)}, recovered: {recovered}")

    print()
    print("=" * 60)
    print("SILENT FAILURE RECOVERY RESULTS")
    print("=" * 60)
    print(f"Total silent failures tested:  {len(silent_failures)}")
    print(f"Recovered (extracted signal):  {recovered} ({100*recovered/len(silent_failures):.1f}%)")
    print(f"Type correct (of recovered):   {type_correct}/{recovered} ({100*type_correct/recovered if recovered > 0 else 0:.1f}%)")
    print(f"Parse errors:                  {parse_errors}")
    print(f"Avg latency per extraction:    {total_latency//len(silent_failures)}ms")
    print("=" * 60)

    # Save results
    output = {
        "total_silent_failures": len(silent_failures),
        "recovered": recovered,
        "recovery_rate": recovered / len(silent_failures) if silent_failures else 0,
        "type_correct": type_correct,
        "type_accuracy": type_correct / recovered if recovered > 0 else 0,
        "parse_errors": parse_errors,
        "avg_latency_ms": total_latency // len(silent_failures) if silent_failures else 0,
    }

    with open("experiments/results/quick_recovery_test.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to experiments/results/quick_recovery_test.json")


if __name__ == "__main__":
    main()
