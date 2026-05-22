"""
Validation script for the support triage pipeline.
Usage: python validate.py
"""

import json
import os
import sys

PASS = "✓"
FAIL = "✗"
errors = []
warnings = []

def check(condition, label, detail=""):
    if condition:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}" + (f" — {detail}" if detail else ""))
        errors.append(label)

def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

print("\n" + "="*60)
print("  VALIDATION")
print("="*60)

# 1. Required artifacts exist
print("\n[1] Required artifacts exist")
required = [
    "tickets.json", "triage_config.json", "normalized_tickets.json",
    "triage_predictions.json", "review_overrides.json",
    "final_queue.json", "queue_summary.md"
]
for path in required:
    check(os.path.exists(path), f"Exists: {path}")

# 2. JSON files are valid
print("\n[2] JSON files are valid")
json_files = [
    "tickets.json", "triage_config.json", "normalized_tickets.json",
    "triage_predictions.json", "review_overrides.json", "final_queue.json"
]
for path in json_files:
    if os.path.exists(path):
        try:
            with open(path) as f:
                json.load(f)
            check(True, f"Valid JSON: {path}")
        except Exception as e:
            check(False, f"Valid JSON: {path}", str(e))

# 3. Load data
tickets         = load("tickets.json") or []
config          = load("triage_config.json") or {}
normalized      = load("normalized_tickets.json") or []
predictions     = load("triage_predictions.json") or []
overrides       = load("review_overrides.json") or []
final_queue     = load("final_queue.json") or []
allowed_cats    = config.get("allowed_categories", [])
allowed_pris    = config.get("allowed_priorities", [])
routing         = config.get("routing_rules", {})
max_words       = config.get("reply_style", {}).get("max_words", 80)

# 4. Normalization happened before LLM (normalized file exists and has text_for_model)
print("\n[3] Normalization before LLM call")
check(len(normalized) > 0, "normalized_tickets.json is not empty")
if normalized:
    check(
        all("text_for_model" in t and "char_count" in t for t in normalized),
        "All normalized tickets have text_for_model and char_count"
    )
    check(
        all("text_for_model" not in t for t in tickets),
        "Raw tickets.json does NOT have text_for_model (normalization is separate)"
    )

# 5. Every ticket has exactly one prediction
print("\n[4] Every ticket has exactly one prediction")
ticket_ids  = {t["ticket_id"] for t in tickets}
pred_ids    = [p["ticket_id"] for p in predictions]
check(len(pred_ids) == len(ticket_ids), f"Prediction count ({len(pred_ids)}) matches ticket count ({len(ticket_ids)})")
check(len(pred_ids) == len(set(pred_ids)), "No duplicate predictions")
for tid in ticket_ids:
    check(tid in pred_ids, f"Prediction exists for {tid}")

# 6. Category and priority values are restricted to config values
print("\n[5] Category and priority restricted to config")
for p in predictions:
    check(p["category"] in allowed_cats, f"{p['ticket_id']} category '{p['category']}' is allowed")
    check(p["priority"] in allowed_pris, f"{p['ticket_id']} priority '{p['priority']}' is allowed")

# 7. Route mappings match config
print("\n[6] Route mappings match config")
for p in predictions:
    expected_route = routing.get(p["category"])
    check(p["route_to"] == expected_route,
          f"{p['ticket_id']} route_to '{p['route_to']}' matches config",
          f"expected '{expected_route}'")

# 8. Overrides are valid and applied in final outputs
print("\n[7] Overrides applied in final outputs")
if overrides:
    final_map = {f["ticket_id"]: f for f in final_queue}
    for o in overrides:
        tid = o["ticket_id"]
        if tid in final_map:
            check(final_map[tid]["final_category"] == o["new_category"],
                  f"{tid} final_category matches override")
            check(final_map[tid]["final_priority"] == o["new_priority"],
                  f"{tid} final_priority matches override")
            check(final_map[tid]["was_overridden"] == True,
                  f"{tid} was_overridden is True")
        else:
            check(False, f"{tid} exists in final_queue")
else:
    print("  —  No overrides to validate (review_overrides.json is empty)")

# 9. Reply length
print("\n[8] Reply length respects max_words")
for p in predictions:
    wc = len(p.get("suggested_reply","").split())
    check(wc <= max_words,
          f"{p['ticket_id']} reply word count ({wc}) <= {max_words}")

# 10. Final queue categories/priorities valid
print("\n[9] Final queue values valid")
for f in final_queue:
    check(f["final_category"] in allowed_cats, f"{f['ticket_id']} final_category allowed")
    check(f["final_priority"] in allowed_pris, f"{f['ticket_id']} final_priority allowed")
    expected = routing.get(f["final_category"])
    check(f["final_route_to"] == expected,
          f"{f['ticket_id']} final_route_to correct",
          f"got '{f['final_route_to']}', expected '{expected}'")

# Summary
print("\n" + "="*60)
if errors:
    print(f"  FAILED — {len(errors)} check(s) failed:")
    for e in errors:
        print(f"    {FAIL} {e}")
    sys.exit(1)
else:
    print("  ALL CHECKS PASSED ✓")
print("="*60 + "\n")
