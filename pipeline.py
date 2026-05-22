"""
Support Triage Pipeline - Uses Groq (free, no credit card needed)
Stages: INIT -> INPUTS_LOADED -> TICKETS_NORMALIZED -> TRIAGE_PREDICTED
     -> HUMAN_REVIEW_COMPLETE -> FINAL_QUEUE_GENERATED -> VALIDATION_COMPLETE
     -> RESULTS_FINALISED
"""

import json, os, sys, hashlib, datetime, re

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

def log(stage, msg):
    print(f"[{stage}] {msg}")

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved -> {path}")

def word_count(text):
    return len(text.split())

def prompt_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]

def append_llm_log(entry):
    with open("llm_calls.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def stage_init():
    log("INIT", "Starting pipeline...")
    if os.path.exists("llm_calls.jsonl"):
        os.remove("llm_calls.jsonl")

def stage_inputs_loaded():
    log("INPUTS_LOADED", "Reading tickets.json and triage_config.json from disk...")
    tickets = load_json("tickets.json")
    config  = load_json("triage_config.json")
    log("INPUTS_LOADED", f"  {len(tickets)} tickets loaded.")
    log("INPUTS_LOADED", f"  Allowed categories: {config['allowed_categories']}")
    return tickets, config

def normalize_text(subject, message):
    s = re.sub(r'\s+', ' ', subject.strip())
    m = re.sub(r'\s+', ' ', message.strip())
    return f"Subject: {s} | Message: {m}"

def stage_tickets_normalized(tickets):
    log("TICKETS_NORMALIZED", "Normalizing tickets (deterministic, pre-LLM)...")
    normalized = []
    for t in tickets:
        text_for_model = normalize_text(t["subject"], t["message"])
        normalized.append({
            "ticket_id":      t["ticket_id"],
            "subject":        t["subject"].strip(),
            "message":        t["message"].strip(),
            "channel":        t["channel"].strip(),
            "created_at":     t["created_at"],
            "text_for_model": text_for_model,
            "char_count":     len(text_for_model)
        })
    save_json("normalized_tickets.json", normalized)
    log("TICKETS_NORMALIZED", "Done.")
    return normalized

def build_triage_prompt(normalized_tickets, config):
    tickets_block = json.dumps(
        [{"ticket_id": t["ticket_id"], "text": t["text_for_model"]} for t in normalized_tickets],
        indent=2
    )
    categories = config["allowed_categories"]
    priorities = config["allowed_priorities"]
    tone       = config["reply_style"]["tone"]
    max_words  = config["reply_style"]["max_words"]
    routing    = config["routing_rules"]
    return f"""You are a support triage assistant. Classify each ticket below.

ALLOWED CATEGORIES (use exactly): {categories}
ALLOWED PRIORITIES (use exactly): {priorities}
ROUTING RULES: {json.dumps(routing)}
REPLY TONE: {tone}
REPLY MAX WORDS: {max_words}

TICKETS:
{tickets_block}

Return ONLY a valid JSON array, no markdown, no explanation. Each element:
{{
  "ticket_id": "string",
  "category": "<one of allowed_categories>",
  "priority": "<one of allowed_priorities>",
  "confidence": <float 0.0-1.0>,
  "reason": "one sentence",
  "suggested_reply": "reply under {max_words} words",
  "route_to": "<from routing rules>"
}}"""

def call_groq(prompt, api_key):
    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

def parse_predictions(raw_text, normalized_tickets, config):
    categories = config["allowed_categories"]
    priorities = config["allowed_priorities"]
    routing    = config["routing_rules"]
    clean      = re.sub(r"```(?:json)?", "", raw_text).strip().strip("`").strip()

    try:
        predictions = json.loads(clean)
    except json.JSONDecodeError:
        log("TRIAGE_PREDICTED", "WARNING: Could not parse LLM response. Using fallback.")
        predictions = []

    pred_map  = {p["ticket_id"]: p for p in predictions if isinstance(p, dict)}
    validated = []

    for t in normalized_tickets:
        tid = t["ticket_id"]
        p   = pred_map.get(tid, {})
        if not p:
            log("TRIAGE_PREDICTED", f"  WARNING: No prediction for {tid} - fallback used.")

        category   = p.get("category", "other")
        if category not in categories: category = "other"
        priority   = p.get("priority", "normal")
        if priority not in priorities: priority = "normal"
        route_to   = routing.get(category, "manual_review_queue")
        reply      = p.get("suggested_reply", "Thank you for contacting support. An agent will assist you shortly.")
        confidence = max(0.0, min(1.0, float(p.get("confidence", 0.5))))
        if word_count(reply) > config["reply_style"]["max_words"]:
            reply = " ".join(reply.split()[:config["reply_style"]["max_words"]])

        validated.append({
            "ticket_id":       tid,
            "category":        category,
            "priority":        priority,
            "confidence":      round(confidence, 2),
            "reason":          p.get("reason", "Fallback classification."),
            "suggested_reply": reply,
            "route_to":        route_to
        })
    return validated

def stage_triage_predicted(normalized_tickets, config, api_key):
    log("TRIAGE_PREDICTED", "Calling LLM (Groq - free) to classify all tickets...")
    prompt    = build_triage_prompt(normalized_tickets, config)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    raw       = call_groq(prompt, api_key)
    append_llm_log({
        "stage": "TRIAGE_PREDICTED", "timestamp": timestamp,
        "provider": "groq", "model": "llama3-8b-8192",
        "prompt_hash": prompt_hash(prompt),
        "input_artifacts": ["normalized_tickets.json", "triage_config.json"],
        "output_artifact": "triage_predictions.json"
    })
    predictions = parse_predictions(raw, normalized_tickets, config)
    save_json("triage_predictions.json", predictions)
    log("TRIAGE_PREDICTED", f"  {len(predictions)} predictions saved.")
    return predictions

def stage_human_review(predictions, config):
    print("\n" + "="*60)
    print("  HUMAN REVIEW CHECKPOINT")
    print("="*60)
    print(f"  {'Ticket':<10} {'Category':<20} {'Priority':<10} {'Confidence'}")
    print(f"  {'-'*10} {'-'*20} {'-'*10} {'-'*10}")
    for p in predictions:
        print(f"  {p['ticket_id']:<10} {p['category']:<20} {p['priority']:<10} {p['confidence']}")
    print("="*60)
    print("\nEnter overrides as:  ticket_id,category,priority")
    print("Press Enter on an empty line when done.\n")

    categories = config["allowed_categories"]
    priorities = config["allowed_priorities"]
    routing    = config["routing_rules"]
    pred_map   = {p["ticket_id"]: p for p in predictions}
    overrides  = []

    while True:
        try:
            line = input("Override> ").strip()
        except EOFError:
            break
        if not line:
            break
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 3:
            print("  Format: ticket_id,category,priority")
            continue
        tid, new_cat, new_pri = parts
        if tid not in pred_map:
            print(f"  Unknown ticket: {tid}")
            continue
        if new_cat not in categories:
            print(f"  Invalid category. Allowed: {categories}")
            continue
        if new_pri not in priorities:
            print(f"  Invalid priority. Allowed: {priorities}")
            continue
        old_cat = pred_map[tid]["category"]
        old_pri = pred_map[tid]["priority"]
        overrides.append({"ticket_id": tid, "old_category": old_cat,
                          "new_category": new_cat, "old_priority": old_pri, "new_priority": new_pri})
        pred_map[tid]["category"] = new_cat
        pred_map[tid]["priority"] = new_pri
        pred_map[tid]["route_to"] = routing.get(new_cat, "manual_review_queue")
        print(f"  Override applied for {tid}")

    save_json("review_overrides.json", overrides)
    log("HUMAN_REVIEW_COMPLETE", f"  {len(overrides)} override(s) applied.")
    return list(pred_map.values()), overrides

def stage_final_queue(predictions, overrides):
    log("FINAL_QUEUE_GENERATED", "Building final queue...")
    overridden_ids = {o["ticket_id"] for o in overrides}
    final_queue    = []
    for p in predictions:
        final_queue.append({
            "ticket_id":       p["ticket_id"],
            "final_category":  p["category"],
            "final_priority":  p["priority"],
            "final_route_to":  p["route_to"],
            "suggested_reply": p["suggested_reply"],
            "was_overridden":  p["ticket_id"] in overridden_ids
        })
    save_json("final_queue.json", final_queue)

    by_cat = {}; by_pri = {}; by_dest = {}
    for item in final_queue:
        by_cat[item["final_category"]]  = by_cat.get(item["final_category"], 0) + 1
        by_pri[item["final_priority"]]  = by_pri.get(item["final_priority"], 0) + 1
        by_dest[item["final_route_to"]] = by_dest.get(item["final_route_to"], 0) + 1

    lines = [f"# Queue Summary\n\n**Total Tickets:** {len(final_queue)}\n\n## By Category\n"]
    for k, v in by_cat.items(): lines.append(f"- {k}: {v}\n")
    lines.append("\n## By Priority\n")
    for k, v in by_pri.items(): lines.append(f"- {k}: {v}\n")
    lines.append("\n## By Destination Queue\n")
    for k, v in by_dest.items(): lines.append(f"- {k}: {v}\n")
    lines.append("\n## Overridden Tickets\n")
    if overrides:
        for o in overrides:
            lines.append(f"- {o['ticket_id']}: {o['old_category']} -> {o['new_category']}, {o['old_priority']} -> {o['new_priority']}\n")
    else:
        lines.append("- None\n")

    with open("queue_summary.md", "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("  Saved -> queue_summary.md")
    log("FINAL_QUEUE_GENERATED", "Done.")
    return final_queue

def stage_escalations(predictions):
    log("VALIDATION_COMPLETE", "Computing escalations...")
    escalations = []
    for p in predictions:
        reason = None
        if p["category"] == "other":
            reason = "category is 'other'"
        elif p["confidence"] < 0.60:
            reason = f"confidence {p['confidence']} < 0.60"
        if reason:
            escalations.append({"ticket_id": p["ticket_id"], "category": p["category"],
                                 "priority": p["priority"], "confidence": p["confidence"],
                                 "escalation_reason": reason})
    save_json("escalations.json", escalations)
    log("VALIDATION_COMPLETE", f"  {len(escalations)} ticket(s) flagged.")

def main():
    api_key = os.environ.get("GROQ_API_KEY") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not api_key:
        print("ERROR: python pipeline.py YOUR_GROQ_API_KEY")
        sys.exit(1)
    if not GROQ_AVAILABLE:
        print("ERROR: Run:  pip install groq")
        sys.exit(1)

    stage_init()
    tickets, config        = stage_inputs_loaded()
    normalized             = stage_tickets_normalized(tickets)
    predictions            = stage_triage_predicted(normalized, config, api_key)
    predictions, overrides = stage_human_review(predictions, config)
    stage_final_queue(predictions, overrides)
    stage_escalations(predictions)

    print("\n" + "="*60)
    print("  RESULTS_FINALISED - Pipeline complete!")
    for f in ["normalized_tickets.json","triage_predictions.json","review_overrides.json",
              "final_queue.json","queue_summary.md","escalations.json","llm_calls.jsonl"]:
        print(f"    [{'OK' if os.path.exists(f) else 'MISSING'}]  {f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
