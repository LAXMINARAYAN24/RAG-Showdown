# scripts/export_log.py
import json

rows = []
with open("logs/qa_log.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

with open("logs/qa_report.md", "w", encoding="utf-8") as out:
    out.write("| Timestamp | Strategy | Question | Answer | Latency (s) |\n")
    out.write("|---|---|---|---|---|\n")
    for r in rows:
        answer_short = r["answer"].replace("\n", " ")[:150] + "..."
        out.write(f"| {r['timestamp']} | {r['strategy']} | {r['question']} | {answer_short} | {r['latency_seconds']:.2f} |\n")

print(f"Wrote {len(rows)} rows to logs/qa_report.md")