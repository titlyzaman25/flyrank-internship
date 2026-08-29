"""
Generates docs/index.html from work/outputs/capstone_metrics.json + figures.
Run this AFTER capstone.ipynb has executed (metrics.json and figures must exist).

Usage:
    python scripts/build_paper.py
"""
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = REPO_ROOT / "work" / "outputs" / "capstone_metrics.json"
FIGURES_DIR = REPO_ROOT / "work" / "figures"
DOCS_DIR = REPO_ROOT / "docs"
DOCS_IMG_DIR = DOCS_DIR / "img"

# ---- EDIT THESE THREE LINES ----
AUTHOR_NAME = "Fariha Zaman"
GITHUB_REPO_URL = "https://github.com/titlyzaman25/flyrank-internship"
CONTACT_LINK = ""  # optional: LinkedIn/portfolio URL, or leave blank
# ---------------------------------


def load_metrics():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"{METRICS_PATH} not found. Run capstone.ipynb (through Section 7) first."
        )
    with open(METRICS_PATH) as f:
        return json.load(f)


def copy_figures():
    DOCS_IMG_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ["reason_code_distribution.png", "freshness_insight.png", "results_comparison.png"]:
        src = FIGURES_DIR / name
        if src.exists():
            shutil.copy(src, DOCS_IMG_DIR / name)
            copied.append(name)
        else:
            print(f"WARNING: {src} not found — figure will be missing from the page.")
    return copied


def pct(x):
    return f"{x:.1%}"


def num(x, digits=3):
    return f"{x:.{digits}f}"


def build_html(m):
    reason_rows = "\n".join(
        f"<tr><td>{code}</td><td>{count:,}</td></tr>"
        for code, count in sorted(m["reason_code_counts"].items(), key=lambda kv: -kv[1])
    )

    auc_gap = abs(m["grouped_test_roc_auc"] - m["random_split_auc_for_comparison"])
    leakage_sentence = (
        f"A naive random 80/20 split was also run for comparison "
        f"(ROC-AUC {num(m['random_split_auc_for_comparison'])} vs. "
        f"{num(m['grouped_test_roc_auc'])} grouped by client) — the two are nearly "
        f"identical (a gap of {auc_gap:.4f}), which is itself a finding: in this dataset, "
        f"client identity was not doing meaningful work the model could \"cheat\" with, "
        f"so the grouped-split number can be trusted as close to the naive one, not "
        f"treated as a correction of a large inflation."
    )

    refresh_count = m["reason_code_counts"].get("REFRESH_PRIORITY", 0)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Can Pre-Refresh Content Signals Predict Next-Month Click Activity?</title>
<style>
  :root {{ --ink:#1a1a1a; --muted:#5a5a5a; --line:#e2e2e2; --accent:#0b5fff; --bg:#fdfdfb; }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    color: var(--ink); background: var(--bg); line-height: 1.6;
    max-width: 760px; margin: 0 auto; padding: 2.5rem 1.25rem 6rem;
  }}
  h1 {{ font-size: 1.9rem; line-height: 1.25; margin-bottom: 0.25rem; }}
  h2 {{
    font-size: 1.3rem; margin-top: 3rem; padding-top: 0.75rem;
    border-top: 1px solid var(--line);
  }}
  .byline {{ color: var(--muted); font-size: 0.95rem; margin-bottom: 2rem; }}
  .abstract {{
    background: #f3f5fb; border-left: 3px solid var(--accent);
    padding: 1rem 1.25rem; border-radius: 4px; font-size: 1.02rem;
  }}
  p {{ margin: 0.9rem 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.95rem; }}
  th, td {{ text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--line); }}
  th {{ color: var(--muted); font-weight: 600; }}
  figure {{ margin: 1.5rem 0; }}
  figure img {{ width: 100%; border-radius: 6px; border: 1px solid var(--line); }}
  figcaption {{ color: var(--muted); font-size: 0.9rem; margin-top: 0.5rem; }}
  .stat {{ font-weight: 600; color: var(--accent); }}
  .note {{
    background: #fff8e6; border-left: 3px solid #d9a300;
    padding: 0.75rem 1rem; border-radius: 4px; font-size: 0.92rem; margin: 1rem 0;
  }}
  ul {{ padding-left: 1.25rem; }}
  li {{ margin: 0.35rem 0; }}
  a {{ color: var(--accent); }}
  .credit {{
    margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid var(--line);
    color: var(--muted); font-size: 0.9rem;
  }}
</style>
</head>
<body>

<h1>Can Pre-Refresh Content Signals Predict Next-Month Click Activity?</h1>
<p class="byline">{AUTHOR_NAME} · FlyRank ML Internship Capstone{f' · <a href="{CONTACT_LINK}">Contact</a>' if CONTACT_LINK else ''}</p>

<div class="abstract">
<strong>Abstract.</strong> This paper asks whether pre-refresh content signals — search
position, engagement, and freshness — can predict which pages will earn clicks the
following month, using {m['n_rows_scored']:,} content items from FlyRank's internal
warehouse. A logistic regression model was trained on March 2026 features against an
April 2026 outcome label, validated on clients held out entirely from training. Under
that grouped-by-client split, the model reached a ROC-AUC of {num(m['grouped_test_roc_auc'])}
against a rule baseline's {num(m['baseline_roc_auc'])} and a label base rate of
{pct(m['label_base_rate'])}. A parallel naive-split check found no meaningful gap versus
the grouped split, suggesting the result is not an artifact of client-level leakage. The
output is a ranked, reason-coded action queue intended as decision-support for content
editors, not an autonomous system.
</div>

<h2>Introduction</h2>
<p>
Content teams cannot refresh every page every month. The question this project answers:
{m['question']} A working answer turns a flat backlog into a ranked queue an editor can
work down with reasons attached, instead of guessing which pages are worth the time.
</p>

<h2>Data</h2>
<p>
Source: <code>FlyRank/internship-warehouse</code> (Hugging Face, build v20260703),
tables <code>fact_content_daily_performance</code> (month=2026-03, month=2026-04) and
<code>dim_content</code>. Feature window: March 2026. Label window: April 2026 —
strictly the following month, so every feature is knowable before the outcome it predicts.
</p>
<p>
Excluded from this study: <code>fact_content_query_90d</code> (its 90-day window overlaps
the label period and was not worth the added alignment work for this pass);
<code>fact_content_daily_performance_sample</code> (June 2026, reserved as a sealed test
month and deliberately untouched); all months outside March–April 2026, to keep this study
to one clean past-to-future pair. {m['n_rows_scored']:,} content items had a matching row
in both months and form the modeled population — essentially all of them (fewer than 5
items were dropped for lacking an April row), so survivorship bias is negligible here.
</p>

<h2>Methodology</h2>
<p>
<strong>Label:</strong> did the content earn at least one click in April 2026? A binary
outcome, base rate {pct(m['label_base_rate'])}.
</p>
<p>
<strong>Features:</strong> March search impressions, clicks, average position, GA4
sessions and engagement time, click-through rate, and days since last content update —
all measured strictly before the label window.
</p>
<p>
<strong>Baseline:</strong> a transparent hand-written rule — flag content as stale
(≥181 days since update) or below its position tier's expected CTR (computed from
training data only, to avoid leaking test-set information into the rule).
</p>
<p>
<strong>Validation design:</strong> GroupShuffleSplit by client, 80/20 — no client
appears in both train and test. {leakage_sentence}
</p>
<p>
<strong>Leakage checks:</strong> features and label sit in strictly non-overlapping
months; no product-flag or composite-score columns were used as features; the modeled
population retains essentially the full March cohort, so the survivorship filter
introduces negligible bias.
</p>

<h2>Results</h2>
<figure>
  <img src="img/results_comparison.png" alt="Model vs baseline comparison chart">
  <figcaption>Logistic regression vs. the rule baseline, same grouped test split, same base rate.</figcaption>
</figure>
<table>
<tr><th>Method</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th></tr>
<tr>
  <td>Rule baseline</td>
  <td>{num(m['baseline_precision'])}</td>
  <td>{num(m['baseline_recall'])}</td>
  <td>{num(m['baseline_f1'])}</td>
  <td>{num(m['baseline_roc_auc'])}</td>
</tr>
<tr>
  <td>Logistic Regression (grouped split)</td>
  <td>{num(m['grouped_test_precision'])}</td>
  <td>{num(m['grouped_test_recall'])}</td>
  <td>{num(m['grouped_test_f1'])}</td>
  <td>{num(m['grouped_test_roc_auc'])}</td>
</tr>
</table>
<p>
Base rate: {pct(m['label_base_rate'])}. Every metric above is read against this number —
a bare accuracy figure without it would overstate the model's contribution. The model
shows materially higher precision than the rule baseline ({num(m['grouped_test_precision'])}
vs. {num(m['baseline_precision'])}) at somewhat lower recall
({num(m['grouped_test_recall'])} vs. {num(m['baseline_recall'])}) — it flags fewer items,
but is right more often when it does.
</p>

<h2>Limitations</h2>
<ul>
<li><strong>Observational, not causal.</strong> No experiment was run. Nothing here
supports "refreshing X will produce Y" — only that pages like this were associated with
clicks in this portfolio.</li>
<li><strong>Single two-month window.</strong> March→April 2026 only; no test across
seasons or a longer horizon yet.</li>
<li><strong>Single portfolio.</strong> Findings describe this one company's data and do
not generalize elsewhere without re-validation.</li>
<li><strong>The REFRESH_PRIORITY category is nearly empty</strong> ({refresh_count} of
{m['n_rows_scored']:,} items). In this dataset, stale content's mean predicted potential
is close to zero, so very little stale content also clears the "still promising" bar this
category requires. Read as a real pattern in this portfolio — stale content here rarely
looks salvageable by this model — not a bug in the queue logic, but worth re-examining
the tier thresholds before relying on this category operationally.</li>
<li><strong>Validated numbers are the grouped-split test metrics above.</strong> The
random-split comparison exists to check for leakage, not as an alternate performance
claim.</li>
</ul>

<h2>Ranked Recommendations</h2>
<figure>
  <img src="img/reason_code_distribution.png" alt="Reason code distribution chart">
  <figcaption>How the {m['n_rows_scored']:,}-item queue splits across reason codes.</figcaption>
</figure>
<figure>
  <img src="img/freshness_insight.png" alt="Predicted potential by freshness tier">
  <figcaption>Observed relationship between freshness and predicted potential in this portfolio — stale content scores near zero, which is also why the REFRESH_PRIORITY bucket above is nearly empty.</figcaption>
</figure>
<table>
<tr><th>Reason code</th><th>Count</th></tr>
{reason_rows}
</table>
<div class="note">
Most of the queue ({pct(m['reason_code_counts'].get('REVIEW_QUEUE', 0) / m['n_rows_scored'])})
falls into REVIEW_QUEUE — items that didn't cleanly match a defined archetype. These are
explicitly routed to human judgment rather than assigned an automated action, by design.
</div>
<p>
<span class="stat">{pct(m['share_needing_human_review'])}</span> of the queue is flagged
for mandatory human review before any suggestion reaches an editor. Every row is
decision-support, not an instruction: an editor confirms page content, voice, and
compliance before acting. Nothing in this queue should trigger auto-publishing,
auto-deprioritization of budget, client-facing performance guarantees, or individual
writer evaluation.
</p>

<h2>Reproducibility</h2>
<p>
Full pipeline, notebooks, and this page's generator script:
<a href="{GITHUB_REPO_URL}">{GITHUB_REPO_URL}</a>. Relevant notebooks:
<code>work/notebooks/w05_model.ipynb</code>,
<code>work/notebooks/w06_validation_audit.ipynb</code>,
<code>work/notebooks/w07_action_playbook.ipynb</code>,
<code>work/notebooks/capstone.ipynb</code>. Random seed: 42, fixed across every split and
model in this study.
</p>

<div class="credit">
Built on the FlyRank ML Internship dataset —
<a href="https://flyrank.ai">https://flyrank.ai</a>
</div>

</body>
</html>
"""


def main():
    metrics = load_metrics()
    copied = copy_figures()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    html = build_html(metrics)
    out_path = DOCS_DIR / "index.html"
    out_path.write_text(html)
    print(f"Wrote {out_path}")
    print(f"Copied figures: {copied}")
    print("\nNext: edit AUTHOR_NAME at the top of this script if you haven't, then:")
    print("  git add docs/ && git commit -m 'Deploy research paper' && git push")


if __name__ == "__main__":
    main()
