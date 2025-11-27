import os
import json
import time
from email.utils import formatdate
from datetime import datetime, timedelta, timezone

import requests
import feedparser
from dateutil import parser as dateparser
from openai import OpenAI
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# ===== ENV / CONFIG =====

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")

FROM_EMAIL = os.getenv("JOB_FROM_EMAIL")   # verified SendGrid sender
RECIPIENT_EMAIL = os.getenv("JOB_TO_EMAIL")  # where the shortlist lands

# Lookback + limits
LOOKBACK_DAYS = int(os.getenv("JOB_LOOKBACK_DAYS", "2"))
MAX_JOBS_TOTAL = int(os.getenv("JOB_MAX_TOTAL", "80"))
MAX_JOBS_TO_MODEL = int(os.getenv("JOB_MAX_TO_MODEL", "5"))  # keep small: we’re generating full CV + letter

# Target filters
TARGET_LOCATIONS = [
    s.strip().lower()
    for s in os.getenv(
        "TARGET_LOCATIONS",
        "sydney, melbourne, brisbane, perth, riyadh, dubai, remote"
    ).split(",")
    if s.strip()
]

TARGET_TITLES = [
    s.strip().lower()
    for s in os.getenv(
        "TARGET_TITLES",
        "chief marketing, cmo, head of marketing, vp marketing, gm marketing, marketing director, growth, commercial, revenue"
    ).split(",")
    if s.strip()
]

MODEL_NAME = os.getenv("JOB_MODEL_NAME", "gpt-4o-mini")
BASE_RESUME_PATH = os.getenv("BASE_RESUME_PATH", "resume_base.txt")

FEED_HTTP_TIMEOUT = int(os.getenv("FEED_HTTP_TIMEOUT", "8"))
TIMEBOX_SECONDS = int(os.getenv("JOB_TIMEBOX_SECONDS", "90"))

DIAGNOSTICS = os.getenv("JOB_DIAGNOSTICS", "0").lower() in ("1", "true", "yes")

client = OpenAI(api_key=OPENAI_API_KEY)

# ===== JOB FEEDS (placeholder – replace with your saved RSS searches) =====

JOB_FEEDS = [
    # Replace these with RSS feeds from LinkedIn / SEEK / company job boards where possible.
    "https://weworkremotely.com/categories/remote-marketing-jobs.rss",
    "https://stackoverflow.com/jobs/feed?tl=marketing",
]

def debug(msg: str) -> None:
    if DIAGNOSTICS:
        print(msg)

# ===== FETCH JOBS =====

def fetch_recent_jobs(feeds):
    start_ts = time.time()
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    jobs = []

    headers = {"User-Agent": "JobRadar/1.0 (+https://github.com/marmik)"}

    for feed_url in feeds:
        if time.time() - start_ts > TIMEBOX_SECONDS:
            debug(f"⏱️ Timebox reached ({TIMEBOX_SECONDS}s). Stopping feed fetch.")
            break

        try:
            resp = requests.get(feed_url, headers=headers, timeout=FEED_HTTP_TIMEOUT)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
        except Exception as e:
            debug(f"⚠️ Fetch/parse error {feed_url}: {e}")
            continue

        entries = getattr(feed, "entries", []) or []
        debug(f"🔎 Feed ok: {feed_url} → entries={len(entries)}")

        for entry in entries:
            pub_date = None
            for k in ("published", "updated", "created"):
                if k in entry:
                    try:
                        pub_date = dateparser.parse(entry[k])
                        break
                    except Exception:
                        pass
            if not pub_date:
                pub_date = datetime.now(timezone.utc)
            if not pub_date.tzinfo:
                pub_date = pub_date.replace(tzinfo=timezone.utc)

            if pub_date < cutoff:
                continue

            title = (entry.get("title") or "").strip()
            url = (entry.get("link") or "").strip()
            if not title or not url:
                continue

            summary = (entry.get("summary", "") or "")
            summary_plain = summary[:3000]  # enough context, avoid huge tokens

            company = (entry.get("author") or "").strip()

            # crude location detection
            location = ""
            text_blob = (title + " " + summary).lower()
            for loc in TARGET_LOCATIONS:
                if loc in text_blob:
                    location = loc.title()
                    break

            jobs.append({
                "title": title,
                "url": url,
                "summary": summary_plain,
                "published": pub_date.isoformat(),
                "company": company,
                "location": location,
                "source": feed.get("feed", {}).get("title", feed_url),
            })

            if len(jobs) >= MAX_JOBS_TOTAL:
                debug(f"🔪 Reached MAX_JOBS_TOTAL={MAX_JOBS_TOTAL}.")
                break

        if len(jobs) >= MAX_JOBS_TOTAL:
            break

    # Deduplicate by URL & sort newest first
    seen = set()
    deduped = []
    for j in sorted(jobs, key=lambda x: x["published"], reverse=True):
        if j["url"] in seen:
            continue
        seen.add(j["url"])
        deduped.append(j)

    debug(f"📦 Totals: fetched={len(jobs)} deduped={len(deduped)}")
    return deduped

# ===== FILTER FOR YOUR PROFILE =====

def filter_jobs_for_profile(jobs):
    """
    Filter for senior marketing / commercial roles and prefer known locations.
    """
    filtered = []
    for j in jobs:
        title_l = j["title"].lower()

        if not any(t in title_l for t in TARGET_TITLES):
            continue

        # location score: known target locations > unknown
        loc_score = 2 if j.get("location") else 1

        filtered.append((loc_score, j))

    # Sort: location score desc, then recency (published already ISO string)
    filtered.sort(key=lambda x: (-x[0], x[1]["published"]), reverse=False)
    result = [j for _, j in filtered][:MAX_JOBS_TO_MODEL]
    debug(f"Filtered jobs count: {len(result)}")
    return result

# ===== LOAD BASE RESUME =====

def load_base_resume():
    if not os.path.exists(BASE_RESUME_PATH):
        raise RuntimeError(
            f"Base resume file not found at {BASE_RESUME_PATH}. "
            "Create this file with your latest CV text."
        )
    with open(BASE_RESUME_PATH, "r", encoding="utf-8") as f:
        return f.read()

# ===== OPENAI ENRICHMENT (shortlist + tailored CV + full cover letter) =====

def enrich_jobs_with_ai(jobs, base_resume_text):
    if not jobs:
        return []

    system_prompt = (
        "You are an executive recruiter and CV strategist for Marmik Vyas, "
        "a GM/CMO-level marketing & commercial leader with 24+ years experience across Ogilvy, Dell, Lenovo, "
        "nbn Australia, and ALAT (PIF-backed tech manufacturing). "
        "You deeply understand senior B2C/B2B marketing, martech & AI, commercial growth, and P&L ownership.\n\n"
        "Your job: for each short-listed role, evaluate fit and generate:\n"
        "- A clear fit score (0–100)\n"
        "- A short explanation of fit and risk\n"
        "- A tailored resume version (full text) grounded in the base CV but optimised to the JD\n"
        "- A complete cover letter addressed generically to the hiring manager\n"
        "- A concise email subject line for direct outreach.\n\n"
        "Tone: senior, commercial, confident, not waffly."
    )

    user_prompt = (
        "Here is Marmik's current base resume (may be truncated if very long):\n\n"
        f"{base_resume_text[:9000]}\n\n"
        "Here is the list of potential roles to analyse:\n"
        f"{json.dumps(jobs)}\n\n"
        "For EACH job, return JSON ONLY (no commentary) in this format:\n"
        "[{\n"
        '  "job_title": "",\n'
        '  "company": "",\n'
        '  "location": "",\n'
        '  "url": "",\n'
        '  "published": "",\n'
        '  "fit_score": 0,\n'
        '  "why_fit": "1–3 sentences on why this role is a strong fit or not.",\n'
        '  "risk_flags": "Any obvious mismatch (location, level, sector).",\n'
        '  "resume_version": "Full tweaked resume text for this role. Keep it realistic and consistent with his background.",\n'
        '  "cover_letter_full": "A complete 3–5 paragraph cover letter to the hiring manager for this specific role.",\n'
        '  "email_subject": "Concise subject line for direct email to the hiring manager."\n'
        "}]\n\n"
        "Important:\n"
        "- Do not invent experience he does not have.\n"
        "- You may re-order and rephrase his achievements to match the JD.\n"
        "- Ensure the resume_version remains credible for executive roles.\n"
        "- Keep fit_score honest. 80+ only if it is genuinely strong."
    )

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.25,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = resp.choices[0].message.content
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            # sort by fit_score desc
            parsed_sorted = sorted(
                parsed,
                key=lambda x: x.get("fit_score", 0),
                reverse=True
            )
            return parsed_sorted
        raise ValueError("Parsed response is not a list")
    except Exception as e:
        debug(f"⚠️ Could not parse AI JSON: {e}")
        # Fallback: basic wrapping
        enriched = []
        for j in jobs:
            enriched.append({
                "job_title": j["title"],
                "company": j.get("company", ""),
                "location": j.get("location", ""),
                "url": j["url"],
                "published": j["published"],
                "fit_score": 50,
                "why_fit": "Potentially relevant senior marketing/commercial role.",
                "risk_flags": "",
                "resume_version": base_resume_text,
                "cover_letter_full": "I am writing to express my interest in this senior marketing role.",
                "email_subject": f"Exploring fit for {j['title']} role",
            })
        return enriched

# ===== EMAIL BUILDER =====

def build_email_html(enriched_jobs):
    if not enriched_jobs:
        return "<h3>No suitable roles found in the last window. Consider widening feeds or keywords.</h3>"

    rows = []
    for i, item in enumerate(enriched_jobs, 1):
        rows.append(f"""
        <tr>
          <td style="padding:16px;border-bottom:1px solid #ddd;vertical-align:top;">
            <strong>{i}. {item.get('job_title','')} – {item.get('company','')}</strong><br>
            <em>{item.get('location','')}</em><br>
            <a href="{item.get('url','')}">{item.get('url','')}</a><br>
            <em>Posted:</em> {item.get('published','')}<br>
            <em>Fit score:</em> {item.get('fit_score',0)}/100<br>
            <em>Why fit:</em> {item.get('why_fit','')}<br>
            <em>Risks:</em> {item.get('risk_flags','')}<br><br>
            <strong>Tailored Resume Version:</strong><br>
            <pre style="white-space:pre-wrap;font-family:Menlo,Consolas,monospace;font-size:13px;max-height:400px;overflow:auto;">
{item.get('resume_version','')}
            </pre>
            <strong>Full Cover Letter:</strong><br>
            <pre style="white-space:pre-wrap;font-family:Menlo,Consolas,monospace;font-size:13px;max-height:300px;overflow:auto;">
{item.get('cover_letter_full','')}
            </pre>
            <strong>Email Subject (if direct outreach):</strong> {item.get('email_subject','')}
          </td>
        </tr>
        """)

    sent_date = formatdate(localtime=True)
    return f"""
    <html>
      <body style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;">
        <h2>Executive Job Opportunity Radar – Shortlist & Application Pack</h2>
        <p style="color:#555;margin-top:-6px;">Generated {sent_date}</p>
        <p>
          Below are short-listed senior roles based on your base CV and target criteria,
          each with a tweaked resume version and full cover letter ready for review.
        </p>
        <table width="100%" cellspacing="0" cellpadding="0">
          {''.join(rows)}
        </table>
        <p style="margin-top:24px;font-size:13px;color:#888;">
          Workflow suggestion: for any role you like, copy the tailored resume into your CV doc,
          paste the cover letter into the application or email, tweak tone if needed, then submit.
        </p>
      </body>
    </html>
    """

# ===== SEND EMAIL =====

def send_email(subject, html_body):
    if not SENDGRID_API_KEY:
        raise RuntimeError("SENDGRID_API_KEY missing")
    if not FROM_EMAIL:
        raise RuntimeError("JOB_FROM_EMAIL (verified sender) missing")
    if not RECIPIENT_EMAIL:
        raise RuntimeError("JOB_TO_EMAIL (recipient) missing")

    msg = Mail(
        from_email=FROM_EMAIL,
        to_emails=[e.strip() for e in RECIPIENT_EMAIL.split(",") if e.strip()],
        subject=subject,
        html_content=html_body
    )
    sg = SendGridAPIClient(SENDGRID_API_KEY)
    resp = sg.send(msg)
    print(f"SENDGRID_STATUS {resp.status_code} to={RECIPIENT_EMAIL}")

# ===== MAIN =====

def main():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing")

    base_resume_text = load_base_resume()

    jobs = fetch_recent_jobs(JOB_FEEDS)
    filtered_jobs = filter_jobs_for_profile(jobs)

    enriched = enrich_jobs_with_ai(filtered_jobs, base_resume_text)
    html = build_email_html(enriched)

    send_email("Marmik | Executive Job Opportunity Radar – Daily Shortlist", html)

if __name__ == "__main__":
    main()

