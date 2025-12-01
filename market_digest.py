import os
from datetime import datetime, timedelta, timezone

import yfinance as yf
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from openai import OpenAI

# ---------------------------------------------------------
# CONFIG – WEEKLY WATCHLIST (BROAD UNIVERSE)
# ---------------------------------------------------------

WATCHLIST = [
    # ASX ETFs
    "VAS.AX",
    "VGS.AX",
    "VGE.AX",
    "NDQ.AX",
    "IVV.AX",
    "STW.AX",

    # AU large caps
    "TLS.AX",
    "CBA.AX",
    "BHP.AX",
    "RIO.AX",
    "WES.AX",
    "CSL.AX",
    "XRO.AX",

    # US / global ETFs & sectors
    "QQQ",
    "SPY",
    "XLK",
    "XLF",
    "XLE",
    "XLU",
    "SMH",
    "VNQ",

    # US big tech
    "NVDA",
    "MSFT",
    "AAPL",
    "AMZN",
    "GOOGL",
    "META",
]

# Thresholds – still useful to flag standouts in the week
MIN_DAILY_MOVE_PCT = 2.0
MIN_WEEKLY_MOVE_PCT = 5.0

MODEL_NAME = "gpt-4.1-mini"


# ---------------------------------------------------------
# DATA FETCHING
# ---------------------------------------------------------

def fetch_market_data(tickers):
    # Look back ~10 days to be safe (captures last week + a bit)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=10)

    results = []

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(start=start, end=end)

            if hist.empty or len(hist) < 2:
                continue

            latest = hist.iloc[-1]
            prev = hist.iloc[-2]

            price = float(latest["Close"])
            prev_price = float(prev["Close"])
            day_change_pct = ((price - prev_price) / prev_price) * 100

            # approx 1-week change – 5 trading days back if available
            if len(hist) >= 6:
                week_price = float(hist.iloc[-6]["Close"])
            else:
                week_price = float(hist.iloc[0]["Close"])
            week_change_pct = ((price - week_price) / week_price) * 100

            info = t.info if hasattr(t, "info") else {}
            name = info.get("shortName") or info.get("longName") or ticker

            results.append({
                "ticker": ticker,
                "name": name,
                "price": round(price, 2),
                "day_change_pct": round(day_change_pct, 2),
                "week_change_pct": round(week_change_pct, 2),
            })
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    return results


def filter_significant_moves(market_data):
    # Still flag big daily OR big weekly movers – for the “highlights” section
    return [
        item for item in market_data
        if abs(item["day_change_pct"]) >= MIN_DAILY_MOVE_PCT
        or abs(item["week_change_pct"]) >= MIN_WEEKLY_MOVE_PCT
    ]


# ---------------------------------------------------------
# AI PROMPT BUILDER – NOW WEEKLY
# ---------------------------------------------------------

def build_ai_prompt(market_data, significant):
    today = datetime.now().strftime("%d %b %Y")

    snapshot_lines = []
    snapshot_lines.append("Week ending: " + today + "\n")
    snapshot_lines.append("WATCHLIST SNAPSHOT (PRICE / DAY / WEEK):")

    for item in market_data:
        snapshot_lines.append(
            "- {ticker} ({name}): Price {price}, Day {day}%, Week {week}%".format(
                ticker=item["ticker"],
                name=item["name"],
                price=item["price"],
                day=item["day_change_pct"],
                week=item["week_change_pct"],
            )
        )

    snapshot_lines.append("")
    snapshot_lines.append("SIGNIFICANT WEEKLY MOVES (FILTERED):")
    if significant:
        for item in significant:
            snapshot_lines.append(
                "- {ticker} ({name}): Day {day}%, Week {week}%".format(
                    ticker=item["ticker"],
                    name=item["name"],
                    day=item["day_change_pct"],
                    week=item["week_change_pct"],
                )
            )
    else:
        snapshot_lines.append("- None")

    snapshot_text = "\n".join(snapshot_lines)

    system_msg = (
        "You are an investment research assistant for a senior marketing and commercial leader. "
        "He thinks like a CIO: in themes, risk, and capital allocation. "
        "This is a WEEKLY briefing — focus on what happened across the last trading week: "
        "leadership vs laggards, rotations between sectors and styles, and where momentum "
        "is building or fading. "
        "Never give buy/sell advice. Use language like: accumulation trend, fatigue, "
        "rotation out of, momentum building, de-risking zone, structural uptrend."
    )

    user_msg = (
        "Below is the snapshot of the full watchlist with daily and weekly % moves.\n\n"
        + snapshot_text
        + "\n\n"
        "Write a WEEKLY CIO-style market summary using this structure:\n\n"
        "1. Weekly Market Mood (5-day tone)\n"
        "   - Summarise whether the week felt risk-on, risk-off, or mixed.\n\n"
        "2. Leadership & Laggards (Week-on-week)\n"
        "   - Call out which indices, sectors, and names led or lagged over the week.\n\n"
        "3. Rotations & Style Shifts\n"
        "   - Describe any rotations between growth vs value, defensives vs cyclicals, "
        "     tech vs energy, domestic vs global, etc.\n\n"
        "4. Structural & Thematic Moves\n"
        "   - Highlight any structural themes (AI, energy, yields, property, banks) "
        "     where the week reinforced or challenged the existing narrative.\n\n"
        "5. De-risking & Accumulation Zones\n"
        "   - Identify areas showing de-risking behaviour vs quiet accumulation, "
        "     using directional language only (no advice).\n\n"
        "6. Watchlist Highlights for Next Week\n"
        "   - 3–5 bullets on what is worth watching into the coming week based on "
        "     this week's moves.\n\n"
        "Rules:\n"
        "- This is a WEEKLY note, not daily noise.\n"
        "- Do NOT give explicit recommendations or portfolio advice.\n"
        "- Keep it sharp, mobile-friendly, and written like an internal CIO wrap.\n"
    )

    return system_msg, user_msg


# ---------------------------------------------------------
# OPENAI + SENDGRID
# ---------------------------------------------------------

def generate_email(system_msg, user_msg):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content


def send_email(subject, body):
    sg = SendGridAPIClient(os.environ["SENDGRID_API_KEY"])
    message = Mail(
        from_email=os.environ["MARKET_DIGEST_FROM"],
        to_emails=os.environ["MARKET_DIGEST_TO"],
        subject=subject,
        html_content=body.replace("\n", "<br>"),
    )
    sg.send(message)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    print("Fetching market data...")
    data = fetch_market_data(WATCHLIST)

    print("Filtering significant moves...")
    sig = filter_significant_moves(data)

    print("Building AI prompt...")
    system_msg, user_msg = build_ai_prompt(data, sig)

    print("Generating email...")
    body = generate_email(system_msg, user_msg)

    subject = "Weekly Market Scan – CIO Pulse (week ending {})".format(
        datetime.now().strftime("%d %b %Y")
    )

    print("Sending email...")
    send_email(subject, body)

    print("Done.")


if __name__ == "__main__":
    main()
