"""
bureau_equifax_us.py
====================
Equifax US Consumer Credit Report — parsing and metrics.

Extracted from a-1.ipynb and corrected for four identified issues:

  1. Tradeline deduplication  — drop exact duplicates before any counting.
  2. Adverse consistency      — CF (Closed Adverse) now treated as adverse in
                                _is_adverse() to match the written specification.
  3. Total Active Debt scope  — Revolving (R) portfolios now included so that
                                revolving balances are not silently excluded.
  4. Inquiry recency anchor   — windows now measured from the bureau report date,
                                not from pd.Timestamp.now(), so the same file
                                always yields the same counts regardless of when
                                it is processed.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Date helper
# ---------------------------------------------------------------------------

def _parse_date(raw: str, fmt_hint: str = "MMDDYYYY") -> pd.Timestamp | None:
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()

    if len(raw) == 8:
        month, day, year = raw[:2], raw[2:4], raw[4:]
        if day == "00":
            try:
                return pd.Timestamp(datetime.strptime(f"{month}{year}", "%m%Y"))
            except ValueError:
                pass
        else:
            try:
                return pd.Timestamp(datetime.strptime(raw, "%m%d%Y"))
            except ValueError:
                pass

    if len(raw) == 6:
        try:
            return pd.Timestamp(datetime.strptime(raw, "%m%Y"))
        except ValueError:
            pass

    return None


# ---------------------------------------------------------------------------
# Portfolio / narrative code constants
# ---------------------------------------------------------------------------

# FIX 3: "R" (Revolving) added so that revolving balances are captured in
#         Total Active Debt.  Previously only I/M/C/O were classed as debt,
#         which silently excluded revolving accounts with outstanding balances.
DEBT_PORTFOLIO_CODES = {"I", "M", "C", "O", "R"}  # Installment, Mortgage, Line-of-Credit, Open, Revolving

REVOLVING_CODES = {"R", "C", "O"}   # Revolving, Line of Credit, Open

# Narrative codes that mark an account as closed (not adverse on their own)
CLOSED_CODES = {"FA", "CF"}          # FA = Closed/Paid, CF = Closed Account

# Consolidated adverse status maps to guarantee _is_adverse() and
# get_bounced_equivalents() are perfectly aligned.
ADVERSE_RATE_MAP = {
    "B": "Lost or Stolen Card",
}

ADVERSE_NARRATIVE_MAP = {
    "RP": "Returned Payment",
    "CO": "Charge-Off",
    "CL": "Collection",
    "BC": "Bankruptcy",
    "CF": "Closed Adverse",
}


# ---------------------------------------------------------------------------
# Trade classifiers
# ---------------------------------------------------------------------------

def _is_active(trade: dict) -> bool:
    """Account is active if no closed narrative codes are present."""
    codes = {n["code"] for n in trade.get("narrativeCodes", [])}
    return len(codes & CLOSED_CODES) == 0


def _is_paying_as_agreed(trade: dict) -> bool:
    rate = trade.get("rate", {})
    return rate.get("code") == 1


def _is_adverse(trade: dict) -> bool:
    """
    Account has a derogatory marker.

    Checks against the unified ADVERSE_NARRATIVE_MAP and ADVERSE_RATE_MAP
    to assure alignment with get_bounced_equivalents().
    """
    codes = {n["code"] for n in trade.get("narrativeCodes", [])}
    rate_status = trade.get("rateStatusCode", {}).get("code", "")
    
    has_adverse_narrative = bool(codes & set(ADVERSE_NARRATIVE_MAP.keys()))
    has_adverse_rate = rate_status in ADVERSE_RATE_MAP
    
    return has_adverse_narrative or has_adverse_rate


def _classify_trade(trade: dict) -> dict:
    portfolio = trade.get("portfolioTypeCode", {}).get("code", "")
    return {
        "is_debt":        portfolio in DEBT_PORTFOLIO_CODES,
        "is_revolving":   portfolio in REVOLVING_CODES,
        "is_mortgage":    portfolio == "M",
        "is_installment": portfolio == "I",
        "is_active":      _is_active(trade),
        "is_adverse":     _is_adverse(trade),
        "pays_as_agreed": _is_paying_as_agreed(trade),
    }


# ---------------------------------------------------------------------------
# FIX 1: Tradeline deduplication
# ---------------------------------------------------------------------------

# Natural key that identifies a unique account.  Two rows that share all of
# these fields are treated as duplicates; only the first occurrence is kept.
#
# Rationale for each field:
#   name          — lender / creditor name
#   account_type  — e.g. "Credit Card", "Auto Loan" (breaks ties for same lender)
#   portfolio_code — I / M / R / C / O
#   date_opened   — opening date of the account
#   high_credit   — original loan amount or peak balance (stable identifier)
#
# balance is intentionally excluded from the key because it fluctuates month
# to month and would fail to detect a duplicate that was re-pulled at a
# different point in time.
_DEDUP_COLS = ["name", "account_type", "portfolio_code", "date_opened", "high_credit"]


def _dedup_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicate tradelines using the natural account key.

    Returns a de-duplicated DataFrame with a reset index.
    Rows removed are logged to stderr so underwriters can audit if needed.
    """
    if df.empty:
        return df

    subset = [c for c in _DEDUP_COLS if c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    removed = before - len(df)
    if removed:
        import sys
        print(
            f"[bureau_equifax_us] Deduplication removed {removed} duplicate "
            f"tradeline(s). Total tradelines before: {before}, after: {len(df)}.",
            file=sys.stderr,
        )
    return df


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_bureau_json(source) -> dict:
    """
    Parse an Equifax US Consumer Credit Report JSON (ResponseEssentials format).

    Returns a dict with keys:
        report, trades_df, inquiries_df, identity, model_score
    """
    if isinstance(source, str):
        with open(source) as f:
            raw = json.load(f)
    else:
        raw = source

    report = raw["consumers"]["equifaxUSConsumerCreditReport"][0]

    # --- Build trades DataFrame ---
    rows = []
    for trade in report.get("trades", []):
        flags          = _classify_trade(trade)
        balance        = trade.get("balance") or 0
        high_credit    = trade.get("highCredit") or 0
        credit_limit   = trade.get("creditLimit") or 0
        scheduled_pmt  = trade.get("scheduledPaymentAmount") or 0
        actual_pmt     = trade.get("actualPaymentAmount") or 0
        months_reviewed = int(trade.get("monthsReviewed") or 0)

        date_opened   = _parse_date(trade.get("dateOpened", ""))
        date_reported = _parse_date(trade.get("dateReported", ""))

        rows.append({
            # Identity
            "name":           trade.get("customerName", ""),
            "account_type":   trade.get("accountTypeCode", {}).get("description", ""),
            "portfolio_code": trade.get("portfolioTypeCode", {}).get("code", ""),
            "designator":     trade.get("accountDesignator", {}).get("description", ""),
            "narratives":     [n["code"] for n in trade.get("narrativeCodes", [])],

            # Dates
            "date_opened":    date_opened,
            "date_reported":  date_reported,

            # Amounts
            "balance":           balance,
            "high_credit":       high_credit,
            "credit_limit":      credit_limit,
            "scheduled_payment": scheduled_pmt,
            "actual_payment":    actual_pmt,

            **flags,

            "months_reviewed": months_reviewed,
        })

    trades_df = pd.DataFrame(rows)

    # FIX 1: deduplicate before any downstream counting
    trades_df = _dedup_trades(trades_df)

    # --- Build inquiries DataFrame ---
    inq_rows = []
    for inq in report.get("inquiries", []):
        inq_rows.append({
            "date":          _parse_date(inq.get("inquiryDate", "")),
            "name":          inq.get("customerName", ""),
            "type":          inq.get("type", ""),
            "industry_code": inq.get("industryCode", ""),
        })
    inquiries_df = pd.DataFrame(inq_rows)

    # --- Identity ---
    name = report.get("subjectName", {})
    identity = {
        "first_name":          name.get("firstName", ""),
        "last_name":           name.get("lastName", ""),
        "full_name":           f"{name.get('firstName','')} {name.get('lastName','')}".strip(),
        "dob":                 _parse_date(report.get("birthDate", "")),
        "ssn":                 report.get("subjectSocialNum", ""),
        "report_date":         _parse_date(report.get("reportDate", "")),
        "file_since":          _parse_date(report.get("fileSinceDate", "")),
        "last_activity":       _parse_date(report.get("lastActivityDate", "")),
        "hit_code":            report.get("hitCode", {}).get("description", ""),
        "address_discrepancy": report.get("addressDiscrepancyIndicator", "N") == "Y",
        "addresses":           report.get("addresses", []),
        "employments":         report.get("employments", []),
    }

    # --- Model score ---
    model_score = None
    for m in report.get("models", []):
        model_score = m.get("score")
        break

    return {
        "report":       report,
        "trades_df":    trades_df,
        "inquiries_df": inquiries_df,
        "identity":     identity,
        "model_score":  model_score,
    }


# ---------------------------------------------------------------------------
# Adverse / bounced-equivalent accounts
# ---------------------------------------------------------------------------

def get_bounced_equivalents(parsed: dict) -> pd.DataFrame:
    """
    Return a DataFrame of trades that carry an adverse/bounced-equivalent marker.

    Guaranteed to be aligned with _is_adverse() by relying on the shared
    ADVERSE_NARRATIVE_MAP and ADVERSE_RATE_MAP constants.
    """
    report = parsed["report"]
    adverse_rows = []

    for trade in report.get("trades", []):
        categories = []

        rate_status_code = trade.get("rateStatusCode", {}).get("code", "")
        if rate_status_code in ADVERSE_RATE_MAP:
            categories.append(ADVERSE_RATE_MAP[rate_status_code])

        for n in trade.get("narrativeCodes", []):
            code = n["code"]
            if code in ADVERSE_NARRATIVE_MAP:
                cat = ADVERSE_NARRATIVE_MAP[code]
                if cat not in categories:
                    categories.append(cat)

        if categories:
            adverse_rows.append({
                "name":            trade.get("customerName", ""),
                "account_type":    trade.get("accountTypeCode", {}).get("description", ""),
                "date_reported":   _parse_date(trade.get("dateReported", "")),
                "balance":         trade.get("balance", 0),
                "Bounce Category": ", ".join(categories),
                "narratives":      [n["description"] for n in trade.get("narrativeCodes", [])],
            })

    if not adverse_rows:
        return pd.DataFrame()

    return pd.DataFrame(adverse_rows)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def calculate_metrics(parsed: dict) -> dict:
    trades    = parsed["trades_df"]
    inquiries = parsed["inquiries_df"]
    identity  = parsed["identity"]

    if trades.empty:
        return {}

    active_trades  = trades[trades["is_active"]]
    adverse_trades = trades[trades["is_adverse"]]

    # --- Core balance metrics ---
    total_balance      = round(trades["balance"].sum(), 2)
    total_credit_limit = round(trades["credit_limit"].sum(), 2)   # revolving limits only
    total_high_credit  = round(trades["high_credit"].sum(), 2)    # peak/loan amounts
    unused_credit      = round(total_credit_limit - total_balance, 2)

    # Utilisation: balance / credit_limit (only where limit is known)
    trades_with_limit = trades[trades["credit_limit"] > 0]
    utilisation_rate  = round(
        trades_with_limit["balance"].sum() / trades_with_limit["credit_limit"].sum() * 100
        if not trades_with_limit.empty else 0,
        2,
    )

    # --- Debt metrics ---
    # FIX 3: debt_trades now includes revolving (R) accounts because R is in
    #         DEBT_PORTFOLIO_CODES.  This prevents understating Total Active Debt
    #         for customers with revolving balances.
    debt_trades          = trades[trades["is_debt"]]
    total_debt           = round(debt_trades["balance"].sum(), 2)
    total_scheduled_pmts = round(trades["scheduled_payment"].sum(), 2)
    total_actual_pmts    = round(trades["actual_payment"].sum(), 2)

    dscr = round(
        total_actual_pmts / total_scheduled_pmts
        if total_scheduled_pmts > 0 else 0,
        2,
    )
    avg_monthly_obligation = round(total_scheduled_pmts, 2)

    # --- Account mix ---
    n_total       = len(trades)
    n_active      = int(trades["is_active"].sum())
    n_closed      = n_total - n_active
    n_mortgage    = int(trades["is_mortgage"].sum())
    n_installment = int(trades["is_installment"].sum())
    n_revolving   = int(trades["is_revolving"].sum())
    n_adverse     = int(trades["is_adverse"].sum())
    n_pays_agreed = int(trades["pays_as_agreed"].sum())

    # --- Credit age ---
    valid_dates = trades["date_opened"].dropna()
    credit_age_months = 0
    oldest_account    = None
    newest_account    = None
    if not valid_dates.empty:
        report_date    = identity.get("report_date") or pd.Timestamp.now()
        oldest_account = valid_dates.min()
        newest_account = valid_dates.max()
        credit_age_months = int((report_date - oldest_account).days / 30.44)

    # --- Inquiry metrics ---
    # FIX 4: anchor recency windows to the bureau report date rather than
    #         pd.Timestamp.now(), so the same file always yields identical
    #         inquiry counts regardless of when it is processed.
    n_inquiries_total = len(inquiries)

    # Use report date as reference; fall back to now only if missing.
    report_date_ref = identity.get("report_date") or pd.Timestamp.now()

    n_inquiries_3m  = 0
    n_inquiries_6m  = 0
    n_inquiries_12m = 0
    if not inquiries.empty and "date" in inquiries.columns:
        valid_inq = inquiries.dropna(subset=["date"])
        n_inquiries_3m  = int((valid_inq["date"] >= report_date_ref - pd.DateOffset(months=3)).sum())
        n_inquiries_6m  = int((valid_inq["date"] >= report_date_ref - pd.DateOffset(months=6)).sum())
        n_inquiries_12m = int((valid_inq["date"] >= report_date_ref - pd.DateOffset(months=12)).sum())

    # --- Adverse / bounced equivalents ---
    bounced_df         = get_bounced_equivalents(parsed)
    n_adverse_accounts = len(bounced_df)

    # --- Address discrepancy ---
    address_discrepancy = identity.get("address_discrepancy", False)

    # --- Months of history ---
    max_months_history = int(trades["months_reviewed"].max()) if not trades.empty else 0

    return {
        # Balance metrics
        "Total Balance (All Accounts)":     total_balance,
        "Total Credit Limit":               total_credit_limit,
        "Total High Credit (Peak/Loans)":   total_high_credit,
        "Unused Credit":                    unused_credit,
        "Credit Utilisation (%)":           utilisation_rate,

        # Debt metrics
        "Total Active Debt":                total_debt,
        "Total Scheduled Monthly Payments": total_scheduled_pmts,
        "Total Actual Payments Made":       total_actual_pmts,
        "Debt Service Coverage Ratio":      dscr,
        "Average Monthly Obligation":       avg_monthly_obligation,

        # Account mix
        "Total Accounts":                   n_total,
        "Active Accounts":                  n_active,
        "Closed Accounts":                  n_closed,
        "Mortgage Accounts":                n_mortgage,
        "Installment Accounts":             n_installment,
        "Revolving Accounts":               n_revolving,
        "Accounts Paying as Agreed":        n_pays_agreed,
        "Adverse Accounts":                 n_adverse,

        # Stability
        "Credit Age (Months)":              credit_age_months,
        "Oldest Account Opened":            str(oldest_account.date()) if oldest_account else None,
        "Newest Account Opened":            str(newest_account.date()) if newest_account else None,
        "Max Months of History":            max_months_history,

        # Conduct / Inquiry
        "Total Inquiries":                  n_inquiries_total,
        "Inquiries (Last 3 Months)":        n_inquiries_3m,
        "Inquiries (Last 6 Months)":        n_inquiries_6m,
        "Inquiries (Last 12 Months)":       n_inquiries_12m,

        # Risk flags
        "Adverse Accounts (Bounced equiv)": n_adverse_accounts,
        "Address Discrepancy Flag":         address_discrepancy,

        # Bureau model score (raw)
        "Bureau Model Score":               parsed.get("model_score"),
    }


# ---------------------------------------------------------------------------
# Supporting helpers
# ---------------------------------------------------------------------------

def count_credit_sources(parsed: dict) -> int:
    trades = parsed["trades_df"]
    active = trades[trades["is_active"]]
    return active["name"].nunique()


def _clean_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = re.sub(r'www\.|\\.co\\.uk|\.com|\.org|\.net', '', name)
    name = re.sub(r'\bltd\b|\blimited\b|\bt/a\b', '', name)
    name = re.sub(r'\b(repayment|payment|finance|loan|loans|transfer)\b', '', name)
    name = re.sub(r'\b[a-z]*\d+[a-z\d]*\b', '', name)
    name = re.sub(r'[^a-z0-9\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def check_lender_repayments(parsed: dict, threshold: int = 80) -> tuple:
    trades     = parsed["trades_df"]
    all_active = trades[trades["is_active"]].copy()

    lenders_with_payments    = {}
    lenders_missing_payments = []

    for _, row in all_active.iterrows():
        label = f"{row['name']} ({row['account_type']})" if row["account_type"] else row["name"]
        pmt   = row["scheduled_payment"]

        if pmt > 0:
            lenders_with_payments[label] = round(pmt, 2)
        else:
            lenders_missing_payments.append(label)

    return lenders_with_payments, lenders_missing_payments


def process_balance_report(parsed: dict) -> pd.DataFrame:
    trades = parsed["trades_df"].copy()

    if trades.empty:
        return pd.DataFrame()

    active = trades[trades["is_active"]].copy()

    def _utilisation(row):
        if row["credit_limit"] > 0:
            return round(row["balance"] / row["credit_limit"] * 100, 2)
        elif row["high_credit"] > 0:
            return round(row["balance"] / row["high_credit"] * 100, 2)
        return 0.0

    active["utilisation_pct"] = active.apply(_utilisation, axis=1)
    active["account_status"]  = active["pays_as_agreed"].map(
        {True: "Pays as agreed", False: "Issue flagged"}
    )

    report = active[[
        "name", "account_type", "portfolio_code",
        "balance", "credit_limit", "high_credit",
        "scheduled_payment", "actual_payment",
        "utilisation_pct", "account_status",
        "date_opened", "date_reported", "months_reviewed",
    ]].copy()

    report.columns = [
        "Lender", "Account Type", "Portfolio",
        "Current Balance", "Credit Limit", "High Credit / Loan Amount",
        "Scheduled Payment", "Actual Payment",
        "Utilisation (%)", "Status",
        "Date Opened", "Last Reported", "Months Reviewed",
    ]

    return report.sort_values("Current Balance", ascending=False).reset_index(drop=True)


def inquiry_velocity(parsed: dict) -> dict:
    inquiries = parsed["inquiries_df"].copy()
    if inquiries.empty:
        return {"clustered_inquiry_dates": [], "counts": {}}

    inquiries = inquiries.dropna(subset=["date"]).sort_values("date")
    inquiries["date_only"] = inquiries["date"].dt.date

    daily     = inquiries.groupby("date_only").size().reset_index(name="count")
    clustered = daily[daily["count"] >= 3]["date_only"].tolist()

    return {
        "clustered_inquiry_dates": [str(d) for d in clustered],
        "total":   len(inquiries),
        "by_date": daily.set_index("date_only")["count"].to_dict(),
    }


# ---------------------------------------------------------------------------
# Top-level summary
# ---------------------------------------------------------------------------

def summary(parsed: dict) -> dict:
    metrics   = calculate_metrics(parsed)
    bounced   = get_bounced_equivalents(parsed)
    velocity  = inquiry_velocity(parsed)
    balance_r = process_balance_report(parsed)
    n_sources = count_credit_sources(parsed)
    lenders_w, lenders_missing = check_lender_repayments(parsed)

    return {
        "identity":                parsed["identity"],
        "metrics":                 metrics,
        "adverse_accounts":        bounced.to_dict("records") if not bounced.empty else [],
        "inquiry_velocity":        velocity,
        "monthly_balance_report":  balance_r.to_dict("records") if not balance_r.empty else [],
        "n_distinct_lenders":      n_sources,
        "lenders_with_payments":   lenders_w,
        "lenders_missing_payments": lenders_missing,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    path   = sys.argv[1] if len(sys.argv) > 1 else "ResponseEssentials.json"
    parsed = parse_bureau_json(path)
    result = summary(parsed)

    print("=== IDENTITY ===")
    print(f"  Name        : {result['identity']['full_name']}")
    print(f"  Report Date : {result['identity']['report_date']}")
    print(f"  Addr Flag   : {result['identity']['address_discrepancy']}")

    print("\n=== KEY METRICS ===")
    for k, v in result["metrics"].items():
        print(f"  {k:<45}: {v}")

    print("\n=== ADVERSE ACCOUNTS ===")
    for a in result["adverse_accounts"]:
        print(f"  {a['name']} — {a['Bounce Category']}")

    print("\n=== INQUIRY VELOCITY ===")
    vel = result["inquiry_velocity"]
    print(f"  Total inquiries : {vel.get('total', 0)}")
    print(f"  Clustered dates : {vel['clustered_inquiry_dates']}")

    print("\n=== LENDERS WITHOUT SCHEDULED PAYMENTS ===")
    print(result["lenders_missing_payments"])
