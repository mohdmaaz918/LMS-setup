"""
Chirp US Open Banking — Streamlit app mirroring the UK HCSTC Streamlit layout
(Upload & Process, Results Dashboard, Help) while using Chirp JSON + chirp_plaid_bridge.

Does not modify the UK app (app.py at repo root).

Run from repository root:
    streamlit run Chirp/app.py
"""

from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_CHIRP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CHIRP_DIR.parent
if str(_CHIRP_DIR) not in sys.path:
    sys.path.insert(0, str(_CHIRP_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chirp_engine_runner import (  # noqa: E402
    rows_to_dataframe,
    run_chirp_scoring_pipeline,
    scoring_result_to_row,
)
def load_files_from_uploads(uploaded_files) -> List[Tuple[str, bytes]]:
    extracted = []
    for f in uploaded_files:
        if f.name.endswith(".zip"):
            with zipfile.ZipFile(f) as z:
                for name in z.namelist():
                    if name.endswith(".json") and not name.startswith("__MACOSX"):
                        extracted.append((name, z.read(name)))
        else:
            extracted.append((f.name, f.read()))
    return extracted

from chirp_plaid_bridge import (  # noqa: E402
    chirp_json_to_engine_transactions,
    normalize_chirp_payload,
)

from scoring_config import PRODUCT_CONFIG, SCORING_CONFIG  # noqa: E402


def _init_session() -> None:
    defaults = {
        "chirp_cumulative_mode": False,
        "chirp_batch_count": 0,
        "chirp_processed_filenames": set(),
        "chirp_results_df": None,
        "chirp_errors_df": None,
        "chirp_results_details": [],
        "chirp_last_processing_time": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _approve_threshold() -> int:
    return int(SCORING_CONFIG["score_ranges"]["approve"]["min"])


def _refer_threshold() -> int:
    return int(SCORING_CONFIG["score_ranges"]["refer"]["min"])


def _process_chirp_file(
    filename: str,
    content: bytes,
    loan_amount: float,
    loan_term: int,
    days_covered: int,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Returns (result_row, result_detail_dict) or (None, None) on failure.
    result_detail = {"filename", "result": full API dict}
    """
    try:
        raw = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"Invalid JSON: {e}") from e

    try:
        data = normalize_chirp_payload(raw)
    except ValueError as e:
        raise ValueError(str(e)) from e

    engine_tx = chirp_json_to_engine_transactions(data)
    if not engine_tx:
        raise ValueError("No transactions after Chirp→engine mapping")

    result_dict, scoring_result = run_chirp_scoring_pipeline(
        engine_tx,
        requested_amount=loan_amount,
        requested_term=loan_term,
        days_covered=days_covered,
    )
    row = scoring_result_to_row(scoring_result, application_ref=filename)
    detail = {"filename": filename, "result": result_dict}
    return row, detail


def _make_custom_card(label: str, value: str, delta: str = "", theme: str = "blue", icon: str = "📊") -> str:
    delta_html = f'<div class="card-subtitle">{delta}</div>' if delta else ''
    return f"""
    <div class="custom-card card-{theme}">
        <div class="card-title"><span>{label}</span><span style="font-size:1.5rem">{icon}</span></div>
        <div class="card-value">{value}</div>
        {delta_html}
    </div>
    """

def _display_processing_summary(
    results_df: pd.DataFrame,
    errors_df: Optional[pd.DataFrame],
    processing_time: float,
) -> None:
    st.markdown('<p class="main-subheader" style="margin-bottom:0.5rem">📈 Processing Summary</p>', unsafe_allow_html=True)
    n_ok = len(results_df)
    n_err = len(errors_df) if errors_df is not None and not errors_df.empty else 0
    total = n_ok + n_err
    avg_score = float(results_df["Score"].mean()) if n_ok else 0.0
    success_rate = (n_ok / total * 100) if total else 0.0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(_make_custom_card("Total Files", str(total), "", "purple", "📁"), unsafe_allow_html=True)
    with col2:
        st.markdown(_make_custom_card("Successful", str(n_ok), f"{success_rate:.1f}% rate", "green", "✅"), unsafe_allow_html=True)
    with col3:
        st.markdown(_make_custom_card("Failed", str(n_err), "", "red", "❌"), unsafe_allow_html=True)
    with col4:
        st.markdown(_make_custom_card("Avg Score", f"{avg_score:.1f}", "", "blue", "🎯"), unsafe_allow_html=True)
    with col5:
        st.markdown(_make_custom_card("Processing", f"{processing_time:.1f}s", "Time elapsed", "orange", "⏱️"), unsafe_allow_html=True)

    st.markdown('<p class="main-subheader" style="margin-top:1.5rem; margin-bottom:0.5rem">📊 Decision Breakdown</p>', unsafe_allow_html=True)
    if n_ok:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(_make_custom_card("APPROVED", str(int((results_df["Decision"] == "APPROVE").sum())), "Ready for funding", "green", "🏦"), unsafe_allow_html=True)
        with c2:
            st.markdown(_make_custom_card("REFER", str(int((results_df["Decision"] == "REFER").sum())), "Requires manual review", "orange", "📋"), unsafe_allow_html=True)
        with c3:
            st.markdown(_make_custom_card("DECLINED", str(int((results_df["Decision"] == "DECLINE").sum())), "Did not pass rules", "red", "⛔"), unsafe_allow_html=True)

    if errors_df is not None and not errors_df.empty:
        with st.expander("⚠️ Error Summary", expanded=False):
            for _, er in errors_df.iterrows():
                st.text(f"• {er.get('file_name', '?')}: {er.get('error_message', '')}")


def _render_categorised_transactions_from_session() -> None:
    """
    Show per-transaction categorisation for each processed application (Upload & process tab).
    Uses chirp_results_details populated after a successful run.
    """
    details = st.session_state.get("chirp_results_details") or []
    if not details:
        return

    st.divider()
    st.subheader("📑 Categorised transactions")
    st.caption(
        "Each transaction from the uploaded file(s), after the Chirp→engine bridge, with engine "
        "**category**, **subcategory**, **confidence**, and **match method**. "
        "Match method uses a **chirp_** prefix (for example `chirp_mapped_strict`): Chirp categories "
        "were mapped into the engine’s taxonomy (not live Plaid data)."
    )
    names = [d["filename"] for d in details]
    if len(names) == 1:
        pick = names[0]
        st.markdown(f"**File:** `{pick}`")
    else:
        pick = st.selectbox(
            "Application / file",
            options=names,
            key="chirp_upload_tab_txn_file",
        )
    chosen = next((d for d in details if d["filename"] == pick), None)
    if not chosen:
        return
    res = chosen.get("result") or {}
    cat_rows = res.get("categorized_transactions") or []
    if not cat_rows:
        st.info("No categorised transaction rows for this file.")
        return

    df = pd.DataFrame(cat_rows)
    preferred = [
        "date",
        "amount",
        "description",
        "category",
        "subcategory",
        "confidence",
        "match_method",
        "weight",
        "is_stable",
        "is_housing",
        "risk_level",
    ]
    ordered = [c for c in preferred if c in df.columns]
    ordered += [c for c in df.columns if c not in ordered]
    df = df[ordered]

    n = len(df)
    show = df.head(500)
    st.dataframe(show, use_container_width=True, hide_index=True)
    if n > 500:
        st.caption(f"Showing first 500 of {n} rows.")

    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    safe_fn = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in pick)[:80]
    st.download_button(
        label="📥 Download categorised transactions (CSV)",
        data=csv_buf.getvalue(),
        file_name=f"categorised_{safe_fn}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key="chirp_dl_categorised_upload_tab",
    )


def render_upload_tab(
    loan_amount: float,
    loan_term: int,
    days_covered: int,
    use_auto_months: bool,
    months_override: Optional[int],
) -> None:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 2rem; border-radius: 16px; color: white; margin-bottom: 1.5rem; box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.3); border: 1px solid rgba(255,255,255,0.1);">
            <h2 style="margin-top: 0; font-size: 2rem; font-weight: 700; display: flex; align-items: center; gap: 12px; color: white;">
                <span style="font-size: 2.5rem;">📤</span> Data Processing Engine
            </h2>
            <p style="font-size: 1.1rem; opacity: 0.8; margin-bottom: 0; font-weight: 300;">
                Securely drop your <b>Chirp Open Banking JSON files</b> or bulk ZIP archives here. Our system will map Plaid structures to our taxonomy and execute live decisioning protocols instantly.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🔄 Batch processing mode")
        cumulative_mode = st.checkbox(
            "**Add to results** (cumulative mode)",
            value=st.session_state.get("chirp_cumulative_mode", False),
            help="Append new runs to existing results instead of replacing.",
        )
        st.session_state["chirp_cumulative_mode"] = cumulative_mode
        if cumulative_mode:
            st.info(
                "🔄 **Cumulative mode** — new uploads are merged. Use **Clear all results** to reset."
            )
        else:
            st.info("🔁 **Replace mode** — each run replaces previous results.")

    with col2:
        st.subheader("🗑️ Clear data")
        if st.button("Clear all results", type="secondary", use_container_width=True):
            st.session_state["chirp_results_df"] = None
            st.session_state["chirp_errors_df"] = None
            st.session_state["chirp_batch_count"] = 0
            st.session_state["chirp_processed_filenames"] = set()
            st.session_state["chirp_results_details"] = []
            st.success("Cleared.")
            st.rerun()

    if cumulative_mode and st.session_state.get("chirp_results_df") is not None:
        df = st.session_state["chirp_results_df"]
        bc = st.session_state.get("chirp_batch_count", 0)
        st.subheader("📊 Cumulative progress")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Batches processed", bc)
        c2.metric("Total rows", len(df))
        c3.metric("Applications OK", len(df))
        err = st.session_state.get("chirp_errors_df")
        n_err = len(err) if err is not None and not err.empty else 0
        c4.metric("Total errors", n_err)

    st.divider()
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["json", "zip"],
        accept_multiple_files=True,
        help="Chirp JSON with TransactionSummaries, or ZIP of JSON files",
    )

    if not uploaded_files:
        return

    st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
    with st.expander("📁 Uploaded files", expanded=False):
        for f in uploaded_files:
            st.text(f"• {f.name} ({f.size / 1024:.1f} KB)")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_button = st.button(
            "🚀 Process applications",
            type="primary",
            use_container_width=True,
        )

    if not process_button:
        return

    t0 = time.time()
    files = load_files_from_uploads(uploaded_files)
    if not files:
        st.error("No valid JSON files found in uploads.")
        return

    cumulative_mode = st.session_state.get("chirp_cumulative_mode", False)
    processed = set(st.session_state.get("chirp_processed_filenames", set()))

    files_to_run = []
    duplicates = set()
    if cumulative_mode and processed:
        for fn, content in files:
            if fn in processed:
                duplicates.add(fn)
            else:
                files_to_run.append((fn, content))
        if duplicates:
            st.info(
                f"⏭️ Skipping {len(duplicates)} already-processed file(s). "
                f"Processing {len(files_to_run)} new file(s)."
            )
    else:
        files_to_run = list(files)

    if not files_to_run:
        st.warning("All files were already processed (cumulative mode).")
        return

    # months: bridge ignores days_covered in engine; keep UI parity with UK app
    del use_auto_months, months_override  # reserved for future use

    rows: List[Dict] = []
    errors: List[Dict] = []
    details: List[Dict] = []

    progress = st.progress(0)
    status = st.empty()
    n_total = len(files_to_run)

    for i, (filename, content) in enumerate(files_to_run):
        progress.progress((i + 1) / n_total)
        status.text(f"[{i + 1}/{n_total}] {filename}")
        try:
            row, detail = _process_chirp_file(
                filename, content, loan_amount, loan_term, days_covered
            )
            if row:
                rows.append(row)
            if detail:
                details.append(detail)
            processed.add(filename)
        except Exception as e:
            errors.append(
                {
                    "file_name": filename,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            )

    elapsed = time.time() - t0
    progress.progress(1.0)
    status.text(f"✅ Done in {elapsed:.1f}s")

    new_df = rows_to_dataframe(rows)
    err_df = pd.DataFrame(errors) if errors else None

    current_batch = st.session_state.get("chirp_batch_count", 0) + 1

    if cumulative_mode and st.session_state.get("chirp_results_df") is not None:
        old_df = st.session_state["chirp_results_df"]
        if old_df is None or old_df.empty:
            combined = new_df
        elif new_df.empty:
            combined = old_df
        else:
            combined = pd.concat([old_df, new_df], ignore_index=True)

        st.session_state["chirp_results_df"] = combined
        st.session_state["chirp_results_details"] = (
            st.session_state.get("chirp_results_details", []) + details
        )
        st.session_state["chirp_batch_count"] = current_batch
        st.session_state["chirp_processed_filenames"] = processed

        if err_df is not None and not err_df.empty:
            old_err = st.session_state.get("chirp_errors_df")
            if old_err is not None and not old_err.empty:
                st.session_state["chirp_errors_df"] = pd.concat(
                    [old_err, err_df], ignore_index=True
                )
            else:
                st.session_state["chirp_errors_df"] = err_df

        st.subheader(f"📈 Batch {current_batch} results")
        _display_processing_summary(new_df, err_df, elapsed)
        st.divider()
        st.subheader("📊 Combined results (all batches)")
        _display_processing_summary(
            combined, st.session_state.get("chirp_errors_df"), elapsed
        )
    else:
        st.session_state["chirp_results_df"] = new_df
        st.session_state["chirp_errors_df"] = err_df
        st.session_state["chirp_batch_count"] = 1
        st.session_state["chirp_processed_filenames"] = processed
        st.session_state["chirp_results_details"] = details
        _display_processing_summary(new_df, err_df, elapsed)

    st.session_state["chirp_last_processing_time"] = elapsed
    st.success("✨ Processing complete! Scroll down to view the dashboard.")

def _risk_flag_parts(flags_str: str) -> List[str]:
    if not flags_str or not str(flags_str).strip():
        return []
    out = []
    for part in str(flags_str).split(";"):
        part = part.strip()
        if not part:
            continue
        out.append(part.split(":")[0].strip() if ":" in part else part)
    return out


def _fig_monthly_flow(categorized: List[Dict]) -> Optional[go.Figure]:
    if not categorized:
        return None
    buckets: Dict[str, float] = {}
    for r in categorized:
        d = r.get("date")
        if not d:
            continue
        month = str(d)[:7]
        amt = float(r.get("amount") or 0)
        buckets[month] = buckets.get(month, 0.0) + abs(amt)
    if not buckets:
        return None
    months = sorted(buckets.keys())
    fig = go.Figure(
        data=[
            go.Bar(x=months, y=[buckets[m] for m in months], name="Total |amount|"),
        ]
    )
    fig.update_layout(
        title="Monthly volume (sum of absolute amounts)",
        xaxis_title="Month",
        yaxis_title="USD",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def _fig_category_pie(categorized: List[Dict]) -> Optional[go.Figure]:
    if not categorized:
        return None
    counts: Dict[str, int] = {}
    for r in categorized:
        cat = r.get("category") or "?"
        sub = r.get("subcategory") or ""
        key = f"{cat}/{sub}" if sub else str(cat)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    df = pd.DataFrame({"label": list(counts.keys()), "n": list(counts.values())})
    fig = px.pie(df, values="n", names="label", title="Categorised transactions (count by engine bucket)")
    fig.update_layout(
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def _render_individual_dashboard(row: pd.Series, details: List[Dict]) -> None:
    ref = row["Application Ref"]
    decision = row["Decision"]
    score = row["Score"]
    risk = row["Risk Level"]
    
    st.markdown(f"### Deep-Dive: {ref}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Decision", decision)
    col2.metric("Score", score)
    col3.metric("Risk Level", risk)
    if "Affordability Score" in row.index:
        col4.metric("Affordability Score", f"{row['Affordability Score']}/30")
        
    st.subheader("📊 Key underwriting metrics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(_make_custom_card("Gross Income", f"${row.get('Monthly Income', 0):,.2f}", "Monthly average", "green", "💵"), unsafe_allow_html=True)
    with m2:
        st.markdown(_make_custom_card("Total Expenses", f"${row.get('Monthly Expenses', 0):,.2f}", "Monthly average", "red", "📉"), unsafe_allow_html=True)
    with m3:
        st.markdown(_make_custom_card("Disposable", f"${row.get('Monthly Disposable', 0):,.2f}", "Pre-loan", "blue", "💳"), unsafe_allow_html=True)
    with m4:
        st.markdown(_make_custom_card("Proposed Repay", f"${row.get('Monthly Repayment', 0):,.2f}", f"Term: {row.get('Approved Term', 0)}m", "purple", "📅"), unsafe_allow_html=True)
        
    st.markdown("<br/>", unsafe_allow_html=True)
    
    m5, m6, m7, m8 = st.columns(4)
    with m5:
        post_disp = row.get("Post-Loan Disposable", 0)
        theme_post = "orange" if post_disp < 0 else "green"
        st.markdown(_make_custom_card("Post-Loan Disp.", f"${post_disp:,.2f}", "Estimated", theme_post, "💰"), unsafe_allow_html=True)
    with m6:
        st.markdown(_make_custom_card("Total Repayable", f"${row.get('Total Repayable', 0):,.2f}", "Interest + Principal", "purple", "🏦"), unsafe_allow_html=True)
    with m7:
        od_days = row.get("Overdraft Days per Month", 0)
        st.markdown(_make_custom_card("Overdraft Activity", f"{od_days:.1f} days/mo", "Historical avg", "red" if float(od_days) > 3 else "blue", "⚠️"), unsafe_allow_html=True)
    with m8:
        st.markdown(_make_custom_card("Data History", f"{row.get('Months Observed', 0):.1f} mos", "Statement depth", "blue", "📆"), unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    st.subheader("💰 Affordability summary")
    c1, c2 = st.columns(2)
    with c1:
        inc = row.get("Monthly Income", 0)
        exp = row.get("Monthly Expenses", 0)
        rep = row.get("Monthly Repayment", 0)
        disp = row.get("Monthly Disposable", 0)
        post = row.get("Post-Loan Disposable", 0)
        
        fig_cashflow = go.Figure(go.Waterfall(
            name="Cashflow",
            orientation="v",
            measure=["relative", "relative", "total", "relative", "total"],
            x=["Income", "Expenses", "Pre-Loan Disp.", "New Loan Repay", "Post-Loan Disp."],
            textposition="outside",
            text=[f"${v:,.0f}" if i != 2 and i != 4 else f"<b>${v:,.0f}</b>" for i, v in enumerate([inc, -exp, disp, -rep, post])],
            y=[inc, -exp, disp, -rep, post],
            connector={"line":{"color":"rgb(63, 63, 63)"}},
            decreasing={"marker":{"color":"#ef4444"}},
            increasing={"marker":{"color":"#10b981"}},
            totals={"marker":{"color":"#3b82f6"}}
        ))
        fig_cashflow.update_layout(
            title="Monthly Cashflow Breakdown",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_cashflow, use_container_width=True)

    with c2:
        cols = [
            "Affordability Score",
            "Income Quality Score",
            "Account Conduct Score",
            "Risk Indicators Score",
        ]
        if all(c in row.index for c in cols):
            scores = [row[c] for c in cols]
            max_scores = [30, 35, 25, 10]
            
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=scores,
                y=cols,
                name="Applicant Score",
                orientation='h',
                marker=dict(color='#4f46e5'),
                text=[f"{s}/{m}" for s, m in zip(scores, max_scores)],
                textposition='inside'
            ))
            fig_bar.add_trace(go.Bar(
                x=[m - s for m, s in zip(max_scores, scores)],
                y=cols,
                name="Available Points",
                orientation='h',
                marker=dict(color='#e2e8f0'),
                hoverinfo='skip'
            ))
            fig_bar.update_layout(
                barmode='stack',
                title="Score Breakdown (Points Achieved vs Maximum)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=50, b=20),
                showlegend=False,
                xaxis=dict(range=[0, 40], visible=False),
                yaxis={'categoryorder':'array', 'categoryarray': cols[::-1]}
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("⚠️ Risk flags")
    risk_flags = _risk_flag_parts(str(row.get("Risk Flags", "")))
    if risk_flags:
        for f in risk_flags:
            st.warning(f)
    else:
        st.success("No critical risk flags detected.")

    st.subheader("🏷️ Categorisation & cashflow")
    chosen = next((d for d in details if d["filename"] == ref), None)
    if chosen:
        res = chosen.get("result") or {}
        cat_rows = res.get("categorized_transactions") or []
        fc1, fc2 = st.columns(2)
        with fc1:
            fig_cat = _fig_category_pie(cat_rows)
            if fig_cat: st.plotly_chart(fig_cat, use_container_width=True)
            else: st.caption("No categorised rows.")
        with fc2:
            fig_m = _fig_monthly_flow(cat_rows)
            if fig_m: st.plotly_chart(fig_m, use_container_width=True)
            else: st.caption("Could not build monthly chart.")
            
        if cat_rows:
            st.markdown("#### 📄 Transaction data")
            df = pd.DataFrame(cat_rows)
            preferred = ["date", "amount", "description", "category", "subcategory", "confidence", "match_method", "weight", "is_stable", "is_housing", "risk_level"]
            ordered = [c for c in preferred if c in df.columns]
            ordered += [c for c in df.columns if c not in ordered]
            df = df[ordered]
            
            show = df.head(500)
            st.dataframe(show, use_container_width=True, hide_index=True)
            if len(df) > 500:
                st.caption(f"Showing first 500 of {len(df)} rows.")
                
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            safe_fn = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in ref)[:80]
            st.download_button(
                label=f"📥 Download {safe_fn} transactions (CSV)",
                data=csv_buf.getvalue(),
                file_name=f"categorised_{safe_fn}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key=f"dl_txns_{ref}",
            )
            
    st.divider()
    st.markdown("### 📝 Underwriter Decision")
    comments = st.text_area("Underwriting Comments & Rationale", key=f"comment_{ref}")
    bc1, bc2 = st.columns(2)
    with bc1:
        if st.button("✅ Accept Application", key=f"btn_acc_{ref}", use_container_width=True):
            st.success(f"Application {ref} marked as ACCEPTED.")
    with bc2:
        if st.button("❌ Decline Application", key=f"btn_dec_{ref}", use_container_width=True, type="primary"):
            st.error(f"Application {ref} marked as DECLINED.")


def render_results_tab() -> None:
    st.markdown(
        """
        <div style="padding: 1rem 0; margin-bottom: 2rem; border-bottom: 2px solid #e2e8f0;">
            <h2 style="margin-top: 0; font-size: 2.2rem; font-weight: 700; color: #0f172a;">
                📊 Batch Processing Overview & Triage
            </h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    results_df = st.session_state.get("chirp_results_df")
    errors_df = st.session_state.get("chirp_errors_df")
    details: List[Dict] = st.session_state.get("chirp_results_details") or []

    if results_df is None or results_df.empty:
        return

    num_apps = len(results_df)
    decisions = results_df["Decision"].value_counts().to_dict()
    approved = decisions.get("APPROVE", 0)
    referred = decisions.get("REFER", 0)
    declined = decisions.get("DECLINE", 0)
    avg_aff_score = results_df["Affordability Score"].mean() if "Affordability Score" in results_df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="custom-card card-blue"><div class="card-title">Total Processed</div><div class="card-value">{num_apps}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="custom-card card-green"><div class="card-title">Approved</div><div class="card-value">{approved}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="custom-card card-orange"><div class="card-title">Referred</div><div class="card-value">{referred}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="custom-card card-red"><div class="card-title">Declined</div><div class="card-value">{declined}</div></div>', unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        st.markdown(f'<div class="custom-card card-purple"><div class="card-title">Avg Affordability Score</div><div class="card-value">{avg_aff_score:.1f}/30</div></div>', unsafe_allow_html=True)
        
    st.divider()
    st.subheader("🕵️‍♂️ Applicant Triage")
    
    tabs = st.tabs(["🟢 Approved", "🟡 Referred", "🔴 Declined"])
    for tab, outcome in zip(tabs, ["APPROVE", "REFER", "DECLINE"]):
        with tab:
            subset = results_df[results_df["Decision"] == outcome]
            if subset.empty:
                st.info(f"No applications heavily flagged as {outcome}.")
            else:
                app_refs = subset["Application Ref"].tolist()
                selected_ref = st.selectbox(f"Select Applicant ({outcome})", options=app_refs, key=f"sel_{outcome}")
                if selected_ref:
                    row = subset[subset["Application Ref"] == selected_ref].iloc[0]
                    _render_individual_dashboard(row, details)

    st.divider()
    with st.expander("📁 View Complete Results Table & Downloads", expanded=False):
        st.subheader("📋 Detailed results")
        f1, f2, f3 = st.columns(3)
        with f1:
            decision_filter = st.multiselect(
                "Filter by decision",
                options=list(results_df["Decision"].unique()),
                default=list(results_df["Decision"].unique()),
            )
        with f2:
            min_score = st.slider("Minimum score", 0, 100, 0)
        with f3:
            risk_level_filter = st.multiselect(
                "Filter by risk level",
                options=list(results_df["Risk Level"].unique()),
                default=list(results_df["Risk Level"].unique()),
            )

        filtered = results_df[
            (results_df["Decision"].isin(decision_filter))
            & (results_df["Score"] >= min_score)
            & (results_df["Risk Level"].isin(risk_level_filter))
        ]
        st.dataframe(filtered, use_container_width=True, hide_index=True)

        st.subheader("💾 Download results")
        dc1, dc2 = st.columns(2)
        with dc1:
            buf = io.StringIO()
            results_df.to_csv(buf, index=False)
            st.download_button(
                label="📥 Download results CSV",
                data=buf.getvalue(),
                file_name=f"chirp_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        with dc2:
            if errors_df is not None and not errors_df.empty:
                ebuf = io.StringIO()
                errors_df.to_csv(ebuf, index=False)
                st.download_button(
                    label="📥 Download errors CSV",
                    data=ebuf.getvalue(),
                    file_name=f"chirp_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

        if errors_df is not None and not errors_df.empty:
            st.subheader("❌ Processing errors")
            st.dataframe(errors_df, use_container_width=True, hide_index=True)


def render_help_tab() -> None:
    st.header("ℹ️ Help & documentation")

    st.subheader("📁 Chirp JSON format")
    st.markdown(
        """
Expected shape (from Chirp Open Banking): top-level **`TransactionSummaries`** array.
Each transaction should include **`amount`**, **`type`** (`CREDIT` / `DEBIT`), **`date`**,
**`category`**, **`top_level_category`**, and optional flags (`is_direct_deposit`, `is_fee`, …).
"""
    )
    st.code(
        """
{
  "Success": true,
  "Accounts": [ ... ],
  "TransactionSummaries": [
    {
      "date": "2026-03-05",
      "amount": 11.08,
      "type": "DEBIT",
      "description": "Merchant name",
      "category": "Personal Care",
      "top_level_category": "Personal Care",
      "categoryCode": "PRC"
    }
  ]
}
""",
        language="json",
    )
    st.markdown(
        """
**Amount convention:** Chirp typically uses **positive amounts** with **`type`** indicating direction.
The bridge converts to the engine convention (**negative = credit in**, **positive = debit out**).
"""
    )

    st.subheader("📊 Scoring system")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
**Score components (100 points total):**

1. **Affordability (30)** — DTI, disposable income, post-loan affordability  
2. **Income quality (35)** — Stability, regularity, verification, credit-history bonus  
"""
        )
    with col2:
        st.markdown(
            """
3. **Account conduct (25)** — Failed payments, overdraft usage, balance management  
4. **Risk indicators (10)** — Gambling activity, HCSTC-style history  
"""
        )

    st.subheader("📊 Configured score ranges")
    st.markdown(
        f"""
- **≥{SCORING_CONFIG["score_ranges"]["approve"]["min"]}**: APPROVE  
- **{SCORING_CONFIG["score_ranges"]["refer"]["min"]}–{SCORING_CONFIG["score_ranges"]["refer"]["max"]}**: REFER  
- **≤{SCORING_CONFIG["score_ranges"]["decline"]["max"]}**: DECLINE  
"""
    )

    st.subheader("📋 Product parameters (same engine as UK)")
    st.info(
        f"""
**Loan range**: £{PRODUCT_CONFIG['min_loan_amount']:,} – £{PRODUCT_CONFIG['max_loan_amount']:,}  
**Terms**: {', '.join(str(t) for t in PRODUCT_CONFIG['available_terms'])} months  
**Daily interest (FCA cap)**: {PRODUCT_CONFIG['daily_interest_rate']*100}% per day  
**Total cost cap**: {PRODUCT_CONFIG['total_cost_cap']*100:.0f}%  
"""
    )

    st.subheader("🚫 Hard decline rules")
    st.markdown(
        """
Engine-level gates include (among others) very low income stability, critical affordability issues,
and certain risk combinations — see `openbanking_engine/config/scoring_config.py` for the live list.
"""
    )


def main() -> None:
    _init_session()

    st.set_page_config(
        page_title="Chirp Loan Scorer (US)",
        page_icon="🇺🇸",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

            /* Main Typography */
            html, body, [class*="css"]  {
                font-family: 'Outfit', sans-serif;
            }

            /* Better Application Background */
            .stApp {
                background: inherit;
            }
            .stApp > header {
                background-color: transparent;
            }

            /* Stunning Headings */
            .main-header {
                font-family: 'Outfit', sans-serif;
                font-size: 3rem !important;
                font-weight: 700 !important;
                background: linear-gradient(120deg, #4f46e5 0%, #ec4899 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.25rem;
                padding-bottom: 0.5rem;
                text-shadow: 0px 4px 10px rgba(0,0,0,0.05); /* Subtle depth */
            }
            .main-subheader {
                font-size: 1.15rem;
                color: #64748b;
                margin-bottom: 2rem;
                font-weight: 400;
            }

            /* Sleek Metric Cards */
            [data-testid="stMetric"] {
                background-color: rgba(255, 255, 255, 0.4);
                backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
                border: 1px solid rgba(255, 255, 255, 0.5);
                border-radius: 16px;
                padding: 1.5rem 1rem !important;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
                transition: transform 0.25s ease, box-shadow 0.25s ease;
            }
            [data-testid="stMetric"]:hover {
                transform: translateY(-4px);
                box-shadow: 0 12px 30px rgba(0, 0, 0, 0.08);
            }
            [data-testid="stMetricValue"] {
                font-size: 2.2rem;
                font-weight: 700;
                color: #0f172a;
            }
            [data-testid="stMetricLabel"] {
                font-size: 0.95rem;
                color: #475569;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                font-weight: 600;
            }

            /* Custom Primary Button Glow/Hover */
            .stButton > button[data-testid="baseButton-primary"] {
                background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
                color: white;
                font-family: 'Outfit', sans-serif;
                font-weight: 600;
                border-radius: 10px;
                padding: 0.75rem 2rem;
                border: none;
                transition: all 0.3s ease;
                box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.39);
            }
            .stButton > button[data-testid="baseButton-primary"]:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.5);
            }

            /* Expander styling */
            div[data-testid="stExpander"] {
                border-radius: 12px;
                border: 1px solid #e2e8f0;
                box-shadow: 0 2px 10px rgba(0,0,0,0.02);
            }

            /* Divider styling */
            hr {
                border-top: 1px opacity #e2e8f0;
                opacity: 0.6;
            }

            /* Graph and Table Cards */
            .stPlotlyChart, [data-testid="stDataFrame"] {
                background-color: white !important;
                border-radius: 16px !important;
                padding: 1.5rem !important;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05) !important;
                border: 1px solid #e2e8f0 !important;
                transition: transform 0.2s ease, box-shadow 0.2s ease !important;
                margin-bottom: 1rem;
            }
            .stPlotlyChart:hover, [data-testid="stDataFrame"]:hover {
                transform: translateY(-4px) !important;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1) !important;
            }

            /* Custom HTML Cards */
            .custom-card {
                border-radius: 16px;
                padding: 1.5rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
                margin-bottom: 1.5rem;
            }
            .custom-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            }
            .card-blue { background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border: 1px solid #bfdbfe; }
            .card-blue .card-title, .card-blue .card-value { color: #1e3a8a; }
            
            .card-green { background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border: 1px solid #bbf7d0; }
            .card-green .card-title, .card-green .card-value { color: #14532d; }
            
            .card-red { background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); border: 1px solid #fecaca; }
            .card-red .card-title, .card-red .card-value { color: #7f1d1d; }

            .card-orange { background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%); border: 1px solid #fed7aa; }
            .card-orange .card-title, .card-orange .card-value { color: #7c2d12; }
            
            .card-purple { background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%); border: 1px solid #e9d5ff; }
            .card-purple .card-title, .card-purple .card-value { color: #581c87; }

            .card-title {
                font-size: 0.95rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.8;
                display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;
            }
            .card-value {
                font-size: 2.2rem; font-weight: 700; line-height: 1; margin-bottom: 0.25rem;
            }
            .card-subtitle {
                font-size: 0.85rem; font-weight: 600; opacity: 0.7;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="main-header">🇺🇸 Chirp Open Banking</p>'
        '<p class="main-subheader">Intelligent Loan Scoring & Underwriting Dashboard</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("⚙️ Configuration")
        st.subheader("Loan parameters")
        loan_amount = st.number_input(
            "Requested loan amount",
            min_value=float(PRODUCT_CONFIG["min_loan_amount"]),
            max_value=float(PRODUCT_CONFIG["max_loan_amount"]),
            value=500.0,
            step=50.0,
            help="Uses the same product bounds as the UK product config (displayed in GBP in engine docs).",
        )
        loan_term = st.selectbox(
            "Loan term (months)",
            options=PRODUCT_CONFIG["available_terms"],
            index=min(3, len(PRODUCT_CONFIG["available_terms"]) - 1),
        )

        st.subheader("Data parameters")
        use_auto_months = st.checkbox(
            "Auto-calculate months from data",
            value=True,
            help="Matches UK app: lookback is derived from transaction dates inside the engine.",
        )
        months_override: Optional[int] = None
        if use_auto_months:
            st.info("✓ Months are derived from transaction dates (MetricsCalculator).")
        else:
            months_override = st.slider(
                "Months of transaction data (manual override)",
                min_value=1,
                max_value=12,
                value=3,
            )

        days_covered = st.slider(
            "Days covered (API hint)",
            min_value=30,
            max_value=365,
            value=90,
            help="Passed through to the Chirp runner for parity; core lookback is transaction-driven.",
        )

        st.divider()
        st.subheader("📋 Product information")
        st.info(
            f"""
**Loan range**: £{PRODUCT_CONFIG['min_loan_amount']:,} – £{PRODUCT_CONFIG['max_loan_amount']:,}

**Terms**: {', '.join(str(t) for t in PRODUCT_CONFIG['available_terms'])} months

**Interest**: {PRODUCT_CONFIG['daily_interest_rate']*100}% per day

**Total cost cap**: {PRODUCT_CONFIG['total_cost_cap']*100:.0f}%
"""
        )

        st.divider()
        st.markdown("### 📊 Score ranges")
        st.markdown(
            f"""
- **>{SCORING_CONFIG["score_ranges"]["approve"]["min"] - 1}**: APPROVE ✅  
- **{SCORING_CONFIG["score_ranges"]["refer"]["min"]}–{SCORING_CONFIG["score_ranges"]["refer"]["max"]}**: REFER 📋  
- **<{SCORING_CONFIG["score_ranges"]["refer"]["min"]}**: DECLINE ❌  
"""
        )

    render_upload_tab(
        loan_amount=loan_amount,
        loan_term=loan_term,
        days_covered=days_covered,
        use_auto_months=use_auto_months,
        months_override=months_override,
    )

    if st.session_state.get("chirp_results_df") is not None and not st.session_state.get("chirp_results_df").empty:
        st.divider()
        render_results_tab()

    st.divider()
    with st.expander("ℹ️ Help & documentation", expanded=False):
        render_help_tab()


if __name__ == "__main__":
    main()
