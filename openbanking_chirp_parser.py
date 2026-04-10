import json
import re
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
from typing import Dict, Any, List, Tuple

def categorize_transaction(row: Dict[str, Any]) -> str:
    tlc = str(row.get('top_level_category', '')).upper()
    cat = str(row.get('category', '')).upper()
    desc = str(row.get('description', '')).upper()
    orig_desc = str(row.get('original_description', '')).upper()
    merchant = str(row.get('merchant_name', '')).upper()
    txn_type = str(row.get('type', '')).upper()
    amount = float(row.get('amount', 0.0))

    combined_text = f"{desc} {orig_desc} {merchant}".strip()
    
    plaid_detailed = str(row.get('personal_finance_category', {}).get('detailed', '')).upper()
    if not plaid_detailed:
        plaid_detailed = cat

    # 1. categories & Explicit Risk
    if "BANK_FEES_INSUFFICIENT_FUNDS" in plaid_detailed or row.get('is_overdraft_fee') or "OVERDRAFT" in combined_text or "NSF" in combined_text or "INSUFFICIENT" in combined_text:
        return "Risk: NSF / Overdraft"
        
    if "BANK_FEES_OVERDRAFT" in plaid_detailed:
        return "Risk: NSF / Overdraft"

    if "ENTERTAINMENT_CASINOS_AND_GAMBLING" in plaid_detailed or "GAMBLING" in tlc or "CASINO" in combined_text or "BETTING" in combined_text or "LOTTERY" in combined_text:
        return "Risk: Gambling"
        
    if cat == "LATE FEE" or (row.get('is_fee') and "LATE" in combined_text):
        return "Risk: Late Fees"

    # 2. HCSTC / Payday Detection
    HCSTC_LENDERS = [
        "LENDING STREAM", "LENDINGSTREAM", "DRAFTY", "MR LENDER", "MRLENDER",
        "MONEYBOAT", "CREDITSPRING", "CASHFLOAT", "QUIDMARKET", "LOANS 2 GO", "LOANS2GO",
        "CASHASAP", "POLAR CREDIT", "118 118 MONEY", "THE MONEY PLATFORM", "FAST LOAN UK",
        "CONDUIT", "SALAD MONEY", "FAIR FINANCE", "SAVVY LOAN PRODUCTS", "LIKELY LOANS"
    ]
    if row.get('is_payroll_advance') or "PAYDAY" in combined_text or "CASH ADVANCE" in combined_text:
        return "Risk: Payday / HCSTC"
    if any(lender in combined_text for lender in HCSTC_LENDERS):
        return "Risk: Payday / HCSTC"

    # 3. Income Classification
    # Only treat as income if it's a credit (money in)
    is_credit = False
    if amount < 0 or txn_type == "CREDIT":
        is_credit = True
        
    KNOWN_EXPENSE_SERVICES = [
        "PAYPAL", "STRIPE", "SQUARE", "WORLDPAY", "SAGEPAY", "CLEARPAY", "KLARNA", 
        "ZILCH", "LAYBUY", "LENDABLE", "ZOPA", "BAMBOO", "OAKBROOK"
    ]

    # Exclude loan disbursements from income
    if "LOAN_PAYMENTS" in plaid_detailed or "LOANS" in plaid_detailed or "CASH_ADVANCES" in plaid_detailed:
        if is_credit:
            return "Other Credit/Deposit" # Not income
            
    if row.get('is_income') or "INCOME" in tlc or "PAYROLL" in combined_text or "DIRECT DEP" in combined_text:
        return "Income"

    if is_credit:
        # Check against Known Expense Services (refunds are not income)
        is_refund = any(service in combined_text for service in KNOWN_EXPENSE_SERVICES)
        if is_refund and not ("PAYOUT" in combined_text or "DISBURSEMENT" in combined_text):
            return "Other Credit/Deposit"
            
        # Transfer Promotion Logic
        exclusion_keywords = ["OWN ACCOUNT", "INTERNAL", "SELF TRANSFER", "FROM SAVINGS", "TO SAVINGS", "MOVED FROM", "POT", "VAULT", "ROUND UP"]
        if not any(kw in combined_text for kw in exclusion_keywords):
            gig_keywords = ["UBER", "DELIVEROO", "JUST EAT", "STRIPE PAYOUT", "PAYPAL PAYOUT"]
            payroll_keywords = ["SALARY", "WAGES", "PAYROLL", "NET PAY", "WAGE", "PAYSLIP", "EMPLOYER", "MONTHLY PAY", "WEEKLY PAY", "BGC", "BANK GIRO CREDIT"]
            benefit_keywords = ["UNIVERSAL CREDIT", "DWP", "CHILD BENEFIT", "PIP", "DLA", "ESA", "JSA", "HMRC"]
            
            if any(kw in combined_text for kw in gig_keywords + payroll_keywords + benefit_keywords):
                return "Income"
            if re.search(r'\b(LTD|LIMITED|PLC|LLP|INC|CORP)\b', combined_text) and abs(amount) >= 200:
                return "Income"
            if "FP-" in combined_text and abs(amount) >= 200: # Faster payments
                return "Income"

    # 4. Housing & Debt
    if "MORTGAGE" in cat or "RENT" in cat or "MORTGAGE" in combined_text or "RENT" in orig_desc:
        if txn_type == "DEBIT" or amount > 0:
            return "Housing"
        
    if "LOAN" in desc or "FINANCE" in orig_desc or "CREDIT CARD" in desc or "PAYMENT" in orig_desc:
        if txn_type == "DEBIT" or amount > 0:
            return "Debt Servicing"
            
    if "TRANSFER" in plaid_detailed:
        if "ACCOUNT_TRANSFER" in plaid_detailed or "INTERNAL" in combined_text:
            return "Outbound Transfer" if (txn_type == "DEBIT" or amount > 0) else "Inbound Transfer"

    # 5. Essential vs Discretionary
    essential_categories = ["FOOD & DINING", "GROCERIES", "HEALTH & FITNESS", "UTILITIES", "AUTO & TRANSPORT"]
    if tlc in essential_categories:
        if cat in ["RESTAURANTS", "FAST FOOD", "COFFEE SHOPS", "ALCOHOL & BARS"]:
            return "Discretionary"
        return "Essential"
        
    if tlc in ["ENTERTAINMENT", "SHOPPING", "PERSONAL CARE", "TRAVEL"]:
        return "Discretionary"

    if tlc == "TRANSFER" or "TRANSFER" in orig_desc:
        if txn_type == "DEBIT" or amount > 0:
            return "Outbound Transfer"
        return "Inbound Transfer"
        
    if txn_type == "DEBIT" or amount > 0:
        return "Other Untracked Expense"
    
    return "Other Credit/Deposit"

def _calc_income_stability(monthly_income: Dict[str, float]) -> float:
    if len(monthly_income) < 2:
        return 50.0
    values = list(monthly_income.values())
    mean_income = statistics.mean(values)
    if mean_income == 0:
        return 0.0
    std_dev = statistics.stdev(values) if len(values) > 1 else 0
    cv = (std_dev / mean_income) * 100
    return max(0, min(100, 100 - cv))

def _calc_income_regularity(income_dates: List[datetime]) -> float:
    days = [d.day for d in income_dates]
    if len(days) < 2:
        return 50.0
    std_dev = statistics.stdev(days) if len(days) > 1 else 0
    if std_dev <= 2: return 100.0
    elif std_dev <= 5: return 80.0
    elif std_dev <= 10: return 60.0
    elif std_dev <= 15: return 40.0
    return 20.0

def _calc_income_trend(monthly_income: Dict[str, float]) -> Tuple[str, float]:
    if len(monthly_income) < 3:
        return "stable", 0.0
    sorted_months = sorted(monthly_income.keys())
    monthly_values = [monthly_income[m] for m in sorted_months]
    
    recent_avg = sum(monthly_values[-2:]) / 2 
    older_avg = sum(monthly_values[:-2]) / max(1, len(monthly_values) - 2)
    
    if older_avg == 0:
        return "stable", 0.0
    change_pct = (recent_avg - older_avg) / older_avg * 100
    
    if change_pct > 10:
        return "increasing", change_pct
    elif change_pct < -10:
        return "decreasing", change_pct
    return "stable", change_pct

def analyze_open_banking(file_path: str, loan_amount=500.0, loan_term=4) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        data = json.load(f)
        
    accounts = data.get("Accounts", [])
    txns = data.get("TransactionSummaries", [])
    
    if not txns:
        return {"Error": "No transactions found"}

    dates = []
    for t in txns:
        date_str = t.get('date', '')
        if date_str:
            try:
                date_val = datetime.strptime(date_str, "%Y-%m-%d")
                dates.append(date_val)
                t['_parsed_date'] = date_val
            except ValueError:
                t['_parsed_date'] = None
    
    if dates:
        min_date = min(dates)
        max_date = max(dates)
        days_history = (max_date - min_date).days
        months_analyzed = max(1.0, round(days_history / 30.436875, 2))
    else:
        min_date, max_date = "Unknown", "Unknown"
        months_analyzed = 1.0

    cat_totals = defaultdict(float)
    total_outflow = 0.0
    
    monthly_income_dict = defaultdict(float)
    income_dates = []
    risk_counts = defaultdict(int)

    for t in txns:
        category = categorize_transaction(t)
        t['uw_category'] = category
        amount = t.get('amount', 0.0)
        
        cat_totals[category] += amount
        
        if str(t.get('type', '')).upper() == 'DEBIT':
            total_outflow += amount
            
        if category in ['Risk: NSF / Overdraft', 'Risk: Payday / HCSTC', 'Risk: Gambling', 'Risk: Late Fees']:
            risk_counts[category] += 1
            
        if category == "Income":
            d = t.get('_parsed_date')
            if d:
                month_key = d.strftime("%Y-%m")
                monthly_income_dict[month_key] += amount
                if amount >= 100:
                    income_dates.append(d)

    monthly_income = cat_totals["Income"] / months_analyzed
    monthly_outflow = total_outflow / months_analyzed
    monthly_essential = cat_totals["Essential"] / months_analyzed
    monthly_discretionary = cat_totals["Discretionary"] / months_analyzed
    monthly_debt = cat_totals["Debt Servicing"] / months_analyzed
    monthly_housing = cat_totals["Housing"] / months_analyzed
    
    dti = (monthly_debt / monthly_income * 100) if monthly_income > 0 else 0
    hti = (monthly_housing / monthly_income * 100) if monthly_income > 0 else 0
    
    monthly_disposable = monthly_income - monthly_essential - monthly_housing - monthly_debt
    
    # Advanced HCSTC Metrics
    stability_score = round(_calc_income_stability(monthly_income_dict), 1)
    regularity_score = round(_calc_income_regularity(income_dates), 1)
    income_trend, income_trend_pct = _calc_income_trend(monthly_income_dict)
    
    # Affordability / Proposed Loan
    interest_rate_cap = min(loan_amount * (0.008 * 30.4) * loan_term, loan_amount * 1.0) # Using 0.8% FCA daily price cap
    proposed_repayment = (loan_amount + interest_rate_cap) / loan_term
    post_loan_disposable = monthly_disposable - proposed_repayment
    
    # Risk Tier Logic
    flags = 0
    if stability_score < 50: flags += 1
    if monthly_debt < 200: flags += 1 # Low formal debt management flag
    if post_loan_disposable < 0: flags += 1
    if risk_counts['Risk: NSF / Overdraft'] > (months_analyzed * 2): flags += 1 # More than 2 NSFs per month is a massive flag
    
    risk_tier = "CLEAN"
    decision = "APPROVE"
    if flags >= 3:
        risk_tier = "FLAG"
        decision = "DECLINE"
    elif flags == 2:
        risk_tier = "WATCH"
        decision = "REFER"

    report = {
        "1_OVERVIEW": {
            "Decision": decision,
            "Risk Level": risk_tier,
            "Risk Flags Count": flags,
            "Total Accounts Linked": len(accounts),
            "Total Transactions": len(txns),
            "History Period": f"{min_date.date() if isinstance(min_date, datetime) else min_date} to {max_date.date() if isinstance(max_date, datetime) else max_date}",
            "Months Analyzed": months_analyzed
        },
        "2_ADVANCED_METRICS_AND_AFFORDABILITY": {
            "Est. Monthly Income": round(monthly_income, 2),
            "Income Stability Score": stability_score,
            "Income Regularity Score": regularity_score,
            "Income Trend": f"{income_trend} ({round(income_trend_pct, 1)}%)",
            "Est. Monthly Essential + Housing": round(monthly_essential + monthly_housing, 2),
            "Est. Monthly Discretionary": round(monthly_discretionary, 2),
            "Est. Monthly Debt Servicing": round(monthly_debt, 2),
            "Debt-to-Income (DTI) %": round(dti, 2),
            "Housing-to-Income (HTI) %": round(hti, 2),
            "Monthly Disposable (Pre-Loan)": round(monthly_disposable, 2)
        },
        "3_LOAN_SCENARIO (Amount: £500, Term: 4m)": {
            "Proposed Monthly Repayment": round(proposed_repayment, 2),
            "Post-Loan Disposable": round(post_loan_disposable, 2),
            "Affordable?": post_loan_disposable > 0
        },
        "4_RISK_SIGNALS_AND_BEHAVIOR": {
            "NSF / Overdraft Instances": risk_counts['Risk: NSF / Overdraft'],
            "NSF / Overdraft Per Month": round(risk_counts['Risk: NSF / Overdraft'] / months_analyzed, 2),
            "Late Fee Instances": risk_counts['Risk: Late Fees'],
            "Payday Loan Interventions": risk_counts['Risk: Payday / HCSTC'],
            "Gambling Transactions": risk_counts['Risk: Gambling']
        }
    }

    return report

if __name__ == "__main__":
    report = analyze_open_banking("ChirpOB.json")
    print(json.dumps(report, indent=4))
