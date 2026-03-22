"""
Sample loan application inputs for LendSynthetix testing.
Based on the two test cases specified in the hackathon problem statement.
"""

# ─────────────────────────────────────────────
# SAMPLE A: Tech Startup — Tests Sales vs Risk
# High growth, no physical collateral
# ─────────────────────────────────────────────

SAMPLE_A_TECH_STARTUP = {
    "company": "NexaCloud Technologies Pvt. Ltd.",
    "text": """
COMMERCIAL LOAN APPLICATION — CONFIDENTIAL
==========================================

COMPANY PROFILE
Company Name:     NexaCloud Technologies Pvt. Ltd.
Registration No:  U72900MH2021PTC345678
Industry:         B2B SaaS / Cloud Infrastructure
Founded:          2021 (3 years in operation)
Headquarters:     Mumbai, Maharashtra, India

LOAN REQUEST
Loan Amount:        ₹8,00,00,000 (₹8 Crore)
Loan Purpose:       Working capital expansion + sales team hiring
Tenure Requested:   5 years
Proposed EMI:       ₹2,10,000/month (estimated)

FINANCIAL SUMMARY (FY 2023-24)
Annual Revenue:         ₹4.2 Crore
Revenue Growth (YoY):   +47% (FY22: ₹2.8 Cr, FY23: ₹4.2 Cr)
Net Income:             ₹38 Lakhs
EBITDA:                 ₹72 Lakhs
EBITDA Margin:          17.1%

BALANCE SHEET HIGHLIGHTS
Total Assets:           ₹3.1 Crore
Fixed Assets:           ₹0.4 Crore (laptops, servers — no physical property)
Total Liabilities:      ₹2.2 Crore
Total Debt:             ₹1.8 Crore (existing term loan)
Equity / Net Worth:     ₹0.9 Crore
Debt-to-Equity Ratio:   2.0x

DEBT SERVICE
Existing EMI Obligations: ₹45,000/month
DSCR (Current):           1.35
Projected DSCR (post-loan): 1.18

COLLATERAL OFFERED
Primary:    Assignment of recurring SaaS contracts (₹3.1 Cr ARR)
Secondary:  Personal guarantee from both founders
Physical:   None (Pure-play software company)

CLIENT & REVENUE PROFILE
Top 5 Clients:       TCS Digital, Infosys BPM, HDFC Securities, L&T Tech, Wipro
Contract Type:       Annual recurring contracts (auto-renewing)
Recurring Revenue:   82% of total revenue is recurring/subscription
Recent Win:          3-year exclusive deal with HDFC Securities (₹1.2 Cr TCV)
New Contracts (Q1 FY25): +20% growth in recurring bookings vs Q1 FY24
Pipeline (90-day):   ₹2.8 Crore in qualified opportunities

MANAGEMENT TEAM
CEO:    Arjun Mehta — IIT Bombay, 8 years fintech experience
CTO:    Priya Sharma — Ex-Amazon AWS, 10 years cloud architecture
CFO:    Rajan Iyer — CA, Ex-Deloitte, 12 years corporate finance

COMPLIANCE & KYC
KYC Status:         Verified (Aadhaar, PAN, GST)
GST Compliance:     100% filings current
Tax Returns:        ITR filed for FY21, FY22, FY23
Directors:          All directors verified, no watchlist hits
AML Check:          No offshore accounts, no suspicious transactions
PEP Status:         None of the directors are Politically Exposed Persons
Source of Funds:    Domestic VC funding (Blume Ventures, ₹2 Cr seed round 2022)

PITCH HIGHLIGHTS
- Only SaaS provider in India with real-time cloud cost optimization for BFSI
- Proprietary ML model reduces cloud spend by avg 32% for clients
- 94% client retention rate over 3 years
- Target: ₹15 Crore ARR by FY27 (3x current)
""",
    "expected_outcome": "DEBATED — Sales should win with growth story; Risk concerned about low collateral and borderline DSCR"
}


# ─────────────────────────────────────────────
# SAMPLE B: Manufacturing Firm — Tests Compliance Veto
# Strong assets, but director on grey list
# ─────────────────────────────────────────────

SAMPLE_B_MANUFACTURING = {
    "company": "Bharat Steel & Forge Industries Ltd.",
    "text": """
COMMERCIAL LOAN APPLICATION — CONFIDENTIAL
==========================================

COMPANY PROFILE
Company Name:     Bharat Steel & Forge Industries Ltd.
Registration No:  L27100GJ1998PLC034521
Industry:         Heavy Manufacturing / Steel Fabrication
Founded:          1998 (26 years in operation)
Headquarters:     Surat, Gujarat, India

LOAN REQUEST
Loan Amount:        ₹25,00,00,000 (₹25 Crore)
Loan Purpose:       Plant modernization and capacity expansion
Tenure Requested:   7 years
Proposed EMI:       ₹4,80,000/month

FINANCIAL SUMMARY (FY 2023-24)
Annual Revenue:         ₹42 Crore
Revenue Growth (YoY):   +8% (stable, mature business)
Net Income:             ₹4.8 Crore
EBITDA:                 ₹9.2 Crore
EBITDA Margin:          21.9%

BALANCE SHEET HIGHLIGHTS
Total Assets:           ₹68 Crore
Fixed Assets:           ₹55 Crore (land, plant, machinery — fully owned)
Total Liabilities:      ₹18 Crore
Total Debt:             ₹14 Crore (term loans, fully serviced)
Equity / Net Worth:     ₹50 Crore
Debt-to-Equity Ratio:   0.28x (extremely healthy)

DEBT SERVICE
Existing EMI Obligations: ₹3,20,000/month
DSCR (Current):           2.87 (excellent)
Projected DSCR (post-loan): 1.65 (still strong)

COLLATERAL OFFERED
Primary:    Industrial land in Surat GIDC (valued ₹38 Crore)
Secondary:  Plant & machinery (valued ₹17 Crore)
Total Collateral:  ₹55 Crore (220% of loan amount)

MANAGEMENT TEAM
CMD:      Ramesh Patel — 30 years in steel industry
Director: Sunil Kothari — CFO background, CA, clean record
Director: Vikram Shah — Operations, Ex-SAIL
Director: Harish Mehta — International business development

COMPLIANCE & KYC
KYC Status:         Verified (PAN, GST, ROC)
GST Compliance:     100% filings current
Tax Returns:        All ITRs filed and audited FY19-FY24
Directors:          Partial verification complete

⚠️ COMPLIANCE ALERT:
Director Harish Mehta (DIN: 02345678) is listed as an "Associated Person" on the 
FATF (Financial Action Task Force) Grey List watch — specifically flagged as 
an associate of a UAE-based entity currently under scrutiny for potential 
trade-based money laundering.

OFFSHORE DEPOSITS:
A deposit of ₹1,00,00,000 (₹1 Crore) held in a Mauritius-based account 
(MauBank Account #MU17-MAUF-0000-0000-9030) was disclosed. 
Source of funds documentation NOT YET PROVIDED for this deposit.

AML NOTES:
- 3 cash transactions exceeding ₹10 Lakh detected in FY23 (disclosed)
- Trade invoice discrepancy of ₹2.1 Crore between export declaration and bank receipt (under clarification)

PITCH HIGHLIGHTS
- 26-year track record with zero loan defaults
- Supplies to Tata Steel, SAIL, L&T as Tier-1 vendor
- New defense contract with DRDO worth ₹12 Crore (3-year supply agreement)
- Modernization will double capacity from 8,000 MT to 16,000 MT/year
""",
    "expected_outcome": "COMPLIANCE VETO — Director on FATF grey list + unexplained offshore deposit = mandatory rejection"
}


# ─────────────────────────────────────────────
# QUICK ACCESS
# ─────────────────────────────────────────────

ALL_SAMPLES = {
    "tech_startup":   SAMPLE_A_TECH_STARTUP,
    "manufacturing":  SAMPLE_B_MANUFACTURING
}