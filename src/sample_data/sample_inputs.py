"""
LendSynthetix — Sample Loan Application Datasets
"""

SAMPLE_A_TECH_STARTUP = {
    "company": "NovaTech Solutions Pvt Ltd",
    "expected_outcome": "Sales vs Risk debate — likely APPROVED or ESCALATED",
    "text": """
COMMERCIAL LOAN APPLICATION — NOVATECH SOLUTIONS PVT LTD

LOAN REQUEST: ₹2.5 Crore working capital facility
LOAN PURPOSE: Expand SaaS product development team and accelerate enterprise sales

COMPANY OVERVIEW:
NovaTech Solutions was founded in 2021 and provides AI-powered HR automation software.
The company has grown from 3 to 47 employees in 2 years. Headquarters in Bengaluru, Karnataka.
Directors: Arjun Mehta (CEO), Priya Rajan (CTO). No criminal records, clean KYC.

FINANCIAL METRICS (FY 2023-24):
Annual Revenue: ₹3.8 Crore (up 68% YoY from ₹2.26 Crore)
Net Income: ₹28 Lakhs (positive for first time)
EBITDA: ₹55 Lakhs
Total Assets: ₹1.9 Crore
Total Liabilities: ₹1.1 Crore
Total Debt: ₹70 Lakhs (existing vehicle loans)
Equity / Net Worth: ₹80 Lakhs
Debt-to-Equity Ratio: 0.875
DSCR: 1.42 (operating cash flow ₹88L / annual debt service ₹62L)
Operating Cash Flow: ₹88 Lakhs

RECURRING CONTRACTS:
- Infosys BPM (3-year SaaS contract, ₹45L/year)
- HDFC Life Insurance (annual renewal, ₹28L/year)
- 12 SME clients on monthly subscriptions totaling ₹18L/year

COLLATERAL OFFERED:
- Registered office property (Bengaluru): valued at ₹95 Lakhs
- Personal guarantee from both directors
- Lien on receivables

GROWTH TRAJECTORY:
Q1 FY25 revenue already at ₹1.4 Crore (annualized: ₹5.6 Crore, +47% projected growth)
Signed LOI with TCS for ₹80L annual contract (pending final agreement)

COMPLIANCE / AML:
No offshore accounts. All transactions through HDFC Bank (current account for 4 years).
No PEP connections. GST compliant, ITR filed for 3 years. No outstanding tax disputes.
No unusual large cash transactions. Clean CIBIL score: 762.
""",
}

SAMPLE_B_MANUFACTURING = {
    "company": "Shree Ganesh Alloys Ltd",
    "expected_outcome": "Compliance VETO expected — director on grey list + offshore account",
    "text": """
COMMERCIAL LOAN APPLICATION — SHREE GANESH ALLOYS LTD

LOAN REQUEST: ₹8 Crore term loan
LOAN PURPOSE: Purchase of new CNC machinery and plant expansion in Pune

COMPANY OVERVIEW:
Shree Ganesh Alloys Ltd established in 1998. Manufactures precision metal components
for automotive OEMs. 180 employees. Plant located in Chakan, Pune.
Directors: Ramesh Deshmukh (MD), Vikram Joshi (CFO), Anand Kulkarni (Director).

FINANCIAL METRICS (FY 2023-24):
Annual Revenue: ₹24 Crore
Net Income: ₹2.1 Crore
EBITDA: ₹4.8 Crore
Total Assets: ₹31 Crore
Total Liabilities: ₹18 Crore
Total Debt: ₹14 Crore
Equity / Net Worth: ₹13 Crore
Debt-to-Equity Ratio: 1.08
DSCR: 1.61
Operating Cash Flow: ₹4.2 Crore

COLLATERAL:
- Factory land and building (Chakan): ₹18 Crore market value
- Plant and machinery: ₹9 Crore (depreciated book value ₹5.2 Crore)
- FD of ₹1.5 Crore pledged

COMPLIANCE / AML FLAGS:
**CRITICAL: Director Vikram Joshi appears on the Interpol financial crimes grey list
as of October 2023 (case reference: INTERPOL-FIN-2023-4471). Allegation: involvement
in cross-border invoice fraud scheme in UAE (2019-2021). Case under investigation.**

Offshore account detected: Shree Ganesh Alloys maintains a USD account with
Emirates NBD, Dubai (Account ending 8821). Large deposit of $2.1M received in
March 2024 — source of funds documentation NOT provided.

A $500,000 transfer was received from a Cayman Islands entity (Joshi Holdings LLC)
in December 2023. No explanation provided.

BANKING:
Primary account with Bank of Maharashtra for 12 years. Clean domestic transactions.
However, 3 large cash deposits (₹40L each) in Q4 FY24 with no clear source documentation.
""",
}

SAMPLE_C_RETAIL = {
    "company": "FreshMart Retail Chain",
    "expected_outcome": "Borderline ESCALATED — moderate risk, weak collateral",
    "text": """
COMMERCIAL LOAN APPLICATION — FRESHMART RETAIL CHAIN

LOAN REQUEST: ₹1.2 Crore
LOAN PURPOSE: Open 3 new grocery stores in Pune suburbs

COMPANY OVERVIEW:
FreshMart is a regional grocery chain operating 8 stores in Pune since 2018.
Founder: Sneha Patil (Sole proprietor). No co-directors.
Employees: 64 full-time, 30 part-time.

FINANCIAL METRICS (FY 2023-24):
Annual Revenue: ₹9.4 Crore
Net Income: ₹38 Lakhs
EBITDA: ₹72 Lakhs
Total Assets: ₹3.1 Crore
Total Liabilities: ₹2.4 Crore
Total Debt: ₹1.8 Crore (existing retail fitout loans)
Equity / Net Worth: ₹70 Lakhs
Debt-to-Equity Ratio: 2.57
DSCR: 1.18 (marginal)
Operating Cash Flow: ₹65 Lakhs

COLLATERAL:
- Personal residence of proprietor: ₹85 Lakhs
- Store fixtures and equipment: ₹40 Lakhs (low liquidity)
- No commercial property owned — all stores on lease

GROWTH:
Revenue grew 22% YoY. Same-store sales up 8%.
New stores expected to break even in 14 months.
Competition risk: Reliance Smart and DMart expanding in same area.

COMPLIANCE / AML:
Clean KYC. No offshore accounts. GST compliant.
One bounced cheque (EMI, April 2023) — subsequently cleared.
CIBIL: 694 (below preferred 750 threshold).
""",
}

ALL_SAMPLES = {
    "tech_startup":  SAMPLE_A_TECH_STARTUP,
    "manufacturing": SAMPLE_B_MANUFACTURING,
    "retail":        SAMPLE_C_RETAIL,
}
