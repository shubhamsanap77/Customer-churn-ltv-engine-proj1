# Power BI Dashboard Development & Integration Guide
**Project**: Customer Churn Prediction & Lifetime Value (LTV) Engine  
**Contributor**: Rajesh (Rajarsh) — Week 4 Day 4  

---

## 🎯 Executive Summary & Objectives

The goal of Week 4 Day 4 is to translate our machine learning models from Week 2 (Churn Classification) and Week 3 (LTV Regression) into an enterprise-grade **Power BI Interactive Business Intelligence Dashboard**.

### 💼 Key Business Metrics Integrated:
- **Total Portfolio LTV**: **$22.31 Million**
- **Accounts at High Churn Risk**: **2,285 accounts (32.4%)**
- **VIP Accounts at Immediate Risk**: **478 accounts** holding **$2.13 Million** in revenue.
- **Top 20% Accounts (Platinum Tier)**: Generate **53.3% of total company revenue** ($8,435 average LTV).

---

## 📂 Deliverables & File Directory

```text
dashboard/
├── Customer_Churn_LTV.pbids                  # 1-Click Power BI Data Source connector
├── DAX_MEASURES.dax                          # 18 ready-to-use DAX measures
├── PowerQuery_ETL.m                          # Power Query M-code data transform script
├── index.html                                # Interactive browser-based Power BI mockup
├── powerbi_customer_churn_ltv_master.csv     # Master denormalized dataset (7,043 rows)
├── dim_customers.csv                         # Customer demographic dimension table
├── dim_contracts.csv                         # Contract & billing dimension table
├── dim_services.csv                          # Service subscription dimension table
└── fact_churn_ltv_predictions.csv            # ML churn & LTV predictions fact table

week_4/
├── POWER_BI_DASHBOARD_GUIDE.md               # This complete step-by-step guide
├── export_powerbi_data.py                    # ETL pipeline script generating the datasets
├── run_week_4.py                             # Master runner script
└── data/                                     # Exported CSVs
```

---

## 🛠️ Step-by-Step: How to Build the Dashboard in Power BI

### Step 1: Import the Data into Power BI Desktop
1. Open **Microsoft Power BI Desktop**.
2. There are two ways to connect:
   - **Option A (Fastest)**: Double-click [`dashboard/Customer_Churn_LTV.pbids`](./Customer_Churn_LTV.pbids) — Power BI will launch and automatically link to the data!
   - **Option B**: Click **Get Data** ➔ **Text/CSV** ➔ Select [`dashboard/powerbi_customer_churn_ltv_master.csv`](./powerbi_customer_churn_ltv_master.csv) ➔ Click **Transform Data** (or Load).
3. If using **Transform Data**, verify column types:
   - `tenure`, `SeniorCitizen`: Whole Number
   - `MonthlyCharges`, `TotalCharges`, `Churn_Probability`, `Total_Expected_LTV`: Decimal Number
   - `customerID`, `Contract`, `Churn_Risk_Level`, `LTV_Tier`, `Retention_Segment`: Text

---

### Step 2: Add the DAX Measures Library
1. In the **Home** or **Modeling** tab, click **New Measure**.
2. Open [`dashboard/DAX_MEASURES.dax`](./DAX_MEASURES.dax) and copy the formulas:

#### Key DAX Formulas:
- **Total Customers**:
  ```dax
  Total Customers = COUNTROWS('powerbi_customer_churn_ltv_master')
  ```
- **Total Portfolio LTV**:
  ```dax
  Total Portfolio LTV = SUM('powerbi_customer_churn_ltv_master'[Total_Expected_LTV])
  ```
- **High Risk Customer Count**:
  ```dax
  High Risk Customer Count = CALCULATE([Total Customers], 'powerbi_customer_churn_ltv_master'[Churn_Risk_Level] = "High Risk")
  ```
- **VIP Revenue at Stake**:
  ```dax
  VIP Revenue at Stake ($) = CALCULATE([Total Portfolio LTV], 'powerbi_customer_churn_ltv_master'[Retention_Segment] = "VIP at Risk")
  ```
- **What-If Rescued Revenue ($)**:
  ```dax
  Gross Revenue Rescued ($) = [VIP Revenue at Stake ($)] * 0.50
  ```
- **Net Retention Campaign ROI ($)**:
  ```dax
  Net Retention Campaign ROI ($) = [Gross Revenue Rescued ($)] - ([VIP at Risk Customer Count] * 50)
  ```

---

### Step 3: Build the 4 Report Pages

#### 📑 Page 1: Executive KPI & Portfolio Overview
* **Top KPI Cards (Row 1)**:
  1. Card 1: `Total Customers` (7,043)
  2. Card 2: `High Risk Customer %` (32.4%)
  3. Card 3: `Total Portfolio LTV` ($22.31M)
  4. Card 4: `VIP Revenue at Stake` ($2.13M)
* **Visual 1 (Donut Chart)**: Revenue Share by `LTV_Tier` (`Total_Expected_LTV` as Values, `LTV_Tier` as Legend). Shows Platinum accounts hold 53.3% of revenue!
* **Visual 2 (Clustered Column Chart)**: Churn Rate by `Contract` (`Contract` on X-axis, `Average Churn Probability %` on Y-axis). Shows Month-to-month contracts have 42.7% churn hazard vs 2.8% for 2-year contracts.
* **Slicers (Top/Side)**: `Contract`, `InternetService`, `PaymentMethod`.

---

#### 📑 Page 2: Churn Risk & Behavioral Drivers
* **Visual 1 (Bar Chart)**: Churn Hazard by `InternetService` (Fiber optic has highest churn rate when unbundled from Tech Support).
* **Visual 2 (Column Chart)**: Churn Hazard by `tenure_cohort` (Shows new customers 0-12 months are at highest cancellation risk).
* **Visual 3 (Treemap / Bar Chart)**: Churn by `PaymentMethod` (Electronic check accounts have 45% churn rate vs automatic card/bank payments at ~15%).

---

#### 📑 Page 3: Strategic 2x2 Retention Quadrant & VIP Rescue Table
* **Visual 1 (Scatter Plot / 2x2 Matrix)**:
  - **X-Axis**: `Churn_Probability` (0.0 to 1.0)
  - **Y-Axis**: `Total_Expected_LTV` ($0 to $13,000)
  - **Legend / Color**: `Retention_Segment`
    - 🔴 **VIP at Risk**: High LTV + High Churn Risk (478 accounts, $2.13M at risk)
    - 🟢 **Loyal Champions**: High LTV + Low Churn Risk (2,339 accounts, $15.6M value)
    - 🟠 **Low-Value Churn**: Low LTV + High Churn Risk (1,807 accounts, $1.55M)
    - 🔵 **Growth Opportunities**: Low LTV + Low Churn Risk (2,419 accounts, $3.04M)
* **Visual 2 (Drill-Down Table for Marketing Outreach)**:
  - Add fields: `customerID`, `Contract`, `MonthlyCharges`, `Total_Expected_LTV`, `Churn_Probability_Pct`, `Recommended_Action`.
  - Filter by `Retention_Segment = "VIP at Risk"`.
  - Displays exact priority accounts with tailored retention playbooks (e.g. assign dedicated account rep, 20% renewal discount).

---

#### 📑 Page 4: Interactive What-If Retention ROI Simulator
* In Power BI, go to **Modeling** ➔ **New Parameter** ➔ **Numeric Range**:
  - Name: `Outreach Success Rate`
  - Data Type: Percentage (Min: 0.10, Max: 0.90, Increment: 0.05, Default: 0.50)
* Connect the slider to dynamic measures:
  - `Projected Rescued Revenue ($) = [VIP Revenue at Stake ($)] * [Outreach Success Rate Value]`
  - `Campaign Net ROI ($) = [Projected Rescued Revenue ($)] - ([VIP at Risk Customer Count] * 50)`
* Moving the slider dynamically updates executive ROI on the screen!

---

## 🌐 Instant Visual Preview

You can open [`dashboard/index.html`](file:///c:/Users/TUF%20GAMING/Downloads/Customer-churn-ltv-engine-proj1-main/Customer-churn-ltv-engine-proj1-main/dashboard/index.html) in Chrome or Edge right now to interact with the full dashboard mockup, test the What-If slider, and view all charts!
