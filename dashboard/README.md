# Power BI Dashboard & Analytics Integration
**Contributor**: Rajesh (Rajarsh) — Week 4 Day 4  

This directory contains the production-grade **Power BI Data Integration Pipeline**, **DAX Formula Library**, **M-Query ETL Script**, **1-Click Data Source Connector (`.pbids`)**, and **Interactive HTML Dashboard Preview**.

---

## 📂 Assets Included

| File | Purpose |
|---|---|
| [`Customer_Churn_LTV.pbids`](./Customer_Churn_LTV.pbids) | **1-Click Power BI Desktop connection file**. Double-click to open Power BI directly connected to the data! |
| [`DAX_MEASURES.dax`](./DAX_MEASURES.dax) | **18 Production DAX measures** (Portfolio LTV, Churn Risk %, Revenue at Risk, What-If Simulator). |
| [`PowerQuery_ETL.m`](./PowerQuery_ETL.m) | **Power Query M-code** to automate data typing and transformations. |
| [`index.html`](./index.html) | **Full interactive browser dashboard** with live KPI cards, Chart.js visuals, and What-If slider. |
| [`powerbi_customer_churn_ltv_master.csv`](./powerbi_customer_churn_ltv_master.csv) | Master denormalized dataset with ML Churn probabilities, LTV predictions, and 2x2 Retention Segments. |
| [`dim_customers.csv`](./dim_customers.csv) | Customer demographic dimension table. |
| [`dim_contracts.csv`](./dim_contracts.csv) | Contract and billing dimension table. |
| [`dim_services.csv`](./dim_services.csv) | Service subscriptions dimension table. |
| [`fact_churn_ltv_predictions.csv`](./fact_churn_ltv_predictions.csv) | Predictive fact table. |

---

## 📊 Strategic Retention Matrix Summary

From the 7,043 customer accounts analyzed:

- 🚨 **VIP at Risk**: **478 accounts** | **$2,126,651.75 Revenue at Stake** (High-touch rescue priority).
- 🏆 **Loyal Champions**: **2,339 accounts** | **$15,596,623.30 Total Value** (Account retention & advocacy).
- 📈 **Growth Opportunities**: **2,419 accounts** | **$3,042,191.75 Total Value** (Nurture & annual contract migration).
- ⚠️ **Low-Value Churn**: **1,807 accounts** | **$1,547,551.85 Total Value** (Automated digital re-engagement).

---

## 🚀 Quick Start
1. To view the interactive dashboard in your browser immediately, open `dashboard/index.html`.
2. To load into Power BI Desktop, open `dashboard/Customer_Churn_LTV.pbids` or import `dashboard/powerbi_customer_churn_ltv_master.csv`.
3. Follow the full step-by-step instructions in [`week_4/POWER_BI_DASHBOARD_GUIDE.md`](../week_4/POWER_BI_DASHBOARD_GUIDE.md).
