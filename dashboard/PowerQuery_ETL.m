// ============================================================================
// POWER QUERY (M) ETL SCRIPT
// Project: Customer Churn & LTV Power BI Dashboard
// Contributor: Rajesh (Rajarsh) - Week 4 Day 4
// ============================================================================

let
    // 1. Load Master CSV
    Source = Csv.Document(File.Contents("powerbi_customer_churn_ltv_master.csv"), [Delimiter=",", Columns=27, Encoding=65001, QuoteStyle=QuoteStyle.None]),
    #"Promoted Headers" = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),

    // 2. Set Accurate Data Types
    #"Changed Type" = Table.TransformColumnTypes(#"Promoted Headers",{
        {"customerID", type text},
        {"gender", type text},
        {"SeniorCitizen", Int64.Type},
        {"Partner", type text},
        {"Dependents", type text},
        {"tenure", Int64.Type},
        {"PhoneService", type text},
        {"MultipleLines", type text},
        {"InternetService", type text},
        {"OnlineSecurity", type text},
        {"OnlineBackup", type text},
        {"DeviceProtection", type text},
        {"TechSupport", type text},
        {"StreamingTV", type text},
        {"StreamingMovies", type text},
        {"Contract", type text},
        {"PaperlessBilling", type text},
        {"PaymentMethod", type text},
        {"MonthlyCharges", type number},
        {"TotalCharges", type number},
        {"Churn", type text},
        {"Churn_Probability", type number},
        {"Churn_Probability_Pct", type number},
        {"Predicted_Churn", type text},
        {"Churn_Risk_Level", type text},
        {"Historical_LTV", type number},
        {"Projected_Future_LTV", type number},
        {"Total_Expected_LTV", type number},
        {"LTV_Tier", type text},
        {"Retention_Segment", type text},
        {"Recommended_Action", type text}
    }, "en-US")
in
    #"Changed Type"
