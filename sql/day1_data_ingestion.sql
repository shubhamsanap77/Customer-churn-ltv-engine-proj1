-- Day 1: Telco Customer Churn - Database Setup & Validation

-- Create table for Telco Customer Churn dataset
CREATE TABLE IF NOT EXISTS telco_customer_churn (
    customerid VARCHAR(50),
    gender VARCHAR(20),
    seniorcitizen INTEGER,
    partner VARCHAR(10),
    dependents VARCHAR(10),
    tenure INTEGER,
    phoneservice VARCHAR(10),
    multiplelines VARCHAR(30),
    internetservice VARCHAR(30),
    onlinesecurity VARCHAR(30),
    onlinebackup VARCHAR(30),
    deviceprotection VARCHAR(30),
    techsupport VARCHAR(30),
    streamingtv VARCHAR(30),
    streamingmovies VARCHAR(30),
    contract VARCHAR(30),
    paperlessbilling VARCHAR(10),
    paymentmethod VARCHAR(50),
    monthlycharges NUMERIC(10,2),
    totalcharges VARCHAR(30),
    churn VARCHAR(10)
);

-- Validate total number of rows
SELECT COUNT(*) AS total_rows
FROM telco_customer_churn;

-- Validate unique customers
SELECT COUNT(DISTINCT customerid) AS unique_customers
FROM telco_customer_churn;

-- Check for duplicate customer IDs
SELECT customerid, COUNT(*) AS duplicate_count
FROM telco_customer_churn
GROUP BY customerid
HAVING COUNT(*) > 1
LIMIT 10;