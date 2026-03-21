-- Snowflake Setup for Churn Model Deployment
-- Run these commands in your Snowflake Worksheet

-- 1. Create a database for the project
CREATE DATABASE IF NOT EXISTS CHURN_PROJECT;
USE DATABASE CHURN_PROJECT;

-- 2. Create a schema for the models
CREATE SCHEMA IF NOT EXISTS ML_MODELS;
USE SCHEMA ML_MODELS;

-- 3. Create a stage for the model files
CREATE OR REPLACE STAGE CHURN_STAGE;

-- 4. Create the final table for predictions (Optional)
CREATE OR REPLACE TABLE CUSTOMER_CHURN_PREDICTIONS (
    CUSTOMER_ID STRING,
    CREDIT_SCORE FLOAT,
    CHURN_PROBABILITY FLOAT,
    RISK_CATEGORY STRING,
    TIMESTAMP TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Note: To upload files from your local machine to the stage, use the SNOWFLAKE CLI (SnowSQL):
-- snowsql -a <account_id> -u <username>
-- PUT file://./churn_stacking_model.joblib @CHURN_STAGE;
-- PUT file://./scaler.joblib @CHURN_STAGE;
