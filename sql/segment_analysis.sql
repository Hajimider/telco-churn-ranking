CREATE TEMP VIEW overall_churn AS
SELECT
    COUNT(*) AS 客户数,
    SUM(CASE WHEN Churn = 1 THEN 1 ELSE 0 END) AS 流失人数,
    1.0 * SUM(CASE WHEN Churn = 1 THEN 1 ELSE 0 END) / COUNT(*) AS 流失率
FROM training_customers;

CREATE TEMP VIEW customer_segments AS
SELECT
    '合同类型' AS 分群维度,
    Contract AS 分群,
    COUNT(*) AS 客户数,
    SUM(CASE WHEN Churn = 1 THEN 1 ELSE 0 END) AS 流失人数,
    1.0 * SUM(CASE WHEN Churn = 1 THEN 1 ELSE 0 END) / COUNT(*) AS 流失率
FROM training_customers
GROUP BY Contract

UNION ALL

SELECT
    '在网时长' AS 分群维度,
    CASE
        WHEN CAST(tenure AS REAL) <= 12 THEN '1-12 个月'
        WHEN CAST(tenure AS REAL) <= 24 THEN '13-24 个月'
        WHEN CAST(tenure AS REAL) <= 48 THEN '25-48 个月'
        ELSE '49 个月及以上'
    END AS 分群,
    COUNT(*) AS 客户数,
    SUM(CASE WHEN Churn = 1 THEN 1 ELSE 0 END) AS 流失人数,
    1.0 * SUM(CASE WHEN Churn = 1 THEN 1 ELSE 0 END) / COUNT(*) AS 流失率
FROM training_customers
GROUP BY 分群

UNION ALL

SELECT
    '月费区间' AS 分群维度,
    CASE
        WHEN CAST(MonthlyCharges AS REAL) < 35 THEN '低于 35'
        WHEN CAST(MonthlyCharges AS REAL) < 70 THEN '35-69.99'
        ELSE '70 及以上'
    END AS 分群,
    COUNT(*) AS 客户数,
    SUM(CASE WHEN Churn = 1 THEN 1 ELSE 0 END) AS 流失人数,
    1.0 * SUM(CASE WHEN Churn = 1 THEN 1 ELSE 0 END) / COUNT(*) AS 流失率
FROM training_customers
GROUP BY 分群;
