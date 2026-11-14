-- MySQL 8.0+。调用方须先在同一会话中加载 training_customers 临时表。
-- 所有结果表均为会话临时表，断开连接后自动清理。
CREATE TEMPORARY TABLE overall_churn AS
SELECT
    COUNT(*) AS 客户数,
    CAST(SUM(CASE WHEN Churn = 1 THEN 1 ELSE 0 END) AS UNSIGNED) AS 流失人数,
    1e0 * SUM(CASE WHEN Churn = 1 THEN 1 ELSE 0 END) / COUNT(*) AS 流失率
FROM training_customers;

-- 只引用一次训练集临时表，避免 MySQL UNION 重复打开临时表的限制。
CREATE TEMPORARY TABLE customer_segments AS
WITH grouped_segments AS (
    SELECT
        dimensions.分群维度,
        CASE dimensions.分群维度
            WHEN '合同类型' THEN Contract
            WHEN '在网时长' THEN
                CASE
                    WHEN tenure <= 12 THEN '1-12 个月'
                    WHEN tenure <= 24 THEN '13-24 个月'
                    WHEN tenure <= 48 THEN '25-48 个月'
                    ELSE '49 个月及以上'
                END
            WHEN '月费区间' THEN
                CASE
                    WHEN MonthlyCharges < 35 THEN '低于 35'
                    WHEN MonthlyCharges < 70 THEN '35-69.99'
                    ELSE '70 及以上'
                END
        END AS 分群,
        COUNT(*) AS 客户数,
        CAST(SUM(CASE WHEN Churn = 1 THEN 1 ELSE 0 END) AS UNSIGNED) AS 流失人数,
        1e0 * SUM(CASE WHEN Churn = 1 THEN 1 ELSE 0 END) / COUNT(*) AS 流失率
    FROM training_customers
    CROSS JOIN (
        SELECT '合同类型' AS 分群维度
        UNION ALL SELECT '在网时长'
        UNION ALL SELECT '月费区间'
    ) AS dimensions
    GROUP BY dimensions.分群维度, 分群
)
SELECT
    *,
    RANK() OVER (PARTITION BY 分群维度 ORDER BY 流失率 DESC) AS 流失率排名
FROM grouped_segments;

CREATE TEMPORARY TABLE contract_tenure_segments AS
SELECT
    Contract AS 合同类型,
    CASE
        WHEN tenure <= 12 THEN '1-12 个月'
        WHEN tenure <= 24 THEN '13-24 个月'
        WHEN tenure <= 48 THEN '25-48 个月'
        ELSE '49 个月及以上'
    END AS 在网时长分组,
    COUNT(*) AS 客户数,
    CAST(SUM(CASE WHEN Churn = 1 THEN 1 ELSE 0 END) AS UNSIGNED) AS 流失人数,
    1e0 * SUM(CASE WHEN Churn = 1 THEN 1 ELSE 0 END) / COUNT(*) AS 流失率
FROM training_customers
GROUP BY 合同类型, 在网时长分组;
