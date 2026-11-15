"""默认 SQLite 内存分群；可选 MySQL 会话临时表，不改写持久化表。"""
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd
import pymysql
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"


def connect_mysql():
    """读取项目配置，进程环境变量优先；凭据不拼入连接 URL。"""
    local_config = dotenv_values(ENV_FILE, encoding="utf-8-sig", interpolate=False)

    def setting(name, default=""):
        key = f"CHURN_MYSQL_{name}"
        return os.environ.get(key, local_config.get(key) or default)

    password = setting("PASSWORD")
    if not password:
        raise ValueError("请在 .env 或环境变量中设置 CHURN_MYSQL_PASSWORD。")
    try:
        port = int(setting("PORT", "3306"))
        if not 1 <= port <= 65535:
            raise ValueError
    except ValueError:
        raise ValueError("CHURN_MYSQL_PORT 必须是 1 到 65535 之间的整数。") from None
    try:
        return pymysql.connect(
            host=setting("HOST", "127.0.0.1"), port=port,
            user=setting("USER", "root"), password=password,
            database=setting("DATABASE", "telco_churn"),
            charset="utf8mb4", collation="utf8mb4_bin",
            connect_timeout=5, read_timeout=30, write_timeout=30,
        )
    except pymysql.MySQLError as exc:
        raise ValueError(
            f"MySQL 连接失败（错误码 {exc.args[0]}）。"
            "请检查服务、CHURN_MYSQL_* 配置，并先创建项目数据库。"
        ) from None


def analyze_training_segments(training_frame, backend="sqlite", include_pareto=False):
    """默认本机内存 SQLite；显式选择 mysql 时不静默吞掉连接错误。"""
    values = training_frame[["Contract", "tenure", "Churn"]].copy()
    # 交叉分群调用只需三列；缺少月费时不影响其结果。
    values["MonthlyCharges"] = training_frame.get("MonthlyCharges")
    for column in ["tenure", "MonthlyCharges", "Churn"]:
        values[column] = pd.to_numeric(values[column], errors="raise")
    values = values.astype(object).where(values.notna(), None)
    statements = (ROOT / "sql" / "segment_analysis.sql").read_text(encoding="utf-8")
    if backend == "sqlite":
        with closing(sqlite3.connect(":memory:")) as connection:
            values.to_sql("training_customers", connection, index=False)
            connection.executescript(statements)
            tables = ["overall_churn", "customer_segments", "contract_tenure_segments"]
            if include_pareto:
                tables.append("churn_pareto")
            return tuple(pd.read_sql_query(f"SELECT * FROM {table}", connection) for table in tables)
    if backend != "mysql":
        raise ValueError("分群后端须为 sqlite 或 mysql。")
    try:
        with connect_mysql() as connection:
            with connection.cursor() as cursor:
                cursor.execute("""
                    CREATE TEMPORARY TABLE training_customers (
                        Contract VARCHAR(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin,
                        tenure DOUBLE,
                        Churn TINYINT NOT NULL,
                        MonthlyCharges DOUBLE
                    )
                """)
                cursor.executemany(
                    "INSERT INTO training_customers (Contract, tenure, Churn, MonthlyCharges) "
                    "VALUES (%s, %s, %s, %s)",
                    list(values.itertuples(index=False, name=None)),
                )
                for statement in statements.split(";"):
                    if statement.strip():
                        cursor.execute(statement)
                results = []
                tables = ["overall_churn", "customer_segments", "contract_tenure_segments"]
                if include_pareto:
                    tables.append("churn_pareto")
                for table in tables:
                    cursor.execute(f"SELECT * FROM {table}")
                    results.append(pd.DataFrame.from_records(
                        cursor.fetchall(), columns=[column[0] for column in cursor.description]))
                return tuple(results)
    except pymysql.MySQLError as exc:
        raise ValueError(
            f"MySQL 分群查询失败（错误码 {exc.args[0]}）。"
            "需要 MySQL 8.0 及以上版本，以及项目库的 CREATE TEMPORARY TABLES 权限。"
        ) from None


if __name__ == "__main__":
    with connect_mysql() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT VERSION(), DATABASE()")
            version, database = cursor.fetchone()
    print(f"MySQL 连接成功：版本 {version}，数据库 {database}。")
