-- getPredictSalesData_01.sql (예측용)
-- 목표: 예측 시점(YYYY-MM)에 대해 모델 입력 컬럼을 한 번에 제공
--  - 학습 테이블: TBL_EXP_PREDICT_SALES_INFO
--  - Lag 피처: SALES_LAG_1(전월), SALES_LAG_12(전년동월)
--  - TIME_IDX: YYYY*12 + MM
--
-- params:
--   :brand_id  (예: obong)
--   :yyyymm    (예: 2025-10)

WITH base AS (
    SELECT  BRAND_ID
          , STORE_ID
          , YYYYMM
          , SUBSTR(YYYYMM, 1, 4) AS YEAR
          , SUBSTR(YYYYMM, 6, 2) AS MONTH
          , GDP
          , INFLATION_RATE
          , UNEMPLOYMENT
          , INTEREST_RATE
          , CCSI
          , AVG_SALES_3M
          , STORE_STATE_OPEN
          , STORE_STATE_CLOSE
          , STORE_STATE_ALL
      FROM TBL_EXP_PREDICT_SALES_INFO
     WHERE BRAND_ID = :brand_id
       AND YYYYMM   = :yyyymm
),
lag1 AS (
    SELECT  BRAND_ID, STORE_ID, YYYYMM, SALES
      FROM TBL_EXP_PREDICT_SALES_INFO
     WHERE BRAND_ID = :brand_id
),
lag12 AS (
    SELECT  BRAND_ID, STORE_ID, YYYYMM, SALES
      FROM TBL_EXP_PREDICT_SALES_INFO
     WHERE BRAND_ID = :brand_id
)
SELECT  B.BRAND_ID
      , B.STORE_ID
      , B.YYYYMM
      , TO_NUMBER(B.YEAR)  AS YEAR
      , TO_NUMBER(B.MONTH) AS MONTH
      , B.GDP
      , B.INFLATION_RATE
      , B.UNEMPLOYMENT
      , B.INTEREST_RATE
      , B.CCSI
      , B.STORE_STATE_OPEN
      , B.STORE_STATE_CLOSE
      , B.STORE_STATE_ALL
      , (TO_NUMBER(B.YEAR) * 12 + TO_NUMBER(B.MONTH)) AS TIME_IDX
      , NVL(C.SALES, 0)  AS SALES_LAG_1
      , NVL(D.SALES, 0)  AS SALES_LAG_12
FROM base B
LEFT JOIN lag1 C
       ON C.BRAND_ID = B.BRAND_ID
      AND C.STORE_ID = B.STORE_ID
      AND C.YYYYMM   = TO_CHAR(ADD_MONTHS(TO_DATE(B.YYYYMM, 'YYYY-MM'), -1), 'YYYY-MM')
LEFT JOIN lag12 D
       ON D.BRAND_ID = B.BRAND_ID
      AND D.STORE_ID = B.STORE_ID
      AND D.YYYYMM   = TO_CHAR(ADD_MONTHS(TO_DATE(B.YYYYMM, 'YYYY-MM'), -12), 'YYYY-MM')
ORDER BY B.STORE_ID