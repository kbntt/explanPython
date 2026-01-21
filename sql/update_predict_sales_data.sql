
UPDATE TBL_EXP_PREDICT_SALES_RESULT
SET    COST_RATIO = :cost_ratio
WHERE  1 = 1
AND     BRAND_ID = :brand_id
AND     STORE_ID = :store_id
AND     YYYYMM   = :yyyymm
