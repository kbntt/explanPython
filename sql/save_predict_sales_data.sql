MERGE INTO TBL_EXP_PREDICT_SALES_RESULT T
USING (
  SELECT
      :brand_id        AS BRAND_ID
    , :store_id        AS STORE_ID
    , :yyyymm          AS YYYYMM
    , :pred_sales      AS SALES
    , :pred_cogs       AS COGS_AMT
    , :pred_labor      AS LABOR_COST
    , :pred_sell       AS SELL_EXPENSE
    , :pred_admin      AS ADMIN_EXPENSE
    , :pred_other      AS OTHER_GA_EXP
    , :pred_cost_sum   AS COST_SUM
    , :pred_op_profit  AS OPER_PROFIT
    , :pred_op_margin  AS OP_MARGIN
  FROM DUAL
) S
ON (
       T.BRAND_ID   = S.BRAND_ID
   AND T.STORE_ID   = S.STORE_ID
   AND T.YYYYMM     = S.YYYYMM
)
WHEN MATCHED THEN
  UPDATE SET T.SALES           = S.SALES
            ,T.COGS_AMT        = S.COGS_AMT
            ,T.LABOR_COST      = S.LABOR_COST
            ,T.SELL_EXPENSE    = S.SELL_EXPENSE
            ,T.ADMIN_EXPENSE   = S.ADMIN_EXPENSE
            ,T.OTHER_GA_EXP    = S.OTHER_GA_EXP
            ,T.COST_SUM        = S.COST_SUM
            ,T.OPER_PROFIT     = S.OPER_PROFIT
            ,T.OP_MARGIN       = S.OP_MARGIN
            ,T.UPDATE_DATE      = SYSDATE
            ,T.UPDATE_USER      = 'BATCH'
WHEN NOT MATCHED THEN
  INSERT (
         BRAND_ID          ,STORE_ID        ,YYYYMM           
        ,SALES             ,COGS_AMT        ,LABOR_COST       ,SELL_EXPENSE    ,ADMIN_EXPENSE 
        ,OTHER_GA_EXP      ,COST_SUM        ,OPER_PROFIT      ,OP_MARGIN 
        ,CREATE_USER       ,CREATE_DATE
  )
  VALUES (
         S.BRAND_ID        ,S.STORE_ID      ,S.YYYYMM         
        ,S.SALES           ,S.COGS_AMT      ,S.LABOR_COST     ,S.SELL_EXPENSE  ,S.ADMIN_EXPENSE 
        ,S.OTHER_GA_EXP    ,S.COST_SUM      ,S.OPER_PROFIT    ,S.OP_MARGIN 
        ,'BATCH'           ,SYSDATE
  )