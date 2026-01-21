SELECT BRAND_ID      
      ,STORE_ID      
      ,YYYYMM    
      ,SUBSTR(YYYYMM, 1, 4) AS YEAR
      ,SUBSTR(YYYYMM, 6, 2) AS MONTH    
      ,GDP           
      ,INFLATION_RATE
      ,UNEMPLOYMENT  
      ,INTEREST_RATE 
      ,CCSI          
      ,SALES   
      ,COGS_AMT
      ,LABOR_COST
      ,SELL_EXPENSE
      ,ADMIN_EXPENSE
      ,OTHER_GA_EXP
      ,STORE_STATE_OPEN 
      ,STORE_STATE_CLOSE
      ,STORE_STATE_ALL 
FROM   TBL_EXP_PREDICT_SALES_INFO
WHERE  1 = 1
AND    BRAND_ID = :brand_id
AND    ( :yyyymm IS NULL OR YYYYMM = :yyyymm )
ORDER BY YYYYMM, STORE_ID