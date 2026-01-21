WITH TB_1 AS( 
    SELECT BRAND_ID
          ,STORE_ID
          ,MAX(YYYYMM) AS YYYYMM
    FROM   TBL_EXP_PREDICT_SALES_INFO
    WHERE  1 = 1
    AND    BRAND_ID = :brand_id
    AND    YYYYMM   = :yyyymm
    GROUP BY BRAND_ID
            ,STORE_ID 
)
,TB_2 AS( 
    SELECT  A.BRAND_ID      
           ,A.STORE_ID          
           ,A.YYYYMM
           ,A.GDP           
           ,A.INFLATION_RATE
           ,A.UNEMPLOYMENT  
           ,A.INTEREST_RATE 
           ,A.CCSI          
           ,A.SALES
           ,A.COST_RATIO
           ,A.COGS_AMT
           ,A.LABOR_COST
           ,A.SELL_EXPENSE
           ,A.ADMIN_EXPENSE
           ,A.OTHER_GA_EXP
           ,A.STORE_STATE_OPEN 
           ,A.STORE_STATE_CLOSE
           ,A.STORE_STATE_ALL 
    FROM    TBL_EXP_PREDICT_SALES_INFO A
            JOIN TB_1 B
                ON  1 = 1
                AND A.BRAND_ID   = B.BRAND_ID
                AND A.STORE_ID   = B.STORE_ID
                AND A.YYYYMM     = B.YYYYMM
    WHERE  1 = 1
)
SELECT A.BRAND_ID      
      ,A.STORE_ID          
      ,A.YYYYMM
      , EXTRACT(YEAR  FROM TO_DATE(A.YYYYMM, 'YYYY-MM'))  AS YEAR
      , EXTRACT(MONTH FROM TO_DATE(A.YYYYMM, 'YYYY-MM'))  AS MONTH
      ,A.GDP           
      ,A.INFLATION_RATE
      ,A.UNEMPLOYMENT  
      ,A.INTEREST_RATE 
      ,A.CCSI          
      ,A.SALES
      ,A.COST_RATIO
      ,A.COGS_AMT
      ,A.LABOR_COST
      ,A.SELL_EXPENSE
      ,A.ADMIN_EXPENSE
      ,A.OTHER_GA_EXP
      ,A.STORE_STATE_OPEN 
      ,A.STORE_STATE_CLOSE
      ,A.STORE_STATE_ALL 
FROM   (
        SELECT A.BRAND_ID      
              ,A.STORE_ID          
              ,TO_CHAR(ADD_MONTHS(TO_DATE(A.YYYYMM,'YYYY-MM'),6),'YYYY-MM') AS YYYYMM
              ,A.GDP           
              ,A.INFLATION_RATE
              ,A.UNEMPLOYMENT  
              ,A.INTEREST_RATE 
              ,A.CCSI          
              ,A.SALES 
              ,A.COST_RATIO
              ,A.COGS_AMT
              ,A.LABOR_COST
              ,A.SELL_EXPENSE
              ,A.ADMIN_EXPENSE
              ,A.OTHER_GA_EXP
              ,A.STORE_STATE_OPEN 
              ,A.STORE_STATE_CLOSE
              ,A.STORE_STATE_ALL 
        FROM TB_2 A
        UNION ALL
        SELECT A.BRAND_ID      
              ,A.STORE_ID          
              ,TO_CHAR(ADD_MONTHS(TO_DATE(A.YYYYMM,'YYYY-MM'),5),'YYYY-MM') AS YYYYMM
              ,A.GDP           
              ,A.INFLATION_RATE
              ,A.UNEMPLOYMENT  
              ,A.INTEREST_RATE 
              ,A.CCSI          
              ,A.SALES 
              ,A.COST_RATIO
              ,A.COGS_AMT
              ,A.LABOR_COST
              ,A.SELL_EXPENSE
              ,A.ADMIN_EXPENSE
              ,A.OTHER_GA_EXP
              ,A.STORE_STATE_OPEN 
              ,A.STORE_STATE_CLOSE
              ,A.STORE_STATE_ALL 
        FROM TB_2 A
        UNION ALL
        SELECT A.BRAND_ID      
              ,A.STORE_ID          
              ,TO_CHAR(ADD_MONTHS(TO_DATE(A.YYYYMM,'YYYY-MM'),4),'YYYY-MM') AS YYYYMM
              ,A.GDP           
              ,A.INFLATION_RATE
              ,A.UNEMPLOYMENT  
              ,A.INTEREST_RATE 
              ,A.CCSI          
              ,A.SALES 
              ,A.COST_RATIO
              ,A.COGS_AMT
              ,A.LABOR_COST
              ,A.SELL_EXPENSE
              ,A.ADMIN_EXPENSE
              ,A.OTHER_GA_EXP
              ,A.STORE_STATE_OPEN 
              ,A.STORE_STATE_CLOSE
              ,A.STORE_STATE_ALL 
        FROM TB_2 A
        UNION ALL
        SELECT A.BRAND_ID      
              ,A.STORE_ID          
              ,TO_CHAR(ADD_MONTHS(TO_DATE(A.YYYYMM,'YYYY-MM'),3),'YYYY-MM') AS YYYYMM
              ,A.GDP           
              ,A.INFLATION_RATE
              ,A.UNEMPLOYMENT  
              ,A.INTEREST_RATE 
              ,A.CCSI          
              ,A.SALES 
              ,A.COST_RATIO
              ,A.COGS_AMT
              ,A.LABOR_COST
              ,A.SELL_EXPENSE
              ,A.ADMIN_EXPENSE
              ,A.OTHER_GA_EXP
              ,A.STORE_STATE_OPEN 
              ,A.STORE_STATE_CLOSE
              ,A.STORE_STATE_ALL 
        FROM TB_2 A
        UNION ALL
        SELECT A.BRAND_ID      
              ,A.STORE_ID          
              ,TO_CHAR(ADD_MONTHS(TO_DATE(A.YYYYMM,'YYYY-MM'),2),'YYYY-MM') AS YYYYMM
              ,A.GDP           
              ,A.INFLATION_RATE
              ,A.UNEMPLOYMENT  
              ,A.INTEREST_RATE 
              ,A.CCSI          
              ,A.SALES 
              ,A.COST_RATIO
              ,A.COGS_AMT
              ,A.LABOR_COST
              ,A.SELL_EXPENSE
              ,A.ADMIN_EXPENSE
              ,A.OTHER_GA_EXP
              ,A.STORE_STATE_OPEN 
              ,A.STORE_STATE_CLOSE
              ,A.STORE_STATE_ALL 
        FROM TB_2 A
        UNION ALL
        SELECT A.BRAND_ID      
              ,A.STORE_ID          
              ,TO_CHAR(ADD_MONTHS(TO_DATE(A.YYYYMM,'YYYY-MM'),1),'YYYY-MM') AS YYYYMM
              ,A.GDP           
              ,A.INFLATION_RATE
              ,A.UNEMPLOYMENT  
              ,A.INTEREST_RATE 
              ,A.CCSI          
              ,A.SALES 
              ,A.COST_RATIO
              ,A.COGS_AMT
              ,A.LABOR_COST
              ,A.SELL_EXPENSE
              ,A.ADMIN_EXPENSE
              ,A.OTHER_GA_EXP
              ,A.STORE_STATE_OPEN 
              ,A.STORE_STATE_CLOSE
              ,A.STORE_STATE_ALL 
        FROM TB_2 A
) A