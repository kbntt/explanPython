SELECT AMOUNT
FROM   TBL_EXP_FIN_ANAL_STATUS
WHERE  BUYER_ID     = :buyer_id
  AND  YEAR         = :year
  AND  MONTH        = :month
  AND  ACCOUNT_CODE = :account_code
ORDER BY AMOUNT
