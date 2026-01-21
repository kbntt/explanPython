import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime

파일경로 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
데이터 = pd.read_csv(파일경로)
데이터.head()
독립 = 데이터[['온도']]
종속 = 데이터[['판매량']]
print(독립.shape, 종속.shape)
# 모델을 만듭니다
X = tf.keras.layers.Input(shape=[1])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X,Y)
model.compile(loss = 'mse')
# 시작 시간
start_time = datetime.now()
print("시작시간:", start_time)
model.fit(독립, 종속, epochs=10, verbose=0)
# 종료 시간
end_time = datetime.now()
print("종료시간:", end_time)

# 모델을 이용합니다.
print(model.predict(독립))