import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 这里的data type 必须是浮点型
weights = np.array([[115.],[140.],[175.]])
print(weights)

scaler = MinMaxScaler()
scaled_weight = scaler.fit_transform(weights)

print(scaled_weight)
