class Variable:
    def __init__(self,data):
        self.data=data

import numpy as np

data = np.array(1.0) # 1x1 matrix
x = Variable(data)    
print(x.data)
