import numpy as np
class Variable:
    def __init__(self,data):
        self.data = data
        self.grad = None

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input  #입력변수 저장 
        return output
    def forward(self, x):
        raise NotImplementedError()
    def backward(self, gy):
        raise NotImplementedError() #에러를 발생시키는 것 

class Square(Function):
    def forward(self, x):  #오버로딩
        return x**2
    def backward(self, gy):
        x = self.input.data        
        gx = 2*x*gy
        return gx

class Square3(Function):
    def forward(self, x):  #오버로딩
        return x**3
    def backward(self, gy):
        x = self.input.data        
        gx = 3*(x**2)*gy
        return gx
    

class Exp(Function):
    def forward(self,x):
        return np.exp(x)  #numpy의 메소드 지수승 계산 함수
    def backward(self, gy):
        x=self.input.data
        gx = np.exp(x)*gy
        return gx

A = Square()
#B = Exp()
B = Square3()
C = Square()

x = Variable(np.array(2.0))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
print(b.grad)
a.grad = B.backward(b.grad)
print(a.grad)
x.grad = A.backward(a.grad)
print(x.grad)








    

