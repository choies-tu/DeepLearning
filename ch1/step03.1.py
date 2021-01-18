import numpy as np
class Variable:
    def __init__(self,data):
        self.data=data

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
    def forward(self, x):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):  #오버로딩
        return x**2

class Exp(Function):
    def forward(self,x):
        return np.exp(x)  #numpy의 메소드 지수승 계산 함수

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

print(y.data)  #y=(e^(x^2))^2
print(y)  # y Vs  y.data?????
print(a)  # a가 무엇인지, 저장된 주소를 출력함
print(a.data)
print(b)
print(A)
# print(A.data) # Error
print(type(A))
print(B)
print(C)



