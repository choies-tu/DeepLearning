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

def numerical_diff(f,x,eps=1e-4):  #1e-4 : 1*10^(-4)
    x0 = Variable(x.data-eps)
    x1 = Variable(x.data+eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data-y0.data)/(2*eps)


f = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f,x)
print(dy)

def f(x):  #합성함수인 경우 선언이 필요함
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

x = Variable(np.array(0.5))
print(x.data)
print(f)
dy = numerical_diff(f,x)
print(dy)

# dz = numerical_diff(Square(Exp(Square())),x) #error 발생

#error발생
##A = Square()
##B = Exp()
##dz = numerical_diff(B(A()),x)
##print(dz)









    

