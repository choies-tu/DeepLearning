import numpy as np
class Variable:
    def __init__(self,data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):  # func으로 받은 값을 creator에 기록 
        self.creator = func

    def backward(self):
        f = self.creator    #1. 함수가져오기
        if f is not None:
            x = f.input     #2. 입력값 가져오기           
            x.grad = f.backward(self.grad)  #3. 함수의 backward 메서드 호출
            print(x.grad)
            x.backward()    # 재귀, 하나앞 변수의 backward 메서드 호출

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)    #출력변수에 창조자를 설정
        self.input = input  #입력변수 저장
        self.output = output
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
B = Square3()
C = Square()

x = Variable(np.array(2.0))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)

##C1 = y.creator
##b1=C1.input
##b1.grad = C1.backward(y.grad)
##
##
##B1 = b1.creator
##a1=B1.input
##a1.grad = B1.backward(b1.grad)
##
##A1 = a1.creator
##x1=A1.input
##x1.grad = A1.backward(a1.grad)

y.backward()
##print(b.grad)
##print(a.grad)
print(x.grad)








    

