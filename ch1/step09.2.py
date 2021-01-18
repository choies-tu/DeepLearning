import numpy as np
class Variable:
    def __init__(self,data):
        #np.array가 아닌 형식의 input에 대한 error처리하기
        if data is not None:
            if not isinstance(data, np.ndarray):
                print(type(data))
                raise TypeError("{}는 지원하지 않습니다. ".format(type(data)))
                
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):  # func으로 받은 값을 creator에 기록 
        self.creator = func

    def backward(self):  #자동 역전파할 수 있도록 함
        if self.grad is None:
            self.grad = np.ones_like(self.data)  ##grad에 1로 모두 초기화
            
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            print(x.grad)

            if x.creator is not None:
                funcs.append(x.creator)
        
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)        
        output = Variable(as_array(y)) ############# y값을 array로 바꿈!!!!
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

def square(x):
    return Square()(x)


class Square3(Function):
    def forward(self, x):  #오버로딩
        return x**3
    def backward(self, gy): #미분함수
        x = self.input.data        
        gx = 3*(x**2)*gy
        return gx
def square3(x):
    return Square3()(x)    

class Exp(Function):
    def forward(self,x):
        return np.exp(x)  #numpy의 메소드 지수승 계산 함수
    def backward(self, gy):
        x=self.input.data
        gx = np.exp(x)*gy
        return gx
def exp(x):               #f(x)처럼 사용할 수 있도록 함
    return Exp()(x)

def as_array(x):         #입력함수가 스칼라 일때 array로 변환해줌
    if np.isscalar(x):
        return np.array(x)
    return x

x = Variable(np.array(2.0))
#x = Variable(2.0)   # error 발생 
a = square(x)
b = square3(a)
#b = exp(a)
y = square(b)

#y.grad = np.array(1.0)
y.backward()
print(x.grad)









    

