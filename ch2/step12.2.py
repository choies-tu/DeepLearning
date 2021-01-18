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

    def backward(self):
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
    def __call__(self, *inputs): # *:가변일 때 사용함
        xs = [x.data for x in inputs] #x의 복수형,람다식 선언, 리스트내포
        ys = self.forward(*xs)  #리스트 언팩 :리스트 원소를 낱개로 풀어전달
        print(ys)
        print(type(ys))
        if not isinstance(ys, tuple):
            ys = (ys,)   #각 함수의 리턴을 간단하게 함
        print(type(ys))
        outputs = [Variable(as_array(y)) for y in ys] ############# y값을 array로 바꿈!!!!

        for output in outputs:
            output.set_creator(self)    #출력변수에 창조자를 설정

        self.inputs = inputs  #입력변수 저장
        self.outputs = outputs
        #print(type(outputs))
        return outputs if len(outputs) > 1 else outputs[0] #????outputs[0]를 안쓰면
        #return outputs
        
    
    def forward(self, x):
        raise NotImplementedError()
    def backward(self, gy):
        raise NotImplementedError() #에러를 발생시키는 것
    
class Add(Function):
    def forward(self, x0,x1):
        y = x0 + x1  # y는 배열끼리의 합
        return y


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
    def backward(self, gy):
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
def exp(x):
    return Exp()(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

#xs=[Variable(np.array(2)), Variable(np.array(3))]
x0 = Variable(np.array([2.1,3,4]))
x1 = Variable(np.array([3,2.5,3]))
f = Add()
y = f(x0, x1)
#print(type(y))
print(y.data)








    

