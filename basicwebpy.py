class Calculator:
    # __init__은 필수다
    def __init__(self):
        self.result = 0
    def add(self, num):
        self.result += num
        return self.result
cal1 = Calculator()
cal2 = Calculator()
print("Cumulative1:", cal1.add(3), cal1.add(4))
print("Cumulative2:", cal2.add(19), cal2.add(11))



class FourCal:
    def setdata(self, first, second):
        self.first = first
        self.second = second
    def add(self):
        result = self.first + self.second
        return result
    def mul(self):
        result = self.first * self.second
        return result
    def sub(self):
        result = self.first - self.second
        return result       
    def div(self):
        result = self.first / self.second
        return result        
a = FourCal()
b = FourCal()
a.setdata(4,2)
b.setdata(3,8)
print("Method Inside Class:", a.add(), a.mul(), a.sub(), a.div())
print("Method Inside Class:", b.add(), b.mul(), b.sub(), b.div())



class FiveCal:
    def __init__(self, first, second):
        self.first = first
        self.second = second
    def setdata(self, first, second):
        self.first = first
        self.second = second
    def add(self):
        result = self.first + self.second
        return result
    def mul(self):
        result = self.first * self.second
        return result
    def sub(self):
        result = self.first - self.second
        return result
    def div(self):
        if self.second == 0:
            return 0
        else:
            return self.first / self.second
        result = self.first / self.second
        return result
c=FiveCal(4,2)
print("Added __init__:", c.first, c.second, c.add(), c.div())



class MoreFourCal(FourCal):
    def __init__(self, first, second):
        self.first = first
        self.second = second
    def pow(self):
        result = self.first ** self.second
        return result
d = MoreFourCal(4, 2)
print( "Multiplier of Multiplier:", d.pow() )



class Family:
    lastname = "KIM"
print( "Family:", Family.lastname)

Family.lastname = "PAK"
e = Family()
print( "Adopted SAD:", e.lastname )



def apply_one(f):
    return f(1)
def triple(x):
    return x*3
result = apply_one(triple)
print("first-class function:", result)



def trace(func):                             # 호출할 함수를 매개변수로 받음
    def wrapper():
        print(func.__name__, '함수 시작')    # __name__으로 함수 이름 출력
        func()                               # 매개변수로 받은 함수를 호출
        print(func.__name__, '함수 끝')
    return wrapper                           # wrapper 함수 반환
 
@trace    # @데코레이터
def hello():
    print('hello')
 
@trace    # @데코레이터
def world():
    print('world')
 
hello()    # 함수를 그대로 호출
world()    # 함수를 그대로 호출

# hello 함수 시작
# hello
# hello 함수 끝
# world 함수 시작
# world
# world 함수 끝
