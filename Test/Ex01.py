import numpy as np


def main():
    # number => 주소이다
    # 왼쪽 데이터 lvalue(변수만)  오른쪽은 rvalue ( 데이터만 )
    # = 의 이름 assigment operator
    number1: int = 1
    number2: float = 1.5
    my_name: str = "KarL"
    number:tuple = (1,2,3,4)
    value = np.array([1, 2, 3, 4.0])
    print(value)
    print(number1)
    print(type(value))
    print(value)


if __name__ == "__main__":
    main()