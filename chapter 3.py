import numpy as np
import time


# 4.1 最大子数组问题
# 找跨越中点的最大子数组
def FIND_MAX_CROSSING_SUBARRAY(A, low, mid, high):
    left_sum = -float('inf')
    sum = 0

    for i in range(mid, low - 1, -1):
        sum = sum + A[i]
        if sum > left_sum:
            left_sum = sum
            max_left = i
    right_sum = -float('inf')
    sum = 0
    for j in range(mid + 1, high + 1):
        sum = sum + A[j]
        if sum > right_sum:
            right_sum = sum
            max_right = j
    return max_left, max_right, left_sum + right_sum


if __name__ == '__main__':
    A = [-1, 3, 5, 6, 2, 2, 4, -8, 9]
    print(FIND_MAX_CROSSING_SUBARRAY(A, 0, 4, 8))


# 求解最大子数组的分治算法
def FIND_MAXIMUM_SUBARRAY(A, low, high, ):
    if high == low:
        return (low, high, A[low])
    else:
        mid = (low + high) // 2
        (left_low, left_high, left_sum) = FIND_MAXIMUM_SUBARRAY(A, low, mid)
        (right_low, right_high, right_sum) = FIND_MAXIMUM_SUBARRAY(A, mid + 1, high)
        (cross_low, cross_high, cross_sum) = FIND_MAX_CROSSING_SUBARRAY(A, low, mid, high)
        if left_sum >= right_sum & left_sum >= cross_sum:
            return left_low, left_high, left_sum
        elif right_sum >= left_sum & right_sum >= cross_sum:
            return right_low, right_high, right_sum
        else:
            return cross_low, cross_high, cross_sum


if __name__ == '__main__':
    A = [-1, 3, 5, 6, 2, 2, 4, -8, 9]
    B = [-4, -1, -2, -2, -7]
    print(FIND_MAXIMUM_SUBARRAY(A, 0, 8))
    print(FIND_MAXIMUM_SUBARRAY(B, 0, 4))


# 4.1-2求解最大子数组的暴力算法
def BRUTE(A, low, high):
    left = 0
    right = 0
    sum = -float('inf')
    for i in range(low, high + 1):
        current_sum = 0
        for j in range(i, high + 1):
            current_sum += A[j]
        if sum < current_sum:
            sum = current_sum
            left = i
            right = j
    return (left, right, sum)


if __name__ == '__main__':
    A = [-1, 3, 5, 6, 2, 2, 4, -8, 9]
    print(BRUTE(A, 0, 8))


# 求解最大子数组的分治算法,当规模小于n时使用暴力算法
def FIND_MAX_CROSSING_SUBARRAY_MIXED(A, low, mid, high, n):
    left_sum = -10000
    sum = 0
    if high - low <= n:
        return BRUTE(A, low, high)
    else:
        for i in range(mid, low - 1, -1):
            sum = sum + A[i]
            if sum > left_sum:
                left_sum = sum
                max_left = i
        right_sum = -10000
        sum = 0
        for j in range(mid + 1, high + 1):
            sum = sum + A[j]
            if sum > right_sum:
                right_sum = sum
                max_right = j
        return max_left, max_right, left_sum + right_sum

def FIND_MAXIMUM_SUBARRAY_MIXED(A, low, high, n):
    if high == low:
        return (low, high, A[low])
    else:
        mid = (low + high) // 2
        if high - low <= n:
            return BRUTE(A, low, high)
        else:
            (left_low, left_high, left_sum) = FIND_MAXIMUM_SUBARRAY_MIXED(A, low, mid, n)
            (right_low, right_high, right_sum) = FIND_MAXIMUM_SUBARRAY_MIXED(A, mid + 1, high, n)
            (cross_low, cross_high, cross_sum) = FIND_MAX_CROSSING_SUBARRAY_MIXED(A, low, mid, high, n)
        if left_sum >= right_sum & left_sum >= cross_sum:
            return left_low, left_high, left_sum
        elif right_sum >= left_sum & right_sum >= cross_sum:
            return right_low, right_high, right_sum
        else:
            return cross_low, cross_high, cross_sum


if __name__ == '__main__':
    A = [-1, 3, 5, 6, 2, 2, 4, -8, 7] * 10
    print(FIND_MAXIMUM_SUBARRAY_MIXED(A, 0, 89, 20))


# 方阵乘法
def SQUARE_MATRIX_MULTIPLY(A, B):
    n = A.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

if __name__ == '__main__':
    A = B=np.array([[1,2],[3,4]])
    print(SQUARE_MATRIX_MULTIPLY(A, B))


# 方阵乘法的简单递归算法
def SQUARE_MATRIX_MULTIPLY_RECURSIVE(A, B):
    n = A.shape[0]
    C = np.zeros((n, n))

    if n == 1:
        C[0, 0] = A[0, 0] * B[0, 0]
    else:
        C[:(n // 2), :(n // 2)] = SQUARE_MATRIX_MULTIPLY_RECURSIVE(A[:(n // 2), :(n // 2)],
                                                                         B[:(n // 2), :(n // 2)]) + \
                                        SQUARE_MATRIX_MULTIPLY_RECURSIVE(A[:(n // 2), (n // 2):],
                                                                         B[(n // 2):, :(n // 2)])
        C[:(n // 2), (n // 2):] = SQUARE_MATRIX_MULTIPLY_RECURSIVE(A[:(n // 2), :(n // 2)],
                                                                         B[:(n // 2), (n // 2):]) + \
                                        SQUARE_MATRIX_MULTIPLY_RECURSIVE(A[:(n // 2), (n // 2):],
                                                                         B[(n // 2):, (n // 2):])
        C[(n // 2):, :(n // 2)] = SQUARE_MATRIX_MULTIPLY_RECURSIVE(A[(n // 2):, :(n // 2)],
                                                                         B[:(n // 2), :(n // 2)]) + \
                                        SQUARE_MATRIX_MULTIPLY_RECURSIVE(A[(n // 2):, (n // 2):],
                                                                         B[(n // 2):, :(n // 2)])
        C[(n // 2):, (n // 2):] = SQUARE_MATRIX_MULTIPLY_RECURSIVE(A[(n // 2):, :(n // 2)],
                                                                         B[:(n // 2), (n // 2):]) + \
                                        SQUARE_MATRIX_MULTIPLY_RECURSIVE(A[(n // 2):, (n // 2):],
                                                                         B[(n // 2):, (n // 2):])
    return C

if __name__ == '__main__':
    A = B=np.array([[1,2],[3,4]])
    print(SQUARE_MATRIX_MULTIPLY_RECURSIVE(A, B))
# time1=time.time()
# print(SQUARE_MATRIX_MULTIPLY_RECURSIVE(np.ones((32,32)),np.ones((32,32))))
# time2=time.time()
# print(time2-time1)
#  Strassen方法
def Strassen(A, B):
    n = A.shape[0]
    C = np.zeros((n, n))

    if n == 1:
        C[0, 0] = A[0, 0] * B[0, 0]
    else:
        S1 = B[:(n // 2), (n // 2):] - B[(n // 2):, (n // 2):]
        S2 = A[:(n // 2), :(n // 2)] + A[:(n // 2), (n // 2):]
        S3 = A[(n // 2):, :(n // 2)] + A[(n // 2):, (n // 2):]
        S4 = B[(n // 2):, :(n // 2)] - B[:(n // 2), :(n // 2)]
        S5 = A[:(n // 2), :(n // 2)] + A[(n // 2):, (n // 2):]
        S6 = B[:(n // 2), :(n // 2)] + B[(n // 2):, (n // 2):]
        S7 = A[:(n // 2), (n // 2):] - A[(n // 2):, (n // 2):]
        S8 = B[(n // 2):, :(n // 2)] + B[(n // 2):, (n // 2):]
        S9 = A[:(n // 2), :(n // 2)] - A[(n // 2):, :(n // 2)]
        S10 = B[:(n // 2), :(n // 2)] + B[:(n // 2), (n // 2):]
        P1 = Strassen(A[:(n // 2), :(n // 2)], S1)
        P2 = Strassen(S2, B[(n // 2):, (n // 2):])
        P3 = Strassen(S3, B[:(n // 2), :(n // 2)])
        P4 = Strassen(A[(n // 2):, (n // 2):], S4)
        P5 = Strassen(S5, S6)
        P6 = Strassen(S7, S8)
        P7 = Strassen(S9, S10)
        C[:(n // 2), :(n // 2)] = P5 + P4 - P2 + P6
        C[:(n // 2), (n // 2):] = P1 + P2
        C[(n // 2):, :(n // 2)] = P3 + P4
        C[(n // 2):, (n // 2):] = P5 + P1 - P3 - P7
    return C
if __name__ == '__main__':
    A = B=np.array([[1,2],[3,4]])
    print(Strassen(A, B))
# time3=time.time()
# print(Strassen(np.ones((32,22)),np.ones((32,32))))
# time4=time.time()
# print(time4-time3)
#4.2-7三次实数乘法计算复数乘积
def PRODECT():
    a,b,c,d=map(int,input('输入a,b,c,d空格隔开:').split())
    x=(a+b)*(c+d)
    y=a*c
    z=b*d
    return str(y-z)+'+'+str(x-y-z)+"i"
if __name__ == '__main__':
    print(PRODECT())

