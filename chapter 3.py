import math
import time
import numpy as np


# 找跨越中点的最大子数组
def find_max_crossing_subarray(A, low, mid, high):
    left_sum = -10000
    sum = 0

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


# 求解最大子数组的分治算法
def FIND_MAXIMUM_SUBARRAY(A, low, high):
    if high == low:
        return (low, high, A[low])
    else:
        mid = math.floor((low + high) / 2)
        (left_low, left_high, left_sum) = FIND_MAXIMUM_SUBARRAY(A, low, mid)
        (right_low, right_high, right_sum) = FIND_MAXIMUM_SUBARRAY(A, mid + 1, high)
        (cross_low, cross_high, cross_sum) = find_max_crossing_subarray(A, low, mid, high)
        if left_sum >= right_sum & left_sum >= cross_sum:
            return left_low, left_high, left_sum
        elif right_sum >= left_sum & right_sum >= cross_sum:
            return right_low, right_high, right_sum
        else:
            return cross_low, cross_high, cross_sum


# 求解最大子数组的暴力算法
def brute(A, low, high):
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


# 求解最大子数组的分治算法,当规模小于n时使用暴力算法
def find_max_crossing_subarray_mixed(A, low, mid, high, n):
    left_sum = -10000
    sum = 0
    if high - low <= n:
        return brute(A, low, high)
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


def FIND_MAXIMUM_SUBARRAY_mixed(A, low, high, n):
    if high == low:
        return (low, high, A[low])
    else:
        mid = math.floor((low + high) / 2)
        if high - low <= n:
            return brute(A, low, high)
        else:
            (left_low, left_high, left_sum) = FIND_MAXIMUM_SUBARRAY_mixed(A, low, mid, n)
            (right_low, right_high, right_sum) = FIND_MAXIMUM_SUBARRAY_mixed(A, mid + 1, high, n)
            (cross_low, cross_high, cross_sum) = find_max_crossing_subarray_mixed(A, low, mid, high, n)
        if left_sum >= right_sum & left_sum >= cross_sum:
            return left_low, left_high, left_sum
        elif right_sum >= left_sum & right_sum >= cross_sum:
            return right_low, right_high, right_sum
        else:
            return cross_low, cross_high, cross_sum


# 方阵乘法
def SQUARE_MATRIX_MULTIPLY(A, B):
    n = A.shape[0]
    C = np.zeros((n, n))
    print(C)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C


# 方阵乘法的简单递归算法
def SQUARE_MATRIX_MULTIPLY_RECURSIVE(A, B):
    n = A.shape[0]
    C = np.zeros((n, n))

    if n == 1:
        C[0, 0] = A[0, 0] * B[0, 0]
    else:
        C[:int(n // 2), :int(n // 2)] = SQUARE_MATRIX_MULTIPLY_RECURSIVE(A[:int(n // 2), :int(n // 2)],
                                                                         B[:int(n // 2), :int(n // 2)]) + \
                                        SQUARE_MATRIX_MULTIPLY_RECURSIVE(A[:int(n // 2), int(n // 2):],
                                                                         B[int(n // 2):, :int(n // 2)])
        C[:int(n // 2), int(n // 2):] = SQUARE_MATRIX_MULTIPLY_RECURSIVE(A[:int(n // 2), :int(n // 2)],
                                                                         B[:int(n // 2), int(n // 2):]) + \
                                        SQUARE_MATRIX_MULTIPLY_RECURSIVE(A[:int(n // 2), int(n // 2):],
                                                                         B[int(n // 2):, int(n // 2):])
        C[int(n // 2):, :int(n // 2)] = SQUARE_MATRIX_MULTIPLY_RECURSIVE(A[int(n // 2):, :int(n // 2)],
                                                                         B[:int(n // 2), :int(n // 2)]) + \
                                        SQUARE_MATRIX_MULTIPLY_RECURSIVE(A[int(n // 2):, int(n // 2):],
                                                                         B[int(n // 2):, :int(n // 2)])
        C[int(n // 2):, int(n // 2):] = SQUARE_MATRIX_MULTIPLY_RECURSIVE(A[int(n // 2):, :int(n // 2)],
                                                                         B[:int(n // 2), int(n // 2):]) + \
                                        SQUARE_MATRIX_MULTIPLY_RECURSIVE(A[int(n // 2):, int(n // 2):],
                                                                         B[int(n // 2):, int(n // 2):])
    return C
time1=time.time()
print(SQUARE_MATRIX_MULTIPLY_RECURSIVE(np.ones((32,32)),np.ones((32,32))))
time2=time.time()
print(time2-time1)
#  Strassen方法
def Strassen(A, B):
    n = A.shape[0]
    C = np.zeros((n, n))

    if n == 1:
        C[0, 0] = A[0, 0] * B[0, 0]
    else:
        S1 = B[:int(n // 2), int(n // 2):] - B[int(n // 2):, int(n // 2):]
        S2 = A[:int(n // 2), :int(n // 2)] + A[:int(n // 2), int(n // 2):]
        S3 = A[int(n // 2):, :int(n // 2)] + A[int(n // 2):, int(n // 2):]
        S4 = B[int(n // 2):, :int(n // 2)] - B[:int(n // 2), :int(n // 2)]
        S5 = A[:int(n // 2), :int(n // 2)] + A[int(n // 2):, int(n // 2):]
        S6 = B[:int(n // 2), :int(n // 2)] + B[int(n // 2):, int(n // 2):]
        S7 = A[:int(n // 2), int(n // 2):] - A[int(n // 2):, int(n // 2):]
        S8 = B[int(n // 2):, :int(n // 2)] + B[int(n // 2):, int(n // 2):]
        S9 = A[:int(n // 2), :int(n // 2)] - A[int(n // 2):, :int(n // 2)]
        S10 = B[:int(n // 2), :int(n // 2)] + B[:int(n // 2), int(n // 2):]
        P1=Strassen(A[:int(n // 2), :int(n // 2)],S1)
        P2=Strassen(S2,B[int(n // 2):, int(n // 2):])
        P3=Strassen(S3,B[:int(n // 2), :int(n // 2)])
        P4=Strassen(A[int(n // 2):, int(n // 2):],S4)
        P5=Strassen(S5,S6)
        P6=Strassen(S7,S8)
        P7=Strassen(S9,S10)
        C[:int(n // 2), :int(n // 2)]=P5+P4-P2+P6
        C[:int(n // 2), int(n // 2):]=P1+P2
        C[int(n // 2):, :int(n // 2)]=P3+P4
        C[int(n // 2):, int(n // 2):]=P5+P1-P3-P7
    return C
time3=time.time()
print(Strassen(np.ones((32,22)),np.ones((32,32))))
time4=time.time()
print(time4-time3)