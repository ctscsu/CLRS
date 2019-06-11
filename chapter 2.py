import numpy as np


# 2.1 插入排序
# 插入排序非升序
def INSERTION_SORT(A, p, r):
    for j in range(p, r + 1):
        key = A[j]
        i = j - 1
        while i >= p and A[i] > key:
            A[i + 1] = A[i]
            i = i - 1
        A[i + 1] = key


if __name__ == '__main__':
    A = [3, 4, 6, 2, 1]
    INSERTION_SORT(A, 0, 4)
    print(A)


# 2.1-2 插入排序非降序
def INSERTION_SORT_Non_descending(A, p, r):
    for j in range(p, r + 1):
        key = A[j]
        i = j - 1
        while i >= 0 and A[i] < key:
            A[i + 1] = A[i]
            i = i - 1
        A[i + 1] = key


if __name__ == '__main__':
    A = [3, 4, 6, 2, 1]
    INSERTION_SORT_Non_descending(A, 0, 4)
    print(A)


# 2.1-3 线性查找
def linear_search(A, a):
    for i in range(len(A)):
        if A[i] == a:
            return i
    return None


if __name__ == '__main__':
    A = [3, 4, 6, 2, 1]
    print(linear_search(A, 4))
    print(linear_search(A, 5))


# 2.1-4 二进制数相加
def ADD(A, B, C):
    temp = 0
    for i in range(len(A)):
        C.append((A[i] + B[i] + temp) % 2)
        temp = (A[i] + B[i] + temp) // 2
    C.append(temp)


if __name__ == '__main__':
    A = [1, 0, 1, 1, 1]
    B = [1, 0, 0, 1, 1]
    C = []
    ADD(A, B, C)
    print(C)


# 2.2 分析算法
# 2.2-2 选择算法
def SELECT_SORT(A):
    for i in range(len(A) - 1):
        min = i
        for j in range(i + 1, len(A)):
            if A[j] < A[min]:
                min = j
        A[i], A[min] = A[min], A[i]
    return A


if __name__ == '__main__':
    A = [3, 4, 6, 2, 1, 7, 4]
    SELECT_SORT(A)
    print(A)


# 2.3 设计算法
# 使用哨兵合并两个数组
def MERGE_with_guard(A, p, q, r):
    n1 = q - p + 1
    n2 = r - q
    L = np.zeros(n1 + 1)
    R = np.zeros(n2 + 1)
    for i in range(n1):
        L[i] = A[p + i]
    for j in range(n2):
        R[j] = A[q + j + 1]
    i = 0
    j = 0
    L[n1] = float('inf')
    R[n2] = float('inf')
    for k in range(p, r + 1):
        if L[i] <= R[j]:
            A[k] = L[i]
            i += 1
        else:
            A[k] = R[j]
            j += 1


if __name__ == '__main__':
    A = [1, 3, 5, 6, 2, 2, 4, 8, 9]
    MERGE_with_guard(A, 0, 3, 8)
    print(A)


# 有哨兵的归并排序
def MERGESORT(A, p, r):
    if p < r:
        q = (p + r) // 2
        MERGESORT(A, p, q)
        MERGESORT(A, q + 1, r)
        MERGE_with_guard(A, p, q, r)


if __name__ == '__main__':
    A = [1, 3, 5, 6, 2, 2, 4, 8, 9]
    MERGESORT(A, 0, 8)
    print(A)


# 2.3-2 不使用哨兵合并两个数组

def MERGE_without_guard(A, p, q, r):
    n1 = q - p + 1
    n2 = r - q
    L = np.zeros(n1)
    R = np.zeros(n2)
    for i in range(n1):
        L[i] = A[p + i]
    for j in range(n2):
        R[j] = A[q + j + 1]
    i = 0
    j = 0
    for k in range(p, r + 1):
        if i == n1:
            A[k] = R[j]
            j = j + 1
        elif j == n2:
            A[k] = L[i]
            i = i + 1
        elif L[i] <= R[j]:
            A[k] = L[i]
            i = i + 1
        else:
            A[k] = R[j]
            j = j + 1


if __name__ == '__main__':
    A = [1, 3, 5, 6, 2, 2, 4, 8, 9]
    MERGE_without_guard(A, 0, 3, 8)
    print(A)


# 无哨兵的归并排序
def MERGESORT_1(A, p, r):
    if p < r:
        q = (p + r) // 2
        MERGESORT_1(A, p, q)
        MERGESORT_1(A, q + 1, r)
        MERGE_without_guard(A, p, q, r)


if __name__ == '__main__':
    A = [1, 3, 5, 6, 2, 2, 4, 8, 9]
    MERGESORT_1(A, 0, 8)
    print(A)


# 2.3-5二分查找
def Binarysearch(A, v):
    low = 0
    high = len(A) - 1
    while low <= high:
        mid = (low + high) // 2
        if A[mid] == v:
            return mid
        elif A[mid] < v:
            low = mid + 1
        else:
            high = mid - 1
    return None


if __name__ == '__main__':
    A = [1, 3, 5, 6, 8]
    print(Binarysearch(A, 5))
    print(Binarysearch(A, 4))


# 2.3-7 nlog(n)时间判定S中是否存在和为x的两个元素
def PAIREXIST(S, x):
    MERGESORT(S, 0, len(S) - 1)
    for i in range(len(S)):
        if Binarysearch(S, x - S[i]):
            return True
        else:
            return False


if __name__ == '__main__':
    A = [1, 3, 5, 6, 8]
    print(PAIREXIST(A, 5))
    print(PAIREXIST(A, 4))


# 思考题

# 2.1 在归并排序中对小数组采用插入排序
def mixed_sort(A, p, r):
    if p >= r:
        return
    if r - p < 20:
        INSERTION_SORT(A, p, r)
    else:
        q = int((p + r) // 2)
        mixed_sort(A, p, q)
        mixed_sort(A, q + 1, r)
        MERGE_with_guard(A, p, q, r)


if __name__ == '__main__':
    A = [1, 3, 5, 6, 8, 6, 11, 2, 3, 7] * 10
    mixed_sort(A, 0, 99)
    print(A)


# 2-2 冒泡排序
def BUBBLE_SORT(A):
    for i in range(len(A)):
        for j in range(len(A) - 1, i, -1):
            if A[j] < A[j - 1]:
                A[j], A[j - 1] = A[j - 1], A[j]


if __name__ == '__main__':
    A = [1, 3, 5, 6, 8, 6, 11, 2, 3, 7]
    BUBBLE_SORT(A)
    print(A)


# 2-4 用归并排序计算数组中逆序对数量
def MERGE_without_guard_inversions(A, p, q, r):
    n1 = q - p + 1
    n2 = r - q
    L = np.zeros(n1)
    R = np.zeros(n2)
    for i in range(n1):
        L[i] = A[p + i]
    for j in range(n2):
        R[j] = A[q + j + 1]
    i = 0
    j = 0
    inversions = 0
    for k in range(p, r + 1):
        if i == n1:
            A[k] = R[j]
            j = j + 1
        elif j == n2:
            A[k] = L[i]
            i = i + 1
        elif L[i] <= R[j]:
            A[k] = L[i]
            i = i + 1
        else:
            A[k] = R[j]
            j = j + 1
            inversions += n1 - i
    return inversions


def MERGESORT_inversions(A, p, r):
    if p < r:
        inversions = 0
        q = (p + r) // 2
        inversions += MERGESORT_inversions(A, p, q)
        inversions += MERGESORT_inversions(A, q + 1, r)
        inversions += MERGE_without_guard_inversions(A, p, q, r)
        return inversions
    else:
        return 0


if __name__ == '__main__':
    A = [1]
    B = [6, 5, 4, 3, 2, 1]
    print(MERGESORT_inversions(A, 0, 0))
    print(MERGESORT_inversions(B, 0, 5))
