import numpy as np


# 2.1 插入排序非升序
def INSERTION_SORT(A):
    for j in range(len(A)):
        key = A[j]
        i = j - 1
        while i >= 0 and A[i] > key:
            A[i + 1] = A[i]
            i = i - 1
        A[i + 1] = key


if __name__ == '__main__':
    A = [3, 4, 6, 2, 1]
    INSERTION_SORT(A)
    print(A)


# 2.1-2 插入排序非降序
def INSERTION_SORT_Non_descending(A):
    for j in range(len(A)):
        key = A[j]
        i = j - 1
        while i >= 0 and A[i] < key:
            A[i + 1] = A[i]
            i = i - 1
        A[i + 1] = key


if __name__ == '__main__':
    A = [3, 4, 6, 2, 1]
    INSERTION_SORT_Non_descending(A)
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
        C.append((A[i] + B[i]+temp)%2)
        temp=(A[i] + B[i]+temp)//2
    C.append(temp)


if __name__ == '__main__':
    A = [1, 0, 1, 1, 1]
    B = [1, 0, 0, 1, 1]
    C=[]
    ADD(A,B,C)
    print(C)

#2.2 分析算法

#2.2-2 选择算法
def SELECT_SORT(A):
  for i in range(len(A)-1):
      min = i
      for j in range(i+1,len(A)):
          if A[j] < A[min]:
              min = j
      A[i],A[min]=A[min],A[i]
  return A


if __name__ == '__main__':
    A = [3, 4, 6, 2, 1,7,4]
    SELECT_SORT(A)
    print(A)

#2.3 设计算法
# 使用哨兵合并两个数组
def MERGE_with_guard(A,p,q,r):
    n1=q-p+1
    n2=r-q
    L=np.zeros(n1+1)
    R=np.zeros(n2+1)
    for i in range(n1):
        L[i]=A[p+i]
    for j in range(n2):
        R[j]=A[q+j+1]
    i=0
    j=0
    L[n1]=float('inf')
    R[n2]=float('inf')
    for k in range(p,r+1):
        if L[i]<=R[j]:
            A[k]=L[i]
            i+=1
        else:
            A[k]=R[j]
            j+=1

if __name__ == '__main__':
    A = [1, 3, 5, 6, 2,2,4,8,9]
    MERGE_with_guard(A,0,3,8)
    print(A)



#归并排序
def MERGESORT(A,p,r):
    if p<r:
        q=(p+r)//2
        MERGESORT(A,p,q)
        MERGESORT(A,q+1,r)
        MERGE_with_guard(A,p,q,r)

if __name__ == '__main__':
    A = [1, 3, 5, 6, 2,2,4,8,9]
    MERGESORT(A,0,8)
    print(A)


# 2.3-2 不使用哨兵合并两个数组

def MERGE_without_guard(A,p,q,r):
    n1=q-p+1
    n2=r-q
    L=np.zeros(n1)
    R=np.zeros(n2)
    for i in range(n1):
        L[i]=A[p+i]
    for j in range(n2):
        R[j]=A[q+j+1]
    i=0
    j=0
    for k in range(p,r+1):
        if i ==n1:
            A[k]=R[j]
            j=j+1
        elif j==n2:
            A[k]=L[i]
            i=i+1
        elif L[i] <= R[j]:
            A[k] = L[i]
            i = i + 1
        else:
            A[k] = R[j]
            j = j + 1

if __name__ == '__main__':
    A = [1, 3, 5, 6, 2,2,4,8,9]
    MERGE_without_guard(A,0,3,8)
    print(A)



#归并排序
def MERGESORT_1(A,p,r):
    if p<r:
        q=(p+r)//2
        MERGESORT(A,p,q)
        MERGESORT(A,q+1,r)
        MERGE_without_guard(A,p,q,r)

if __name__ == '__main__':
    A = [1, 3, 5, 6, 2,2,4,8,9]
    MERGESORT_1(A,0,8)
    print(A)





