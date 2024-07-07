def hello_cython():
    return "Hello Cython!"


cpdef double pi_cy(int N):
    cpdef double pi = 0
    cpdef int n;
    for n in range(N):
        pi += (-1.0) ** n / (2 * n + 1)
    return 4 * pi