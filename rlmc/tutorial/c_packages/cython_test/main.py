from examples_cy import hello_cython,pi_cy
import time

def pi_py(N):
    pi = 0
    for n in range(N):
        pi += (-1) **n / (2*n + 1)
    return 4*pi

if __name__ == "__main__":
    print(hello_cython())
    
    stime = time.time()
    for _ in range(100):
        out1 = pi_py(1000000)
    etime = time.time()

    stime2 = time.time()
    for _ in range(100):
        out2 = pi_cy(1000000)
    etime2 = time.time()

    print("pi_py result {} time cost {} s".format(out1, etime - stime))
    print("pi_cy result {} time cost {} s".format(out2, etime2 - stime2))
