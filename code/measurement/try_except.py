# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
import numpy as np
import time
import random


N_BITS = 1_000_000

def runner(bits):

    def try_except(value):
        try:
            return 1/value
        except ZeroDivisionError:
            return 0
    
    def if_clause(value):
        if value != 0:
            return 1/value
        else:
            return 0
    
    start1 = time.perf_counter()
    calls1 = [try_except(v) for v in bits]
    end1 = time.perf_counter()
    dx1 = end1-start1

    start2 = time.perf_counter()
    calls2 = [if_clause(v) for v in bits]
    end2 = time.perf_counter()
    dx2 = end2-start2
    assert len(calls1) == len(calls2)

    print("try_except :", dx1, "sec")  # try_except : 0.3197657830000935 sec
    print("if_clause  :", dx2, "sec")  # if_clause  : 0.08006944099997781 sec
    print("if_clause / try_except = ", dx1/dx2, "x")  # if_clause / try_except =  3.9936057877584297 x
    print("try_except / if_clause = ", dx2/dx1, "x")  # try_except / if_clause =  0.25040027813092935 x


def test1():
    # setup test.
    random.seed(5)
    bits = np.array([random.randint(0,1) for _ in range(N_BITS)])
    print("\n")
    print("hits / misses", sum(bits), "/", N_BITS - np.sum(bits))
    runner(bits)


def runner2(hits):
    misses = N_BITS - hits
    print("\n")
    print("hits: ", hits, "misses: ", misses)
    bits = np.array([1 for _ in range(hits)] + [0 for _ in range(misses)], dtype=np.uint8)
    runner(bits)


def test2():
    runner2(500_000)

def test3():
    runner2(999_999)

def test4():
    runner2(1)

def test5():
    runner2(960_000)


if __name__ == "__main__":
    test1()
    test2()
    test3()
    test4()
    test5()
# ---------------------------------
# python3 -O try-or-not-to-try.py 

# hits / misses 500101 / 499899
# try_except : 0.3125843550001264 sec
# if_clause  : 0.07531726600018374 sec
# if_clause / try_except =  4.150235020471452 x
# try_except / if_clause =  0.24095021006458653 x


# hits:  500000 misses:  500000
# try_except : 0.280955866000113 sec
# if_clause  : 0.07346094000013181 sec
# if_clause / try_except =  3.824561270242511 x
# try_except / if_clause =  0.2614678990190661 x


# hits:  999999 misses:  1
# try_except : 0.07674370499989891 sec
# if_clause  : 0.09772015599992301 sec
# if_clause / try_except =  0.7853416136580806 x
# try_except / if_clause =  1.2733312263208003 x


# hits:  1 misses:  999999
# try_except : 0.4878366029997778 sec
# if_clause  : 0.04770775599990884 sec
# if_clause / try_except =  10.225519787615037 x
# try_except / if_clause =  0.09779453961950979 x


# hits:  960000 misses:  40000
# try_except : 0.09383948499998951 sec
# if_clause  : 0.09579665699993711 sec
# if_clause / try_except =  0.9795695167113306 x
# try_except / if_clause =  1.0208565935751652 x