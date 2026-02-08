import time
import multiprocessing
from multiprocessing import Pool
import numpy as np
import NoiseDetectFunctions as Ndf
import SupportFunctions as Sf


class para:
    def __init__(self, ai, bi):
        self.a = ai
        self.b = bi


def func(para):
    # time.sleep(3)
    # print("move box {} from A to B ...".format(para))
    # temp = np.zeros([para.a, para.b])
    temp = Sf.random_density_matrix(para.a)
    temp1 = Ndf.get_alps_label(temp, para.b)
    return temp1


def test2():
    time1 = time.time()
    p = Pool(10)
    paras=[]
    for i in range(50):
        parai = para(4, 4)
        paras.append(parai)
    ans = p.map(func, paras)
    # print(ans)
    time2 = time.time()
    print("time cost {}".format(time2-time1))
    return (ans)
