import numpy
import pickle
import os
import subprocess

def test_artur(test):
    mp = test[0].astype('int32')
    st = test[1]
    fn = test[2]
    m = len(mp)
    n = len(st)
    res = str(m) + " " + str(n)
    for i in range(m):
        for j in range(m):
            res += " " + str(mp[i][j])
    for i in range(n):
        res += " " + str(st[i][0]) + " " + str(st[i][1])

    for i in range(n):
        res += " " + str(fn[i][0]) + " " + str(fn[i][1])
    
    batcmd = f"./main {res}"
    result = subprocess.check_output(batcmd, shell=True).decode("utf-8").strip()  

    lines = result.split('\n')
    time = float(lines[-1])
    tr = [list(map(int, line.split())) for line in lines[:-1]]
    return (tr, time)



def prepare_cpp(filename):
    os.system(f"g++ -O2 -std=c++17 -o './main' '{filename}'")


if __name__ == '__main__':
    prepare_cpp('main.cpp')
