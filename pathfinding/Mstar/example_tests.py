import numpy
import pickle 
from test import prepare_cpp, test_artur

FILENAME = 'main.cpp'
prepare_cpp(FILENAME) #compiling file with c++ solution while importing poshel nahren eng language


#example of usage
if __name__ == '__main__':
    with open('map2.pkl', 'rb') as f: 
        x = pickle.load(f)
        for test in x:
            result, tme = test_artur(test)
            print(*result, tme, sep = '\n', end = '\n---\n')


