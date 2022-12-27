import numpy
import os
import subprocess


def prepare_cpp(filename, main_path):
    os.system(f"g++ -O2 -std=c++17 -o '{main_path}' '{filename}'")


def call_single_mstar_test(test, main_path):
    mp, st, fn = test[0].astype("int32"), test[1], test[2]
    m, n = len(mp), len(st)
    args_array = [m, n]

    for i in range(m):
        for j in range(m):
            args_array.append(mp[i][j])

    for i in range(n):
        args_array.append(st[i][0])
        args_array.append(st[i][1])

    for i in range(n):
        args_array.append(fn[i][0])
        args_array.append(fn[i][1])

    res = " ".join(str(arg) for arg in args_array)

    result = (
        subprocess.check_output(f"{main_path} {res}", shell=True)
        .decode("utf-8")
        .strip()
    )

    lines = result.split("\n")
    time = float(lines[-1])

    return [list(map(int, line.split())) for line in lines[:-1]], time
