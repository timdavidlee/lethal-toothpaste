import numpy as np


def generate_names(n: int = 500_000):
    with open("./lethaltoothpaste/names/boy_names.txt", "r") as f:
        boy_first_names = f.read().split("\n")
        boy_first_names = [b.lower() for b in boy_first_names]

    with open("./lethaltoothpaste/names/girl_names.txt", "r") as f:
        girl_first_names = f.read().split("\n")
        girl_first_names = [g.lower() for g in girl_first_names]

    with open("./lethaltoothpaste/names/last_names.txt", "r") as f:
        last_names = f.read().split("\n")
        last_names = [l.lower() for l in last_names]

    last_names = np.random.choice(last_names, size=n)
    first_names = np.random.choice(boy_first_names + girl_first_names, size=n)
    return ["{} {}".format(f, l) for f, l in zip(first_names, last_names)]