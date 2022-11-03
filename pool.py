import multiprocessing


def init():
    pool = multiprocessing.Pool()
    return pool

# for global use import whole pool.py
pool = None
