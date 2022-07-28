"""
Author: Steve Paul 
Date: 7/28/22 """
from multiprocessing import Process
from multiprocessing import Manager
import random
import os

class MyClass :
    """A simple example class"""
    def __init__(self):
        self.pid = -1
        self.msg = ['hello','world']

    def show(self):
        print('pid = ' + str(self.pid))
        print('msg = "' + ' '.join(self.msg) + '"')

# do_stuff
def do_stuff(pid, np, TheList):
    for i in range(pid,len(TheList),np):
        tmp = MyClass()
        tmp.pid = pid
        tmp.msg[0] = 'sample'
        tmp.msg[1] = str(i)
        tmp.msg.append('processed')
        tmp.msg.append('by')
        tmp.msg.append('pid')
        tmp.msg.append(str(pid))
        TheList[i] = tmp

# main program - fires off a bunch of processes to
# do stuff on a managed list.
if __name__ == "__main__":
    # extract num processes from SLURM
    # np = int(os.getenv('SLURM_NPROCS', '4'))
    np = 5

    samples=100
    # the processes will adjust a managed list of MyClass instances
    m = Manager()
    l = m.list(range(samples))

    # create np processes. each one with invoke do_stuff()
    # and operate on a subset of the total number of samples.
    procs = []
    for i in range(0,np):
        p = Process(target=do_stuff, args=(i, np, l))
        procs.append(p)

    # launch the processes. They will run in parallel.
    for p in procs:
        p.start()

    # halt the child processes
    for p in procs:
        p.join()

    # display results from each process
    for i in l:
        i.show()