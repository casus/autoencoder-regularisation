import multiprocessing
import time

from pkg_resources import find_distributions

start = time.perf_counter()


def do_something(seconds):
    print(f'Sleeping {seconds} second...')
    time.sleep(seconds)
    print('Done sleeping...')

processes = []

for _ in range(10): # _ underscore means that we are not using the variable i for looping but just using the for loop to do something repeteadly
    p = multiprocessing.Process(target=do_something, args=[1.5])
    p.start()
    processes.append(p)

for process in processes:
    process.join()

'''p1 = multiprocessing.Process(target=do_something)
p2 = multiprocessing.Process(target=do_something)

p1.start()
p2.start()

p1.join()
p2.join()  # means that the process first finishes before moing on to the script below
#do_something()
#do_something()
#do_something()'''

finish = time.perf_counter()

print(f'Finished in {round(finish-start, 2)} seconds')