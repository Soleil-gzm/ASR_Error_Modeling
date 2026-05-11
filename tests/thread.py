import time
from threading import Thread
 
COUNT = 100_000_000
 
def countdown():
    n = COUNT
    while n > 0:
        n -= 1
 
# --- 单线程测试 ---
start_time = time.time()
countdown()
countdown()
end_time = time.time()
print(f"单线程耗时: {end_time - start_time:.4f} 秒")
 
# --- 双线程测试 ---
thread1 = Thread(target=countdown)
thread2 = Thread(target=countdown)
start_time = time.time()
thread1.start()
thread2.start()
thread1.join()
thread2.join()
end_time = time.time()
print(f"双线程耗时: {end_time - start_time:.4f} 秒")