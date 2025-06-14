import multiprocessing
import time

# 第一个函数：模拟一个耗时的任务
a = []
def function_1():
    print("Function 1 starting...")
    time.sleep(3)  # 模拟任务处理时间
    a.append("222")
    print("Function 1 completed.")

# 第二个函数：模拟另一个耗时的任务
def function_2():
    print("Function 2 starting...")
    time.sleep(2)  # 模拟任务处理时间
    print("Function 2 completed.")

if __name__ == "__main__":
    # 创建两个进程来分别执行 function_1 和 function_2
    a.append("111")
    process_1 = multiprocessing.Process(target=function_1)

    
    # 启动进程
    process_1.start()

    
    # 等待两个进程结束


    
    print("Both functions are done.")
    print(a)