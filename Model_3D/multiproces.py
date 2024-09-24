# ******************************************************************************
# @author: L. I. Nurtdinova
# ******************************************************************************

import multiprocessing
import os


def run_script(command):
    os.system(command)


if __name__ == "__main__":
    # Определяем команды для запуска скриптов

    command2 = 'python3 monte1.py'
    command3 = 'python3 monte2.py'
    command4 = 'python3 monte3.py'
    command5 = 'python3 monte4.py'
    command6 = 'python3 monte5.py'

    process2 = multiprocessing.Process(target=run_script, args=(command2,))
    process3 = multiprocessing.Process(target=run_script, args=(command3,))
    process4 = multiprocessing.Process(target=run_script, args=(command4,))
    process5 = multiprocessing.Process(target=run_script, args=(command5,))
    process6 = multiprocessing.Process(target=run_script, args=(command6,))
    process2.start()
    process3.start()
    process4.start()
    process5.start()
    process6.start()
    process2.join()
    process3.join()
    process4.join()
    process5.join()
    process6.join()

'''
def run_ves():
    import os
    #В аргумент os.system() надо передать python3 + имя основной программы, считающей координаты частиц
    os.system('python3 monte.py')

def main():
    #Количество процессоров
    num_processors = 15
    processes = []

    for _ in range(num_processors):
        process = Process(target=run_ves)
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

if __name__ == '__main__':
    main()'''
