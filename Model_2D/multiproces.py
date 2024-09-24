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

    process2 = multiprocessing.Process(target=run_script, args=(command2,))
    process3 = multiprocessing.Process(target=run_script, args=(command3,))
    process4 = multiprocessing.Process(target=run_script, args=(command4,))
    process5 = multiprocessing.Process(target=run_script, args=(command5,))

    process2.start()
    process3.start()
    process4.start()
    process5.start()
    process2.join()
    process3.join()
    process4.join()
    process5.join()
