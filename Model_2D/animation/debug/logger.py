import time


def start_timer():
    return time.time()


def format_time_delta(timer_name: str, time_delta: float) -> float:
    time_seconds = time_delta % 60
    time_minutes = int(time_delta / 60) % 60
    time_hours = int(time_delta / 60 / 60) % 60
    print(f"Execution time for '{timer_name}' is {time_hours:.0f}:{time_minutes:.0f}:{time_seconds:.4f}")
    return time_delta


def release_timer(timer_name: str, time_data: time) -> float:
    return format_time_delta(timer_name, time.time() - time_data)


def get_time_diff(time_data: time) -> float:
    return time.time() - time_data
