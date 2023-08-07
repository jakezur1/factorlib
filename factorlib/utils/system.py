import warnings
import inspect
import pandas as pd
import sys
import time
import os
import shutil
import threading

from pathlib import Path
from functools import wraps

from factorlib.utils.types import FactorlibUserWarning


def is_frozen():
    return getattr(sys, 'frozen', False)


def get_root_dir() -> Path:
    if is_frozen():
        # we are running in a bundle
        return Path(sys.executable).parent

    # we are running in a normal Python environment
    return Path(__file__).resolve().parent.parent.parent


def get_datetime_maps_dir() -> Path:
    return get_root_dir() / 'factorlib' / 'utils' / 'datetime_maps'


def get_data_dir() -> Path:
    return get_root_dir() / 'data'


def get_factors_dir() -> Path:
    return get_root_dir() / 'factors'


def silence_warnings():
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    warnings.filterwarnings(action='ignore', message='An input array is constant; '
                                                     'the correlation coefficient is not defined.')
    warnings.filterwarnings(action='ignore', message='ntree_limit is deprecated, use `iteration_range` or '
                                                     'model slicing instead.')

    warnings.filterwarnings(action='ignore', message='Clustering metrics expects discrete values but received '
                                                     'continuous values for label, and continuous values for target')
    warnings.filterwarnings(action='ignore', message='Clustering metrics expects discrete values but received '
                                                     'binary values for label, and continuous values for target')


def print_dynamic_line(character='-'):
    terminal_width, _ = shutil.get_terminal_size()
    print(character * terminal_width)


def print_warning(message: str, category: FactorlibUserWarning):
    print(f'\033[0;31m[WARNING] ({category}): {message}\033[0m')


def show_processing_animation(animation: any, message_func=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the message
            if callable(message_func):
                message = message_func(*args, **kwargs)
            else:
                message = "Processing..."

            stop_event = threading.Event()

            # Start the ellipsis animation in a thread
            t = threading.Thread(target=animation, args=(message, stop_event))
            t.start()

            # Now, run the actual function to do the work
            result = func(*args, **kwargs)

            # Stop the animation
            stop_event.set()
            t.join()

            sys.stdout.write('\033[92m✔\033[0m\n')
            sys.stdout.flush()

            return result

        return wrapper

    return decorator


def _spinner_animation(message: str, stop_event: threading.Event):
    line_position = 60
    num_dashes = line_position - len(message) - 1
    sys.stdout.write(f'{message} {"-" * num_dashes} ')

    sys.stdout.flush()
    spinner_chars = ['|', '/', '-', '\\']
    while not stop_event.is_set():
        for char in spinner_chars:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(0.2)
            if stop_event.is_set():
                break
            sys.stdout.write('\b')
    sys.stdout.write('\b| ')


def _ellipsis_animation(message: str, stop_event: threading.Event):
    sys.stdout.write(message)
    sys.stdout.flush()
    elipses_count = 0
    while not stop_event.is_set():
        for char in "...":
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(0.4)
            elipses_count += 1
            if stop_event.is_set():
                break
            if stop_event.is_set():
                break
        if not stop_event.is_set():
            sys.stdout.write("\b" * elipses_count)
            elipses_count = 0
        sys.stdout.write('\b' * elipses_count)


def _get_defining_class(meth) -> any:
    if inspect.ismethod(meth):
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls
    if inspect.isfunction(meth):
        return getattr(inspect.getmodule(meth),
                       meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0],
                       None)
    return None
