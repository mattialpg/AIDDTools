import multiprocessing
from functools import wraps


class TimeExceededException(Exception):
    pass


## PART 1
def function_runner(*args, **kwargs):
    """Used as a wrapper function to handle
    returning results on the multiprocessing side"""

    send_end = kwargs.pop("__send_end")
    function = kwargs.pop("__function")
    try:
        result = function(*args, **kwargs)
    except Exception as e:
        send_end.send(e)
        return
    send_end.send(result)


@parametrized
def run_with_timer(func, max_execution_time):
    @wraps(func)
    def wrapper(*args, **kwargs):
        recv_end, send_end = multiprocessing.Pipe(False)
        kwargs["__send_end"] = send_end
        kwargs["__function"] = func
        
        ## PART 2
        p = multiprocessing.Process(target=function_runner, args=args, kwargs=kwargs)
        p.start()
        p.join(max_execution_time)
        if p.is_alive():
            p.terminate()
            p.join()
            raise TimeExceededException("Exceeded Execution Time")
        result = recv_end.recv()

        if isinstance(result, Exception):
            raise result

        return result

    return wrapper