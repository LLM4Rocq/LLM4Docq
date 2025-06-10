import re
import subprocess
import time
import signal
import random

from pytanque import Pytanque, PetanqueError



def start_pet_server(port=8765, mean_wait=10):
    """
    Starts the pet-server process and returns the process handle.
    """
    process = subprocess.Popen(["pet-server", "--port", f"{port}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # wait a bit to ensure the server is fully up before proceeding
    wait = random.randint(1, 2*mean_wait)
    time.sleep(wait)
    return process

def stop_pet_server(process):
    """
    Gracefully stops the pet-server process.
    """
    process.terminate()  # Sends SIGTERM
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()  # Force kill if not terminated
        process.wait()

# Define a custom exception for timeouts
class TimeoutError(Exception):
    pass

def timeout(seconds=5, error_message="Function call timed out"):
    """
    A decorator that raises a TimeoutError if the decorated function
    does not return within 'seconds' seconds.
    """
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)
        def wrapper(*args, **kwargs):
            # Set the signal handler and a timeout alarm
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                # Cancel the alarm
                signal.alarm(0)
            return result
        return wrapper
    return decorator

SSR_HEADER = "From Coq Require Import ssreflect ssrfun ssrbool.\n"
def eval_tactics(thm, workspace, filepath, tactics, url="127.0.0.1", port=8765, timeout=10):
    """
    Try to solve theorem "thm" in the source file "filepath" using tactics.
    """
    with open(filepath, 'r') as file:
        content_file = file.read()
    
    new_filepath = filepath.split('.')[0] + 'aux_ssreflect.v'
    with open(new_filepath, 'w') as file:
        file.write(SSR_HEADER + content_file)
    filepath = new_filepath
    with Pytanque(url, port) as pet:
        try:
            pet.set_workspace(True, workspace)
            state = pet.start(file=filepath, thm=thm)
        except PetanqueError as e:
            return [], [{"status": "error", "goals": [], "message": e.message, "tactic": ""}]
        init_goals = [goal.pp for goal in pet.goals(state)]  
        res = []
        for tactic in tactics:
            entry = {"status": "", "goals": [], "message": "", "tactic": tactic}
            try:
                state = pet.run_tac(state, tactic, verbose=False, timeout=timeout)
                goals = pet.goals(state)

                entry['goals'] = [goal.pp for goal in goals]                
                if state.proof_finished:
                    entry['status'] = "finish"
                else:
                    entry['status'] = "ongoing"
                res.append(entry)

                if state.proof_finished:
                    break
            except PetanqueError as e:
                entry['status'] = "error"
                entry['message'] = e.message
                res.append(entry)
                break
        return init_goals, res