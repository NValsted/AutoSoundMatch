from threading import Lock


# References https://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html
class Singleton:
    _instance = None
    _lock: Lock = Lock()

    def __new__(cls):
        with cls._lock:
            if Singleton._instance is None:
                Singleton._instance = object.__new__(cls)
            return Singleton._instance
