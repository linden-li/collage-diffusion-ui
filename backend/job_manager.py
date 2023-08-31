from multiprocessing.managers import SyncManager
from queue import PriorityQueue


class ReadyQueue(PriorityQueue):
    def get_attribute(self, name):
        return getattr(self, name)


class JobManager(SyncManager):
    pass


JobManager.register("PriorityQueue", ReadyQueue)  # Register a shared PriorityQueue


def CreateJobManager():
    m = JobManager()
    m.start()
    return m
