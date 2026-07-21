"""
task_queue.py

Store pending robot delivery tasks.
"""

from queue import Queue


class TaskQueue:

    def __init__(self):

        self.queue = Queue()

    def add(self, task):
        """
        Add a delivery task to the queue.
        """
        self.queue.put(task)

    def has_task(self):
        """
        Check whether there are pending tasks.
        """
        return not self.queue.empty()

    def next_task(self):
        """
        Get the next delivery task.
        """
        return self.queue.get()