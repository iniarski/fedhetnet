import csv
import threading
import queue
import os

class CSVLogger:
    def __init__(self, filepath : str, fieldnames : list[str]):
        self.filepath = filepath
        self.fieldnames = fieldnames

        self.queue = queue.Queue()
        self._stop_event = threading.Event()

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        self._file_exists = os.path.isfile(filepath)

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def log(self, data: dict):
        missing = set(self.fieldnames) - set(data.keys())
        if missing:
            raise ValueError(f"Missing keys in log data: {missing}")

        self.queue.put(data)

    def _worker(self):
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)

            if not self._file_exists:
                writer.writeheader()
                f.flush()
                os.fsync(f.fileno())

            while not self._stop_event.is_set() or not self.queue.empty():
                try:
                    item = self.queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                writer.writerow(item)

                f.flush()
                os.fsync(f.fileno())

                self.queue.task_done()

    def close(self):
        self._stop_event.set()
        self.thread.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
