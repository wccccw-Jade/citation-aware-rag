from src.api import queue
from src.config import Settings


class FakeJob:
    id = "job-1"


class FakeQueue:
    name = "rag-indexing"

    def __init__(self) -> None:
        self.kwargs = {}

    def enqueue(self, *args, **kwargs):
        self.kwargs = kwargs
        return FakeJob()


def test_enqueue_index_task_uses_rq_safe_job_id(monkeypatch) -> None:
    fake_queue = FakeQueue()
    settings = Settings(redis_url="redis://localhost:6379/0")
    monkeypatch.setattr(queue, "get_index_queue", lambda settings=None: fake_queue)

    job_id = queue.enqueue_index_task("task-123", settings=settings)

    assert job_id == "job-1"
    assert fake_queue.kwargs["job_id"].startswith("index-task-123-")
    assert ":" not in fake_queue.kwargs["job_id"]
