from __future__ import annotations

import sys
from pathlib import Path

from redis import Redis
from rq import SimpleWorker, Worker

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.api.logging_config import configure_logging
from src.config import get_settings


def main() -> None:
    configure_logging()
    settings = get_settings()
    redis_conn = Redis.from_url(settings.redis_url)
    worker_cls = SimpleWorker if settings.rq_worker_class.lower() == "simple" else Worker
    worker = worker_cls([settings.task_queue_name], connection=redis_conn)
    worker.work()


if __name__ == "__main__":
    main()
