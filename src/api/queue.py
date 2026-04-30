from __future__ import annotations

import logging
from uuid import uuid4

from redis import Redis
from rq import Queue

from src.config import Settings, get_settings

logger = logging.getLogger("citation_aware_rag.queue")


def get_redis(settings: Settings | None = None) -> Redis:
    resolved = settings or get_settings()
    return Redis.from_url(resolved.redis_url)


def get_index_queue(settings: Settings | None = None) -> Queue:
    resolved = settings or get_settings()
    return Queue(resolved.task_queue_name, connection=get_redis(resolved))


def enqueue_index_task(task_id: str, settings: Settings | None = None) -> str:
    queue = get_index_queue(settings)
    job = queue.enqueue(
        "src.api.worker.process_index_task",
        task_id,
        job_id=f"index-{task_id}-{uuid4().hex}",
        job_timeout="1h",
        result_ttl=86400,
        failure_ttl=86400,
    )
    logger.info("index_task_enqueued task_id=%s job_id=%s queue=%s", task_id, job.id, queue.name)
    return job.id
