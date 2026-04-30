from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

import src.api.main as api_main
from src.api import repository
import src.api.service as api_service
import src.api.worker as api_worker
from src.api.database import create_database_engine, get_db, init_db
from src.api.service import RAGService
from src.config import Settings


def test_upload_persists_document_and_task_status_after_service_restart(tmp_path: Path, monkeypatch) -> None:
    database_url = f"sqlite:///{tmp_path / 'app.db'}"
    engine = create_database_engine(database_url)
    init_db(engine)
    testing_session = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False, future=True)

    def override_db():
        db = testing_session()
        try:
            yield db
        finally:
            db.close()

    settings = Settings(
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        index_dir=tmp_path / "index",
        database_url=database_url,
        embedding_provider="local",
        vector_store_provider="numpy",
        use_faiss=False,
    )
    api_main.app.dependency_overrides[get_db] = override_db
    monkeypatch.setattr(api_main, "rag_service", RAGService(settings))
    monkeypatch.setattr(api_service, "enqueue_index_task", lambda task_id, settings=None: "job-1")

    client = TestClient(api_main.app)
    upload = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"RAG persists document and indexing task status.", "text/plain")},
    )

    assert upload.status_code == 202
    document_id = upload.json()["document_id"]
    task_id = upload.json()["task_id"]
    assert upload.json()["status"] == "queued"

    monkeypatch.setattr(api_main, "rag_service", RAGService(settings))
    restarted_client = TestClient(api_main.app)

    document = restarted_client.get(f"/documents/{document_id}")
    task = restarted_client.get(f"/tasks/{task_id}")

    assert document.status_code == 200
    assert document.json()["status"] == "uploaded"
    assert task.status_code == 200
    assert task.json()["status"] == "queued"
    assert task.json()["retry_count"] == 0

    api_main.app.dependency_overrides.clear()


def test_worker_processes_queued_task_and_persists_indexed_status(tmp_path: Path, monkeypatch) -> None:
    database_url = f"sqlite:///{tmp_path / 'app.db'}"
    engine = create_database_engine(database_url)
    init_db(engine)
    testing_session = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False, future=True)

    def override_db():
        db = testing_session()
        try:
            yield db
        finally:
            db.close()

    settings = Settings(
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        index_dir=tmp_path / "index",
        database_url=database_url,
        embedding_provider="local",
        vector_store_provider="numpy",
        use_faiss=False,
    )
    api_main.app.dependency_overrides[get_db] = override_db
    monkeypatch.setattr(api_main, "rag_service", RAGService(settings))
    monkeypatch.setattr(api_service, "enqueue_index_task", lambda task_id, settings=None: "job-1")
    monkeypatch.setattr(api_worker, "SessionLocal", testing_session)
    monkeypatch.setattr(api_worker, "get_settings", lambda: settings)
    monkeypatch.setattr(api_worker, "init_db", lambda: None)

    client = TestClient(api_main.app)
    upload = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"RAG worker builds the index asynchronously.", "text/plain")},
    )
    task_id = upload.json()["task_id"]
    document_id = upload.json()["document_id"]

    stats = api_worker.process_index_task(task_id)

    assert stats == {"documents": 1, "chunks": 1}
    assert client.get(f"/documents/{document_id}").json()["status"] == "indexed"
    task = client.get(f"/tasks/{task_id}").json()
    assert task["status"] == "indexed"
    assert task["stats"] == {"documents": 1, "chunks": 1}

    api_main.app.dependency_overrides.clear()


def test_api_upload_index_query_cycle(tmp_path: Path, monkeypatch) -> None:
    database_url = f"sqlite:///{tmp_path / 'app.db'}"
    engine = create_database_engine(database_url)
    init_db(engine)
    testing_session = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False, future=True)

    def override_db():
        db = testing_session()
        try:
            yield db
        finally:
            db.close()

    settings = Settings(
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        index_dir=tmp_path / "index",
        database_url=database_url,
        embedding_provider="local",
        vector_store_provider="numpy",
        use_faiss=False,
    )
    api_main.app.dependency_overrides[get_db] = override_db
    monkeypatch.setattr(api_main, "rag_service", RAGService(settings))
    monkeypatch.setattr(api_service, "enqueue_index_task", lambda task_id, settings=None: "job-1")
    monkeypatch.setattr(api_worker, "SessionLocal", testing_session)
    monkeypatch.setattr(api_worker, "get_settings", lambda: settings)
    monkeypatch.setattr(api_worker, "init_db", lambda: None)

    client = TestClient(api_main.app)
    upload = client.post(
        "/documents/upload",
        files={
            "file": (
                "cycle.txt",
                b"Citation-aware RAG indexes documents asynchronously before answering questions.",
                "text/plain",
            )
        },
    )
    assert upload.status_code == 202

    task_id = upload.json()["task_id"]
    api_worker.process_index_task(task_id)

    query = client.post(
        "/query",
        json={"query": "What does citation-aware RAG index?", "top_k": 1},
    )

    assert query.status_code == 200
    payload = query.json()
    assert "Grounded answer:" in payload["answer"]
    assert payload["citations"]
    assert payload["retrieved_chunks"]

    api_main.app.dependency_overrides.clear()


def test_api_errors_use_uniform_response_shape(tmp_path: Path, monkeypatch) -> None:
    database_url = f"sqlite:///{tmp_path / 'app.db'}"
    engine = create_database_engine(database_url)
    init_db(engine)
    testing_session = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False, future=True)

    def override_db():
        db = testing_session()
        try:
            yield db
        finally:
            db.close()

    settings = Settings(
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        index_dir=tmp_path / "index",
        database_url=database_url,
        embedding_provider="local",
        vector_store_provider="numpy",
        use_faiss=False,
    )
    api_main.app.dependency_overrides[get_db] = override_db
    monkeypatch.setattr(api_main, "rag_service", RAGService(settings))

    client = TestClient(api_main.app)
    response = client.post(
        "/documents/upload",
        files={"file": ("bad.exe", b"not a supported document", "application/octet-stream")},
        headers={"X-Request-ID": "test-request-id"},
    )

    assert response.status_code == 400
    assert response.headers["X-Request-ID"] == "test-request-id"
    payload = response.json()
    assert set(payload) == {"error"}
    assert payload["error"]["code"] == "invalid_upload"
    assert payload["error"]["request_id"] == "test-request-id"

    api_main.app.dependency_overrides.clear()


def test_failed_task_can_be_retried(tmp_path: Path, monkeypatch) -> None:
    database_url = f"sqlite:///{tmp_path / 'app.db'}"
    engine = create_database_engine(database_url)
    init_db(engine)
    testing_session = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False, future=True)

    def override_db():
        db = testing_session()
        try:
            yield db
        finally:
            db.close()

    settings = Settings(
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        index_dir=tmp_path / "index",
        database_url=database_url,
        embedding_provider="local",
        vector_store_provider="numpy",
        use_faiss=False,
    )
    api_main.app.dependency_overrides[get_db] = override_db
    monkeypatch.setattr(api_main, "rag_service", RAGService(settings))
    monkeypatch.setattr(api_service, "enqueue_index_task", lambda task_id, settings=None: "job-1")

    client = TestClient(api_main.app)
    upload = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"Retry failed indexing tasks.", "text/plain")},
    )
    task_id = upload.json()["task_id"]

    with testing_session() as db:
        task = repository.get_index_task(db, task_id)
        document = repository.get_document(db, task.document_id)
        repository.mark_index_failure(db, document, task, "forced failure")
        db.commit()

    retry = client.post(f"/tasks/{task_id}/retry")

    assert retry.status_code == 202
    assert retry.json()["status"] == "queued"
    assert retry.json()["retry_count"] == 1
    assert client.get(f"/tasks/{task_id}").json()["status"] == "queued"

    api_main.app.dependency_overrides.clear()
