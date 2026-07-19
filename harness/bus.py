import concurrent.futures
import time
import uuid
from collections.abc import Callable

from sqlmodel import Session, col, select

from harness.db import EventLog, db_client

Listener = Callable[[dict], None]
_listeners: set[Listener] = set()

_db_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def subscribe(listener: Listener) -> Callable[[], None]:
    _listeners.add(listener)
    return lambda: _listeners.discard(listener)


def _db_write(event: dict) -> None:
    with Session(db_client) as session:
        session.add(EventLog(data=event))
        session.commit()


def emit(event: dict) -> None:
    event = {**event, "id": str(uuid.uuid4()), "ts": int(time.time() * 1000)}
    _db_pool.submit(_db_write, event)
    for listener in list(_listeners):
        listener(event)


# replays from postgress
def history() -> list[dict]:
    with Session(db_client) as session:
        rows = session.exec(select(EventLog).order_by(col(EventLog.seq))).all()
        return [row.data for row in rows]
