from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel, create_engine

from config import settings

db_client = create_engine(settings.database_url, pool_size=5)


class EventLog(SQLModel, table=True):
    __tablename__ = "event_log"
    seq: int | None = Field(default=None, primary_key=True)
    data: dict = Field(sa_column=Column(JSONB, nullable=False))


# Create the table
def ensure_schema() -> None:
    SQLModel.metadata.create_all(db_client)
