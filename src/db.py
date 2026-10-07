from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import DB_PATH, DATA_DIR


class Base(DeclarativeBase):
    pass


class RosterUpdate(Base):
    __tablename__ = "roster_updates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_year = Column(Integer, nullable=False, index=True)
    update_id = Column(Integer, nullable=False)
    update_name = Column(String(64), nullable=False)
    update_date = Column(String(32))
    raw_json_path = Column(String(512))


class AttributeChange(Base):
    __tablename__ = "attribute_changes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_year = Column(Integer, nullable=False, index=True)
    update_id = Column(Integer, nullable=False, index=True)
    update_name = Column(String(64))
    update_date = Column(String(32))
    card_uuid = Column(String(64), index=True)
    player_name = Column(String(128), index=True)
    team = Column(String(64))
    position = Column(String(16))
    is_hitter = Column(Integer, default=1)
    attribute_name = Column(String(64), index=True)
    rating_before = Column(Integer)
    rating_after = Column(Integer)
    delta = Column(Integer)
    ovr_before = Column(Integer)
    ovr_after = Column(Integer)
    rarity_before = Column(String(32))
    rarity_after = Column(String(32))
    mlb_player_id = Column(Integer, index=True)


class CardSnapshot(Base):
    __tablename__ = "card_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_year = Column(Integer, nullable=False, index=True)
    card_uuid = Column(String(64), nullable=False, index=True)
    player_name = Column(String(128), index=True)
    team = Column(String(64))
    position = Column(String(16))
    ovr = Column(Integer)
    rarity = Column(String(32))
    series = Column(String(32))
    is_hitter = Column(Integer, default=1)
    mlb_player_id = Column(Integer, index=True)
    attributes_json = Column(Text)
    snapshot_at = Column(DateTime, default=datetime.utcnow)


class PlayerStatWindow(Base):
    __tablename__ = "player_stat_windows"

    id = Column(Integer, primary_key=True, autoincrement=True)
    mlb_player_id = Column(Integer, nullable=False, index=True)
    as_of_date = Column(String(16), nullable=False, index=True)
    window = Column(String(16), nullable=False)
    is_hitter = Column(Integer, default=1)
    stats_json = Column(Text)


class TrainingExample(Base):
    __tablename__ = "training_examples"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_year = Column(Integer, index=True)
    update_id = Column(Integer, index=True)
    update_date = Column(String(32))
    card_uuid = Column(String(64))
    player_name = Column(String(128))
    mlb_player_id = Column(Integer, index=True)
    attribute_name = Column(String(64), index=True)
    rating_before = Column(Integer)
    rating_after = Column(Integer)
    delta = Column(Integer)
    change_label = Column(String(16))
    ovr_before = Column(Integer)
    rarity_before = Column(String(32))
    position = Column(String(16))
    is_hitter = Column(Integer)
    features_json = Column(Text)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    card_uuid = Column(String(64), index=True)
    player_name = Column(String(128))
    mlb_player_id = Column(Integer)
    current_ovr = Column(Integer)
    current_rarity = Column(String(32))
    predicted_ovr_delta = Column(Float)
    upgrade_probability = Column(Float, default=0.0)
    downgrade_probability = Column(Float, default=0.0)
    investment_score = Column(Float)
    expected_value_per_card = Column(Float)
    roi_pct = Column(Float)
    tier_jump_probability = Column(Float)
    tier_down_probability = Column(Float, default=0.0)
    ovr_delta_sd = Column(Float)
    stats_json = Column(Text)
    sample_size_ok = Column(Integer, default=1)
    horizon_days = Column(Integer, default=1)
    attributes_json = Column(Text)
    avg_gap = Column(Float)
    direction_consensus = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelMetrics(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(64))
    metric_name = Column(String(64))
    metric_value = Column(Float)
    fold = Column(String(32))
    created_at = Column(DateTime, default=datetime.utcnow)


def get_engine(db_path: Path | None = None):
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}", echo=False)


@lru_cache(maxsize=8)
def init_db(db_path: Path | None = None) -> sessionmaker[Session]:
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    _migrate_predictions_columns(engine)
    return sessionmaker(bind=engine)


def _migrate_predictions_columns(engine) -> None:
    """Add new columns to the predictions table if missing (SQLite ALTER TABLE)."""
    import sqlalchemy as sa
    new_cols = [
        ("upgrade_probability", sa.Float),
        ("downgrade_probability", sa.Float),
        ("investment_score", sa.Float),
        ("expected_value_per_card", sa.Float),
        ("roi_pct", sa.Float),
        ("tier_down_probability", sa.Float),
        ("ovr_delta_sd", sa.Float),
        ("stats_json", sa.Text),
    ]
    with engine.connect() as conn:
        try:
            existing = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(predictions)")}
        except Exception:
            return
        for name, col_type in new_cols:
            if name not in existing:
                try:
                    conn.exec_driver_sql(
                        f"ALTER TABLE predictions ADD COLUMN {name} {col_type().compile()}"
                    )
                    conn.commit()
                except Exception:
                    pass


def safe_init_db() -> sessionmaker[Session] | None:
    try:
        return init_db()
    except Exception:
        return None


def dumps(obj) -> str:
    return json.dumps(obj, default=str)
