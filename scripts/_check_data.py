#!/usr/bin/env python3
"""Count how many predictions have team info available."""
import sys
sys.path.insert(0, r"C:\Users\luked\mlb-show-roster-predictor")
from src.db import Prediction, AttributeChange, init_db
from sqlalchemy import func

Session = init_db()
with Session() as s:
    # Use a subquery approach: join predictions with their latest attribute_changes
    from sqlalchemy.orm import aliased
    
    cards = s.query(Prediction).filter_by(horizon_days=1).all()
    count = 0
    no_team = 0
    for p in cards:
        info = (
            s.query(AttributeChange.team, AttributeChange.position)
            .filter(AttributeChange.mlb_player_id == p.mlb_player_id)
            .filter(AttributeChange.team.isnot(None))
            .filter(AttributeChange.team != "")
            .order_by(AttributeChange.update_date.desc(), AttributeChange.id.desc())
            .first()
        )
        if info:
            count += 1
        else:
            no_team += 1
    print(f"Have team info (non-empty): {count}/{len(cards)}")
    print(f"No team info: {no_team}/{len(cards)}")
