from __future__ import annotations

from typing import Any, Dict, List, Optional, Callable
import json
from datetime import datetime, timezone

try:
    from database.connector import get_db_connection
except ModuleNotFoundError:  # allow running this file directly
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]  # src/dashboard_project
    if str(root) not in sys.path:
        sys.path.append(str(root))
    from database.connector import get_db_connection


class TeamTaskService:
    """Service for reading/aggregating and mutating team tasks from complaints table."""

    def __init__(
        self,
        connection_getter: Callable[[str], Any] = get_db_connection,
        db_key: str = "database",
        table: str = "complaints",
    ) -> None:
        self._get_conn = connection_getter
        self._db_key = db_key
        self._table = table

    # ------------------------- helpers -------------------------
    @staticmethod
    def _norm_status(s: Optional[str]) -> str:
        if not s:
            return "open"
        s = str(s).strip().lower()
        if s.startswith("resolv") or s.startswith("clos") or s in {"done", "complete", "completed"}:
            return "resolved"
        return "open"

    @staticmethod
    def _parse_dt(val: Optional[str]) -> Optional[datetime]:
        if not val:
            return None
        s = str(val).strip()
        # Try multiple formats; fallback to None
        fmts = (
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y",
        )
        for fmt in fmts:
            try:
                dt = datetime.strptime(s, fmt)
                if not dt.tzinfo:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                continue
        try:
            dt = datetime.fromisoformat(s)
            if not dt.tzinfo:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None

    @staticmethod
    def _time_ago_str(dt: Optional[datetime]) -> str:
        if not dt:
            return ""
        now = datetime.now(timezone.utc)
        diff = now - dt
        seconds = int(diff.total_seconds())
        if seconds < 60:
            return f"{seconds}s ago"
        mins = seconds // 60
        if mins < 60:
            return f"{mins}m ago"
        hours = mins // 60
        if hours < 24:
            return f"{hours}h ago"
        days = hours // 24
        return f"{days}d ago"

    # ------------------------- data access ---------------------
    def fetch_all(self) -> List[Dict[str, Any]]:
        conn = self._get_conn(self._db_key)
        if not conn:
            return []
        try:
            with conn.cursor() as cur:
                cur.execute(f'SELECT * FROM "{self._table}";')
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
                return [{cols[i]: r[i] for i in range(len(cols))} for r in rows]
        except Exception:
            return []
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # ------------------------- queries -------------------------
    def urgent_queue(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Return open issues only, sorted by created_at desc when available."""
        items = self.fetch_all()
        enriched: List[Dict[str, Any]] = []
        for row in items:
            if self._norm_status(str(row.get("status") or "")) != "open":
                continue
            issue = row.get("id") or "Issue"
            customer = row.get("user_name") or "—"
            assigned_to = row.get("team_member_name") or "—"
            when_raw = row.get("created_at")
            dt = self._parse_dt(when_raw if when_raw is not None else None)
            enriched.append(
                {
                    "issue": str(issue),
                    "customer": str(customer),
                    "assigned_to": str(assigned_to),
                    "time_ago": self._time_ago_str(dt),
                    "_dt": dt,
                }
            )
        # sort by datetime desc when present
        enriched.sort(key=lambda x: (x.get("_dt") is None, x.get("_dt") or datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
        # strip helper key and return top N
        out = [{k: v for k, v in d.items() if k != "_dt"} for d in enriched[: max(0, int(limit))]]
        return out

    def team_performance(self) -> List[Dict[str, Any]]:
        items = self.fetch_all()
        by_member: Dict[str, Dict[str, Any]] = {}
        for row in items:
            member_id = str(row.get("team_member_id") or "-")
            member_name = str(row.get("team_member_name") or "Unknown")
            dept = str(row.get("department_name") or "")
            status = self._norm_status(str(row.get("status") or ""))
            agg = by_member.setdefault(
                member_id,
                {"member_id": member_id, "name": member_name, "dept": dept, "assigned": 0, "completed": 0},
            )
            agg["assigned"] += 1
            if status == "resolved":
                agg["completed"] += 1
        out: List[Dict[str, Any]] = []
        for agg in by_member.values():
            assigned = int(agg["assigned"]) or 0
            completed = int(agg["completed"]) or 0
            pct = int(round((completed / assigned) * 100)) if assigned > 0 else 0
            out.append(
                {
                    "member_id": agg["member_id"],
                    "name": agg["name"],
                    "dept": agg["dept"],
                    "assigned": assigned,
                    "completed": completed,
                    "pct": pct,
                }
            )
        out.sort(key=lambda x: (-x["assigned"], x["name"]))
        return out

    # ------------------------- mutations -----------------------
    def close_one_task(self, member_id: str) -> bool:
        conn = self._get_conn(self._db_key)
        if not conn:
            return False
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        'SELECT "id" FROM "{table}" WHERE "team_member_id"=%s '
                        'AND LOWER(COALESCE("status","")) NOT LIKE %s '
                        'AND LOWER(COALESCE("status","")) NOT LIKE %s '
                        'AND LOWER(COALESCE("status","")) NOT IN (%s, %s, %s) '
                        'ORDER BY "created_at" DESC NULLS LAST LIMIT 1;'.format(table=self._table),
                        (member_id, 'resolv%', 'clos%', 'done', 'complete', 'completed'),
                    )
                    row = cur.fetchone()
                    if not row:
                        return False
                    task_id = row[0]
                    cur.execute(f'UPDATE "{self._table}" SET "status"=%s WHERE "id"=%s;', ("resolved", task_id))
            return True
        except Exception:
            return False
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def reopen_one_task(self, member_id: str) -> bool:
        conn = self._get_conn(self._db_key)
        if not conn:
            return False
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        'SELECT "id" FROM "{table}" WHERE "team_member_id"=%s '
                        'AND (LOWER(COALESCE("status","")) LIKE %s '
                        'OR LOWER(COALESCE("status","")) LIKE %s '
                        'OR LOWER(COALESCE("status","")) IN (%s, %s, %s)) '
                        'ORDER BY "created_at" DESC NULLS LAST LIMIT 1;'.format(table=self._table),
                        (member_id, 'resolv%', 'clos%', 'done', 'complete', 'completed'),
                    )
                    row = cur.fetchone()
                    if not row:
                        return False
                    task_id = row[0]
                    cur.execute(f'UPDATE "{self._table}" SET "status"=%s WHERE "id"=%s;', ("open", task_id))
            return True
        except Exception:
            return False
        finally:
            try:
                conn.close()
            except Exception:
                pass
