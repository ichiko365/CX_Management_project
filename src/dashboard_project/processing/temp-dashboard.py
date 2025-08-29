"""
Quick runtime checks for processing.team_task functions.

Run from project root:
    python -m tests.run_team_task_checks
or:
    python tests/run_team_task_checks.py

Notes:
- These tests use your real PostgreSQL databases. Ensure DB credentials in database/connector.py are correct.
- toggle_task_status is potentially destructive (it updates rows). It's provided commented-out; only enable it after reviewing and choosing a safe team_member_id.
- Each block prints a short assurance message indicating success or the caught exception.
"""

import sys
import os
import traceback

# Ensure package imports resolve (adjust if your workspace root differs)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src", "dashboard_project")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Import target functions
from team_task import (
    refresh_data,
    get_support_data,
    process_urgent_queue,
    calculate_performance_metrics,
    get_queue_summary,
    fetch_team_performance_data,
    calculate_team_efficiency,
    toggle_task_status,
    # backward compatibility wrappers
    get_complaints_data,
    refresh_complaints_data,
)

def run_check(fn, name, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
        print(f"[ OK ] {name} executed without exception.")
        return result
    except Exception as e:
        print(f"[FAIL] {name} raised an exception: {e}")
        traceback.print_exc()
        return None

def main():
    print("\n=== TEAM TASKS RUNTIME CHECKS ===\n")

    # 1) Refresh / sync data from customer DB into main DB
    # This will call sync_complaints_from_customer_db and may write to your main DB.
    # print("1) Running refresh_data() -> sync complaints into main DB")
    # refresh_result = run_check(refresh_data, "refresh_data")
    # If refresh_result is True, sync completed (rows processed printed by function).
    # If None or False, there was an error or nothing processed.

    # 2) Fetch support/complaints table from main DB
    # print("\n2) Fetching support data using get_support_data()")
    # df_support = run_check(get_support_data, "get_support_data")
    # if df_support is not None:
    #     try:
    #         print(f"    -> Returned DataFrame with shape: {getattr(df_support, 'shape', 'N/A')}")
    #         print(f"    -> Columns: {list(getattr(df_support, 'columns', []))}")
    #     except Exception:
    #         pass

    # # 3) Process urgent queue display
    # print("\n3) Processing urgent queue with process_urgent_queue(df)")
    # processed_queue = run_check(process_urgent_queue, "process_urgent_queue", df_support or None)
    # if processed_queue is not None:
    #     try:
    #         print(f"    -> Processed queue shape: {processed_queue.shape}")
    #         print(f"    -> Display columns: {list(processed_queue.columns)}")
    #     except Exception:
    #         pass

    # # 4) Calculate performance metrics
    # print("\n4) Calculating performance metrics with calculate_performance_metrics(df)")
    # metrics = run_check(calculate_performance_metrics, "calculate_performance_metrics", df_support or None)
    # if metrics is not None:
    #     try:
    #         print(f"    -> Metrics keys: {list(metrics.keys())}")
    #         # Print summary values for assurance (not exhaustive)
    #         for k in ["total_items", "unresolved_items", "recent_24h"]:
    #             if k in metrics:
    #                 print(f"       - {k}: {metrics[k]}")
    #     except Exception:
    #         pass

    # # 5) Get queue summary from processed queue
    # print("\n5) Getting queue summary using get_queue_summary(processed_queue)")
    # summary = run_check(get_queue_summary, "get_queue_summary", processed_queue or None)
    # if summary is not None:
    #     print(f"    -> Summary: {summary}")

    # # 6) Fetch team performance data from customer DB
    # print("\n6) Fetching team performance data with fetch_team_performance_data()")
    # team_df = run_check(fetch_team_performance_data, "fetch_team_performance_data")
    # if team_df is not None:
    #     try:
    #         print(f"    -> Team performance shape: {team_df.shape}")
    #         print(f"    -> Columns: {list(team_df.columns)}")
    #     except Exception:
    #         pass

    # # 7) Calculate team efficiency
    # print("\n7) Calculating team efficiency with calculate_team_efficiency(team_df)")
    # efficiency = run_check(calculate_team_efficiency, "calculate_team_efficiency", team_df or None)
    # if efficiency is not None:
    #     print(f"    -> Team efficiency: {efficiency}%")

    # # 8) (Optional) Toggle a task status for a given team_member_id
    # print("\n8) toggle_task_status(team_member_id) - DISABLED by default (unsafe write).")
    # print("   If you want to test it, uncomment the lines below and provide a safe team_member_id.")
    # # Example to enable (careful!):
    # # safe_member_id = 1  # replace with a real team_member_id that you are ok updating
    # # toggle_result = run_check(toggle_task_status, "toggle_task_status", safe_member_id)
    # # print(f"    -> toggle_task_status result: {toggle_result}")

    # # 9) Backwards compatibility wrappers (basic smoke checks)
    # print("\n9) Running backward compatibility wrappers (smoke tests)")
    # run_check(get_complaints_data, "get_complaints_data")
    # run_check(refresh_complaints_data, "refresh_complaints_data")

    # print("\n=== CHECKS COMPLETE ===\n")
    # print("If most items show [ OK ], the functions are callable and ran without unhandled exceptions.")
    # print("If you see [FAIL], inspect the traceback above for details and fix DB connectivity / credentials / permissions.")

if __name__ == "__main__":
    main()