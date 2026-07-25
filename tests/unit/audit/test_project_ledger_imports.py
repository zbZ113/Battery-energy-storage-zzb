from __future__ import annotations

import subprocess
import sys


def test_api_service_and_sql_project_ledger_import_in_fresh_process() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from quanxin_life.api.service import ToolInvocationService; "
                "from quanxin_life.audit.sql_project_ledger import "
                "SqlProjectAuditLedger; "
                "assert ToolInvocationService and SqlProjectAuditLedger"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
