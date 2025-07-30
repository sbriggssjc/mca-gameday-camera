"""Utility for uploading play counts to a Google Sheet."""

from __future__ import annotations

import datetime as _dt
from typing import Iterable, List

from roster import get_player_name

import gspread
from google.oauth2.service_account import Credentials


SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def upload_rows(service_account_json: str, spreadsheet_id: str, rows: Iterable[List[str]]) -> None:
    """Append rows of data to the first worksheet of a spreadsheet."""
    creds = Credentials.from_service_account_file(service_account_json, scopes=SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(spreadsheet_id)
    ws = sh.sheet1
    ws.append_rows(list(rows), value_input_option="RAW")


def format_row(jersey: int, total: int, quarters: List[int]) -> List[str]:
    """Return a row for uploading play counts including the player name."""
    timestamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    breakdown = ",".join(str(q) for q in quarters)
    return [timestamp, str(jersey), get_player_name(jersey), str(total), breakdown]
