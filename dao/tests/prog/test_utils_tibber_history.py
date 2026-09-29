"""get_tibber_data melted Tibber's production/consumption nodes into
(time, code, value) rows with tibber_df.loc[tibber_df.shape[0]] = row per
node/field pair. Rewritten to collect a plain list of tuples and build the
frame once; this test pins the resulting values down.
"""

import datetime
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("pandas")

import dao.prog.config.loader as loader_module  # noqa: E402
import dao.lib.db_connections as db_connections_module  # noqa: E402
import dao.prog.utils as utils  # noqa: E402


class FakeDb:
    def __init__(self):
        self.saved = None

    def savedata(self, df, tablename="values"):
        self.saved = df


@pytest.fixture
def tibber_env(monkeypatch):
    secrets = {}
    tibber_options = SimpleNamespace(
        api_url=None,
        api_token=SimpleNamespace(resolve=lambda s: "tok"),
    )
    # Six days back so (now - start) / 3600 comfortably exceeds the
    # function's own "count < 24 hours: skip" guard.
    last_invoice = (datetime.date.today() - datetime.timedelta(days=6))
    config = SimpleNamespace(
        tibber=tibber_options,
        prices=SimpleNamespace(last_invoice=last_invoice),
    )

    class FakeLoader:
        def __init__(self, config_path, secrets_path=None):
            self.secrets = secrets

        def load_and_validate(self):
            return config

    db = FakeDb()
    monkeypatch.setattr(loader_module, "ConfigurationLoader", FakeLoader)
    monkeypatch.setattr(db_connections_module, "make_db_da", lambda cfg, sec: db)
    # An explicit start date takes the simple branch and skips the
    # "query the values table for gaps" path entirely.
    start_str = (datetime.date.today() - datetime.timedelta(days=2)).strftime(
        "%Y-%m-%d"
    )
    monkeypatch.setattr(sys, "argv", ["day_ahead.py", "tibber", start_str])
    return db


def _node(day_offset, hour, **fields):
    when = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        days=day_offset
    )
    when = when.replace(hour=hour, minute=0, second=0, microsecond=0)
    return {"from": when.strftime("%Y-%m-%dT%H:%M:%S.000+00:00"), **fields}


def test_production_and_consumption_nodes_are_melted_into_rows(
    tibber_env, monkeypatch
):
    db = tibber_env
    payload = {
        "data": {
            "viewer": {
                "homes": [
                    {
                        "production": {
                            "nodes": [
                                _node(2, 10, production=1.5, profit=0.20),
                                _node(2, 11, production=None, profit=0.05),
                            ]
                        },
                        "consumption": {
                            "nodes": [
                                _node(2, 10, consumption=2.5, cost=0.40),
                            ]
                        },
                    }
                ]
            }
        }
    }

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    monkeypatch.setattr(utils, "post", lambda *a, **kw: FakeResponse())

    utils.get_tibber_data()

    assert db.saved is not None
    assert set(db.saved["code"]) == {"prod", "profit", "cons", "cost"}
    # 2 production nodes -> 1 "prod" row (the second has production=None)
    # plus 2 "profit" rows (both nodes have a profit); 1 consumption node
    # -> 1 "cons" row + 1 "cost" row. 5 rows total.
    assert len(db.saved) == 5
    prod_row = db.saved[db.saved["code"] == "prod"].iloc[0]
    assert prod_row["value"] == pytest.approx(1.5)
    assert sorted(db.saved[db.saved["code"] == "profit"]["value"]) == pytest.approx(
        [0.05, 0.20]
    )
    cons_row = db.saved[db.saved["code"] == "cons"].iloc[0]
    assert cons_row["value"] == pytest.approx(2.5)


def test_a_node_with_no_missing_fields_omitted_is_still_only_one_row_each(
    tibber_env, monkeypatch
):
    """A None value for one field of a node (e.g. profit not yet settled)
    must not add a row for that field, but must not skip the sibling
    field either."""
    db = tibber_env
    payload = {
        "data": {
            "viewer": {
                "homes": [
                    {
                        "production": {
                            "nodes": [_node(2, 10, production=1.0, profit=None)]
                        },
                        "consumption": {
                            "nodes": [_node(2, 10, consumption=None, cost=0.30)]
                        },
                    }
                ]
            }
        }
    }

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    monkeypatch.setattr(utils, "post", lambda *a, **kw: FakeResponse())

    utils.get_tibber_data()

    assert set(db.saved["code"]) == {"prod", "cost"}
    assert len(db.saved) == 2
