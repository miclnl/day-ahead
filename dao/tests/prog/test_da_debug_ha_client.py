"""ReplayIO's fake Home Assistant client (_FakeHAClient) is the seam that
lets DaBase.__init__ construct without any real network access: it stands in
for homeassistant_api.Client, and only needs get_config() to work, since
that's the one call __init__ makes directly (to resolve
latitude/longitude/time_zone) before get_state/call_service/set_state get
their own, separate patches.

No existing test exercises ReplayIO/RecordingIO end to end (they need a full
snapshot fixture, config, database, etc.), so this test targets exactly the
piece that changed when hassapi was replaced with homeassistant_api: the
patch that intercepts DaBase.__init__'s `self._ha_client = HAClient(...)`
construction.
"""

import dao.prog.da_base as da_base_module
import dao.prog.da_debug as da_debug


def test_fake_ha_client_returns_the_bound_config():
    ha_context = {
        "latitude": 52.1,
        "longitude": 5.2,
        "time_zone": "Europe/Amsterdam",
        "country": "NL",
    }

    client = da_debug._FakeHAClient(ha_context)

    assert client.get_config() == ha_context


def test_patching_haclient_in_da_base_module_intercepts_construction(monkeypatch):
    """Mirrors exactly what ReplayIO._install_patches() does: replace the
    HAClient name da_base.py imported into its own module namespace with a
    factory that ignores the real constructor arguments and always returns
    a client bound to the canned config."""
    ha_context = {
        "latitude": 1.0,
        "longitude": 2.0,
        "time_zone": "UTC",
        "country": "NL",
    }
    monkeypatch.setattr(
        da_base_module,
        "HAClient",
        lambda *a, **k: da_debug._FakeHAClient(ha_context),
    )

    # The exact call DaBase.__init__ makes: construct with real-looking
    # arguments, then call get_config() on the result.
    client = da_base_module.HAClient(
        api_url="http://192.168.1.10:8123/api",
        token="does-not-matter",
        global_request_kwargs={"timeout": 10},
    )

    assert client.get_config() == ha_context
