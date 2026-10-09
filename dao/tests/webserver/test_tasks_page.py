"""The Tasks page has to stay usable: controls compact, output readable.

Reported from a running add-on: "het uitvoervenster is niet goed zichtbaar,
de knoppen staan onder elkaar en vullen het hele scherm op". Four separate
things caused it, and each one gets an assertion here.

The two columns used Bootstrap's ``.col`` without a ``.row`` ancestor, inside
a ``<main>`` that is ``d-flex flex-column``. In a column-direction flexbox
``.col{flex:1 0 0}`` distributes *height*, so the two panes stacked and the
buttons pane -- twelve full-width rows, 644 px -- pushed the output off the
screen. On a 768 px viewport the output pane got less than nothing, and
``overflow:auto`` let it collapse to zero because a flex item with overflow
resolves ``min-height:auto`` to 0.
"""

import re

import pytest

from dao.prog import task_state
from dao.prog import tasks as task_registry
from dao.webserver.app.v2 import routes as v2_routes

from .conftest import INGRESS, SUPERVISOR

STYLESHEET = "dao/webserver/assets/main.scss"
TASKS_TEMPLATE = "dao/webserver/app/templates/v2/tasks.html"
STATUS_TEMPLATE = "dao/webserver/app/templates/v2/task-status.html"


@pytest.fixture
def page(client):
    response = client.get("/v2/tasks", headers=INGRESS, environ_base=SUPERVISOR)
    assert response.status_code == 200
    return response.data.decode()


@pytest.fixture
def fragment(client):
    """The console's contents, as the poll delivers them."""
    response = client.get("/v2/task-state", headers=INGRESS, environ_base=SUPERVISOR)
    assert response.status_code == 200
    return response.data.decode()


def _read(repo_file: str) -> str:
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    return (root / repo_file).read_text(encoding="utf-8")


class TestGrouping:
    """Twelve long Dutch labels in one flat list are not scannable."""

    def test_every_task_on_the_page_sits_in_exactly_one_group(self):
        grouped = [
            entry["key"]
            for group in v2_routes.task_page_groups()
            for entry in group["tasks"]
        ]
        assert len(grouped) == len(set(grouped)), "a task is listed twice"
        assert grouped == [e["key"] for e in v2_routes.task_page_entries()]

    def test_the_groups_cover_the_whole_registry(self):
        grouped = {
            entry["key"]
            for group in v2_routes.task_page_groups()
            for entry in group["tasks"]
        }
        assert grouped == set(task_registry.TASKS)

    def test_every_group_has_a_name_and_is_not_empty(self):
        groups = v2_routes.task_page_groups()
        assert len(groups) >= 2
        for group in groups:
            assert group["name"].strip()
            assert group["tasks"]

    def test_the_group_names_appear_on_the_page(self, page):
        for group in v2_routes.task_page_groups():
            assert group["name"] in page


class TestLayout:
    def test_the_buttons_share_a_row_instead_of_one_per_line(self, page):
        """The regression itself.

        Each task used to be wrapped in its own ``d-flex`` row, so nothing
        could share a line however wide the window was. A responsive grid
        turns twelve rows into four.
        """
        assert "row-cols-" in page, "the task buttons are not laid out in a grid"
        template = _read(TASKS_TEMPLATE)
        assert "d-flex align-items-center gap-2 flex-wrap" not in template

    def test_no_column_is_used_outside_a_row(self, page):
        """``.col`` in a column-direction flexbox divides height, not width.

        That is what made the buttons pane claim the viewport. Bootstrap's
        column classes only mean anything inside a ``.row``.
        """
        body = page.split("</nav>", 1)[-1].split("<footer", 1)[0]
        depth = 0
        for tag in re.finditer(r'<(/?)div\b([^>]*)>', body):
            closing, attributes = tag.group(1), tag.group(2)
            if closing:
                depth = max(0, depth - 1)
                continue
            classes = re.search(r'class="([^"]*)"', attributes)
            classes = classes.group(1).split() if classes else []
            if "row" in classes:
                depth += 1
                continue
            bare = [c for c in classes if c == "col" or c.startswith("col-")]
            assert not bare or depth > 0, f"{bare} sits outside a .row"
            depth += 1

    def test_the_output_pane_cannot_collapse(self, page):
        """It needs a height floor of its own.

        ``flex-grow-1`` alone gave it whatever the buttons left over, which
        on a laptop was nothing at all.
        """
        assert 'id="status-target"' in page
        assert "task-console" in page
        stylesheet = _read(STYLESHEET)
        console = stylesheet.split(".task-console", 1)[-1].split("\n}", 1)[0]
        assert "min-height" in console, "the console has no height floor"

    def test_the_log_wraps_instead_of_scrolling_sideways(self):
        """A bare <pre> keeps ``white-space: pre``.

        Log lines here run to about a hundred characters, so without a wrap
        rule the end of every line sat behind a horizontal scrollbar.
        """
        stylesheet = _read(STYLESHEET)
        assert "white-space: pre-wrap" in stylesheet

    def test_the_status_header_does_not_sit_inside_the_scroller(self, fragment):
        """It was ``sticky-top`` inside the pane it was pinned to.

        That pinned about a hundred pixels of heading inside a pane that had
        barely a hundred pixels to give. The bar is now a sibling of the
        scrolling log instead of a passenger in it.
        """
        assert "sticky-top" not in fragment
        assert "task-console-bar" in fragment
        assert "task-console-log" in fragment


class TestCancel:
    def test_cancel_returns_a_fragment_not_the_whole_page(self, client):
        """It rendered ``tasks.html`` into the output pane.

        htmx swapped a complete document -- nav, main, footer, a second
        ``id="status-target"`` and a second cancel button -- into the small
        pane the output was supposed to use.
        """
        task_state.claim("calc_optimum", source="dashboard")

        response = client.get(
            "/v2/task-cancel", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 200
        body = response.data.decode()
        assert "<nav" not in body
        assert "<footer" not in body
        assert body.count('id="status-target"') == 0

    def test_the_cancel_button_lives_with_the_running_task(self, client, page):
        """It acts on the task that is running, so it belongs in the console.

        Sitting under the task picker, it needed two inline scripts reaching
        across the page to enable and disable itself on every poll.
        """
        assert "run-cancel" not in page
        assert "<script" not in _rendered_fragment(client)

        task_state.claim("calc_optimum", source="dashboard")
        running = _rendered_fragment(client)
        assert "task-cancel" in running, "a running task offers no way to stop it"

        task_state.release("calc_optimum", "done")
        finished = _rendered_fragment(client)
        assert "task-cancel" not in finished, "nothing is running to cancel"


def _rendered_fragment(client) -> str:
    response = client.get("/v2/task-state", headers=INGRESS, environ_base=SUPERVISOR)
    assert response.status_code == 200
    return response.data.decode()
