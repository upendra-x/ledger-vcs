"""A read-only browser for the corpus.

Ledger is built for automations — *"no caller is a human at a terminal"* — so a
The UI is not a first-class surface. It exists for the one job automations cannot
do: letting a person see what is actually in there when something looks wrong.

Which is why it is strictly read-only and strictly a *view*. It renders what the
same API returns, through the same authorization, and offers no button that
changes anything. A UI that could write would be a second write path, and the
whole argument for a single atomic point would have to be made
twice.

Server-rendered, with no template engine and no client-side framework. The pages
are lists and tables over data the services already produce, and a build step
would be more machinery than the pages are worth.
"""

from __future__ import annotations
