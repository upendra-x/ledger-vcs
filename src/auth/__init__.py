"""Authorization: who may ask, and what they get if they do.

Grants are resolved at token-mint time and carried in the token, so a request
costs a signature verification and never a store read — which is what makes
14,000 reads per second affordable.
"""

from __future__ import annotations
