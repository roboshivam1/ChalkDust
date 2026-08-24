"""Component library.

Importing this package registers every component. The compiler resolves a
BeatSpec's `component` string through `make_component`.
"""

from chalkdust.scenes.components.base import (  # noqa: F401
    Component,
    ComponentParams,
    get_component,
    make_component,
    register,
    registered_names,
)

# Import for side effect: each module calls @register at import time.
from chalkdust.scenes.components import bullet_reveal, title_card  # noqa: F401,E402

__all__ = [
    "Component",
    "ComponentParams",
    "get_component",
    "make_component",
    "register",
    "registered_names",
]
