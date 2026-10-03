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
from chalkdust.scenes.components import answer_box  # noqa: F401,E402
from chalkdust.scenes.components import box_flow  # noqa: F401,E402
from chalkdust.scenes.components import bullet_reveal  # noqa: F401,E402
from chalkdust.scenes.components import data_structure_viz  # noqa: F401,E402
from chalkdust.scenes.components import equation_derivation  # noqa: F401,E402
from chalkdust.scenes.components import free_body_diagram  # noqa: F401,E402
from chalkdust.scenes.components import geometry_construct  # noqa: F401,E402
from chalkdust.scenes.components import graph_plot  # noqa: F401,E402
from chalkdust.scenes.components import number_line_walk  # noqa: F401,E402
from chalkdust.scenes.components import problem_statement  # noqa: F401,E402
from chalkdust.scenes.components import raw_scene  # noqa: F401,E402
from chalkdust.scenes.components import solution_step  # noqa: F401,E402
from chalkdust.scenes.components import split_compare  # noqa: F401,E402
from chalkdust.scenes.components import step_trace  # noqa: F401,E402
from chalkdust.scenes.components import title_card  # noqa: F401,E402
from chalkdust.scenes.components import vector_field  # noqa: F401,E402

__all__ = [
    "Component",
    "ComponentParams",
    "get_component",
    "make_component",
    "register",
    "registered_names",
]
