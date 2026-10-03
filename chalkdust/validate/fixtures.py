"""A component's declared fixtures, as the registry-wide walks build them.

tests/test_layout.py and the snapshot harness (SCENE_SPEC.md §11 rule 6) walk
every registered component's fixtures. Most build from params alone. A
component that acts on a carried artifact (SCENE_SPEC.md §6) -- ZoomHighlight,
Callout -- cannot: its build() fetches the artifact with continuity.carried(),
which raises CarryInError on a bare build. Its fixtures therefore carry their
artifacts' recipes (Component.carried_examples / carried_stress), and are built
the way the render builds them: through continuity.beat_component, with the
artifacts on screen first.

A recipe's producer must be able to rebuild its artifact. Until a producer
ships its own @artifact_builder, the consuming component's fixtures may lend
one (Component.fixture_builders); `lent_builders` installs those only for the
duration of a build and never over a real one.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

from chalkdust import continuity
from chalkdust.continuity import ArtifactRecipe
from chalkdust.core.models import BeatSpec
from chalkdust.scenes.components import get_component

Kind = Literal["examples", "stress"]


@dataclass(frozen=True)
class FixtureCase:
    """One fixture: the component's params, and the artifacts it carries in
    (empty for a component that builds from params alone)."""

    params: dict[str, Any]
    carry_in: tuple[ArtifactRecipe, ...] = ()

    def spec(self, component: str, beat_id: str = "b01") -> BeatSpec:
        """The beat this fixture stands for, carry_in included."""
        return BeatSpec(id=beat_id, narration="placeholder narration",
                        component=component, params=self.params,
                        carry_in=[r.name for r in self.carry_in])

    def carry_in_json(self) -> list[dict[str, Any]]:
        return [r.model_dump(mode="json") for r in self.carry_in]


def fixture_cases(name: str, kind: Kind) -> list[FixtureCase]:
    """Every `kind` fixture of component `name`: its plain ones (examples() /
    stress()), then its carried ones (carried_examples() / carried_stress())."""
    cls = get_component(name)
    plain = [FixtureCase(params) for params in getattr(cls, kind)()]
    carried = [
        FixtureCase(case["params"],
                    tuple(ArtifactRecipe.model_validate(r) for r in case["carry_in"]))
        for case in getattr(cls, f"carried_{kind}")()
    ]
    return plain + carried


@contextmanager
def lent_builders(name: str) -> Iterator[None]:
    """Lend component `name`'s fixture builders to producers that have no
    artifact builder of their own, for the duration of the block.

    A producer's real builder always wins, so a fixture never builds an
    artifact the render would not. The registry is restored on exit, whatever
    happens inside, so nothing lent leaks into a real resolve_carry_in.
    """
    registry = continuity._BUILDERS
    lent = {producer: fn for producer, fn in get_component(name).fixture_builders().items()
            if producer not in registry}
    registry.update(lent)
    try:
        yield
    finally:
        for producer in lent:
            registry.pop(producer, None)
