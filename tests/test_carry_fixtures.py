"""BulletReveal's artifact builder (SCENE_SPEC.md §6).

Carry-in consumers (Callout, ZoomHighlight) build their fixtures against the
real BulletReveal builder (Component.fixture_carry_in), and the pipeline uses
the same builder at every cut. A builder that only approximates build() makes
the list visibly jump between the producing beat's last frame and the
consuming beat's first.
"""

from __future__ import annotations

import numpy as np

from chalkdust.continuity import ArtifactRecipe, CarryIn
from chalkdust.scenes.components import make_component
from chalkdust.validate.geometric import LayoutProbe

PARAMS = {"heading": "Three causes",
          "items": ["A weak hash function",
                    "A load factor left too high, so every chain keeps growing "
                    "until the table is resized",
                    "Adversarial keys chosen to collide"]}


def _by_label(scene, name):
    (mob,) = [m for m in scene.mobjects if getattr(m, "_chalk_label", None) == name]
    return mob


def test_carried_list_sits_exactly_where_its_beat_left_it():
    produced = LayoutProbe(make_component("BulletReveal", PARAMS), duration=8.0, strict=False)
    produced.construct()
    shown = _by_label(produced, "bullets")

    recipe = ArtifactRecipe(name="causes", producer="BulletReveal", params=PARAMS)
    consumer = CarryIn(make_component("TitleCard", {"title": "Next"}), [recipe])
    consumed = LayoutProbe(consumer, duration=8.0, strict=False)
    consumed.construct()
    carried = _by_label(consumed, "carried[causes]")

    # One part per item, in item order, each row (dot and text) at the same
    # place and size as on the producing beat's last frame.
    assert len(carried.submobjects) == len(shown.submobjects) == len(PARAMS["items"])
    for before, after in zip(shown.submobjects, carried.submobjects):
        np.testing.assert_allclose(after.get_center(), before.get_center(), atol=1e-6)
        np.testing.assert_allclose((after.width, after.height),
                                   (before.width, before.height), atol=1e-6)
