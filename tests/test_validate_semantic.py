"""Semantic validation (rung 2) and the Component hooks it reads."""

from __future__ import annotations

import pytest

from chalkdust.scenes.components import get_component, registered_names


class TestHookDefaults:
    """The hooks are optional: a component that ignores them must get the
    permissive defaults, never an AttributeError or a refusal."""

    @pytest.mark.parametrize("name", registered_names())
    def test_defaults_are_permissive(self, name):
        cls = get_component(name)
        if not cls.examples():
            pytest.skip(f"{name} declares no examples")
        component = cls(cls.examples()[0])
        assert component.min_seconds() >= 0.0
        assert isinstance(component.latex_strings(), list)
