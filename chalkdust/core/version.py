"""Version constants that participate in cache keys.

Changing any value here invalidates every downstream cached artifact. That is
the point: if rendering behaviour changes, stale artifacts must not be reused.
See SCENE_SPEC.md §10 for the versioning policy.
"""

CHALKDUST_VERSION = "0.1.0"

# Bump when a component's layout or visual behaviour changes.
COMPONENT_LIBRARY_VERSION = "0.1.0"

# Bump when a theme's palette, typography, or motion language changes.
THEME_VERSION = "0.1.0"

# Pinned deliberately. Manim breaks across versions, and an upgrade must not
# silently reuse renders produced by the old one.
#
# 0.21.0 line notes:
#   - Since 0.19.0 Manim uses pyav internally instead of shelling out to
#     ffmpeg. We still need system ffmpeg for our own concat/mux/loudnorm.
#   - The `Code` mobject was reworked in the 0.19 line. Relevant when we
#     build the CodeWalk component in Phase 3 -- pre-0.19 examples online
#     will not work as written.
MANIM_VERSION = "0.21.0"