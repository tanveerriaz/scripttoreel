# Ensure real pydub is imported into sys.modules before any test file can
# inject a MagicMock stub (test_production_plan.py stubs pydub globally if
# it's not already present, which leaks into later tests like test_story_3.py).
try:
    import pydub  # noqa: F401
except ImportError:
    pass
