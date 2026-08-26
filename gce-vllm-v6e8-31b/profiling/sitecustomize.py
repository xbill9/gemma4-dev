"""Auto-imported by Python from any sys.path entry; bind-mounted into the vLLM container.

Kept to one guarded call on purpose. sitecustomize runs during interpreter startup, so an
exception here does not fail this file — it fails the container. profiler_sidecar.install()
is itself total (it catches everything and returns a bool), and the belt-and-braces try here
covers the import going wrong too.
"""

try:
    import profiler_sidecar

    profiler_sidecar.install()
except Exception:  # noqa: BLE001 - a profiler must never break serving
    pass
