"""
PINA Callbacks module with lazy loading for topology monitor.
"""
def TopologyMonitor(*args, **kwargs):
    """Lazy loader for TopologyMonitor to avoid import overhead."""
    from pina.topology.monitor import TopologyMonitor as _TopologyMonitor
    return _TopologyMonitor(*args, **kwargs)