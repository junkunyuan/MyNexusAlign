## Code Style

Keep code simple.

## Docstrings

Write concise docstrings. Avoid unnecessary detail.

Good example:
class RemoveColorFilter(logging.Filter):
    """Strip ANSI color codes from log messages (for file output)."""

Good example:
class Meter:
    """A single windowed metric: a bounded value history plus its running mean.

    precision/notation control display: notation "e" prints scientific form
    (precision=2 -> 2.34e-02), "f" prints fixed-point (precision=4 -> 0.0234).
    """

Bad example:
class Meter:
    """Track one scalar metric over a fixed-size sliding window.

    Stores the most recent ``window_size`` numeric values in a bounded queue and
    recomputes ``mean`` as the arithmetic average of the stored window whenever
    ``update()`` (or ``update_peak()``) is called. ``latest`` returns the most
    recently appended value, or ``None`` when no data has been recorded.

    Display formatting is controlled by ``notation`` and ``precision``:
    - ``notation="e"``: scientific notation with ``precision`` mantissa digits
      (e.g., precision=2 -> 2.34e-02)
    - ``notation="f"``: fixed-point with ``precision`` decimal places
      (e.g., precision=4 -> 0.0234)
    """


## Module Docstrings

Add a concise module docstring at the top of each Python file with the format of "A: B".

Good Example: `"""Training meters: windowed metric tracking and hardware monitoring."""`

