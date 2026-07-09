"""Carbon track (primary model).

* ``features``              - leakage-safe feature construction.
* ``climatology``          - historical climatology (Europe/Paris grouping).
* ``corrected_climatology`` - climatology + EWMA residual correction.
* ``model``                - project forecast model (RF / gradient boosting).
* ``evaluation``           - rolling-origin backtest and metrics.
"""
