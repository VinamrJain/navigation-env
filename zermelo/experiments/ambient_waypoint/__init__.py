"""Bayesian optimization over waypoints, against the problem where an unknown field displaces the actor.

- `schema` -- the settings one episode is built from.
- `build` -- one resolved configuration into the objects an episode runs on.
- `setup` -- what a sweep is written with.
- `registry` -- the world every study runs on.
- `sweeps` -- one module per sweep.
- `metrics` -- what a launch is read by.
- `render` -- what one episode is drawn as.
- `__main__` -- `python -m zermelo.experiments.ambient_waypoint +sweep=<name>`.
"""
