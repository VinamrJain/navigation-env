# zermelo

An actor is carried by a flow it cannot see. It steers a little, the field does the rest, and it has to decide
where to go while it is still learning where the flow runs — a stratospheric balloon holding station, an ocean
drifter dropped to sample a current. The name is Zermelo's navigation problem, with the field unknown.

The state is a tree of named parts: some the actor steers, some the field displaces, and one part **is** the
field, which the actor never sees. Every move it emits a pair `(action, claim)` — where to go, and what it now
holds the field to be. The claim can be scored against the truth to evaluate the quality of the posterior estimate.

The package publishes the contracts, problems, methods, and the harness that runs them
across seeds on a cluster.

## Install and check

    pixi install                      # https://pixi.sh
    pixi run check                    # ruff, mypy, import-linter. Static: runs no episode
    pixi run smoke                    # runs a sweep end to end and draws it. The only thing that verifies a plot
    pixi run -e cuda smoke-gpu        # the same cells as one Slurm array on a gpu node
    pixi run crosscheck               # our acquisitions against botorch's, on a problem both can express

Python 3.14 or newer (uses PEP 695 generics)

## Run

    pixi run sweep <name>             # Lauches a sweep locally
    pixi run submit <name>            # Launches a sweep as one Slurm array
    pixi run draw <directory>         # draw the figures for the launch
    pixi run render <cell>            # Snapshot of an episode at a particular time
    pixi run film <cell>              # the video of the entire episode (per run individually or per seed or even across the sweep)

`pixi task list` names every task, and every sweep is one module under
`zermelo/experiments/ambient_waypoint/sweeps/`. Put `--` before any override; 

    pixi run submit acquisitions -- resources.partition=<partition> resources.cpus=2
    pixi run -e cuda submit acquisitions -- resources.gres=gpu:1

## Layout

    zermelo/interface/     The contracts. Imports stdlib, jax and jaxtyping.
    zermelo/problems/      one package per problem, its modules mirroring the contracts they implement
    zermelo/methods/       one directory per method family;
    zermelo/run/           the driver, the record and the cluster settings. Generic in problem and in method
    zermelo/experiments/   one package per pairing of a problem with a method family.
