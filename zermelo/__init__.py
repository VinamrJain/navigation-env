"""Sequential decision-making under partially-unknown dynamics.

An actor occupies a state that is a tree of named parts: some it steers directly, others an
unknown latent displaces, and one of them *is* that latent. It reads the state through a
problem-supplied readout, and it is scored on an objective that may reward reaching somewhere,
learning the latent, or both.
"""

from beartype.claw import beartype_this_package as _install_runtime_checks

_install_runtime_checks()  # aliased private: this namespace re-exports nothing, not even this

__version__ = "0.1.0"
