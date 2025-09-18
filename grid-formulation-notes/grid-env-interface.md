---
title: "Grid Environment Project Template"
author: "Vinamr Jain"
date: "September 18, 2025"
toc: true
abstract: "Minimal project structure and abstract class definitions for implementing the discrete grid environment with gymnasium interface."
---

# Project Structure

```
grid_env/
├── grid_env/
│   ├── __init__.py
│   ├── environment.py          # Main gymnasium environment
│   ├── field/
│   │   ├── __init__.py
│   │   ├── abstract_field.py   # Abstract field interface
│   │   └── implementations/
│   │       ├── __init__.py
│   │       ├── simple_field.py # Basic field implementation
│   │       └── gaussian_field.py
│   ├── actor/
│   │   ├── __init__.py
│   │   ├── abstract_actor.py   # Abstract actor interface
│   │   └── grid_actor.py       # Basic grid actor implementation
│   └── utils/
│       ├── __init__.py
│       └── types.py            # Common type definitions
├── tests/
│   ├── __init__.py
│   ├── test_environment.py
│   ├── test_field.py
│   └── test_actor.py
├── examples/
│   ├── basic_usage.py
│   └── random_agent.py
├── setup.py
└── README.md
```

# Core Type Definitions

## `grid_env/utils/types.py`

```python
from typing import NamedTuple, Tuple, Dict, Any
import numpy as np

class GridPosition(NamedTuple):
    """3D grid position coordinates."""
    i: int  # x-coordinate
    j: int  # y-coordinate  
    k: int  # z-coordinate

class Displacement(NamedTuple):
    """2D horizontal displacement."""
    u: int  # x-displacement
    v: int  # y-displacement

class VerticalAction(NamedTuple):
    """Vertical control action."""
    action: int  # -1: down, 0: stay, +1: up

class GridConfig(NamedTuple):
    """Grid environment configuration."""
    n_x: int
    n_y: int  
    n_z: int
    d_max: int  # Maximum displacement magnitude

# Type aliases
State = Dict[str, Any]
Observation = Dict[str, Any]
Info = Dict[str, Any]
```

# Abstract Field Interface

## `grid_env/field/abstract_field.py`

```python
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import numpy as np
from ..utils.types import GridPosition, Displacement, GridConfig

class AbstractField(ABC):
    """Abstract base class for environmental fields."""
    
    def __init__(self, config: GridConfig, seed: int = None):
        """Initialize field with configuration and random seed."""
        self.config = config
        self.seed = seed
        self._rng = np.random.RandomState(seed)
    
    @abstractmethod
    def reset(self, seed: int = None) -> None:
        """Reset/regenerate the field configuration."""
        pass
    
    @abstractmethod
    def sample_displacement(self, position: GridPosition) -> Displacement:
        """Sample horizontal displacement at given position."""
        pass
    
    @abstractmethod
    def get_field_state(self) -> Dict[str, Any]:
        """Return complete field state (for analysis/debugging)."""
        pass
    
    @abstractmethod
    def get_displacement_pmf(self, position: GridPosition) -> np.ndarray:
        """Return displacement PMF at given position."""
        pass
    
    def is_valid_position(self, position: GridPosition) -> bool:
        """Check if position is within grid bounds."""
        return (1 <= position.i <= self.config.n_x and
                1 <= position.j <= self.config.n_y and
                1 <= position.k <= self.config.n_z)
    
    def enforce_boundaries(self, position: GridPosition, 
                          displacement: Displacement) -> GridPosition:
        """Apply displacement with boundary enforcement."""
        new_i = max(1, min(position.i + displacement.u, self.config.n_x))
        new_j = max(1, min(position.j + displacement.v, self.config.n_y))
        return GridPosition(new_i, new_j, position.k)
```

# Abstract Actor Interface  

## `grid_env/actor/abstract_actor.py`

```python
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import numpy as np
from ..utils.types import GridPosition, VerticalAction, GridConfig

class AbstractActor(ABC):
    """Abstract base class for actors in the grid environment."""
    
    def __init__(self, config: GridConfig, initial_position: GridPosition):
        """Initialize actor with configuration and starting position."""
        self.config = config
        self.position = initial_position
        self._initial_position = initial_position
    
    @abstractmethod
    def step_vertical(self, action: VerticalAction) -> GridPosition:
        """Execute vertical action and return new position."""
        pass
    
    @abstractmethod
    def get_vertical_displacement_pmf(self, action: VerticalAction) -> np.ndarray:
        """Return PMF over vertical displacements for given action."""
        pass
    
    def reset(self, position: GridPosition = None) -> GridPosition:
        """Reset actor to initial or specified position."""
        self.position = position or self._initial_position
        return self.position
    
    def is_valid_position(self, position: GridPosition) -> bool:
        """Check if position is within grid bounds."""
        return (1 <= position.i <= self.config.n_x and
                1 <= position.j <= self.config.n_y and
                1 <= position.k <= self.config.n_z)
    
    def enforce_vertical_boundaries(self, new_k: int) -> int:
        """Enforce vertical boundary constraints."""
        return max(1, min(new_k, self.config.n_z))
    
    def get_state(self) -> Dict[str, Any]:
        """Return current actor state."""
        return {
            'position': self.position,
            'initial_position': self._initial_position
        }
```

# Main Environment Interface

## `grid_env/environment.py`

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional

from .utils.types import GridPosition, Displacement, VerticalAction, GridConfig
from .field.abstract_field import AbstractField
from .actor.abstract_actor import AbstractActor

class GridEnvironment(gym.Env):
    """Gymnasium environment for discrete grid world with stochastic fields."""
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                 field: AbstractField,
                 actor: AbstractActor,
                 config: GridConfig,
                 reward_fn: callable = None,
                 max_steps: int = 1000):
        """
        Initialize grid environment.
        
        Args:
            field: Environmental field instance
            actor: Actor instance  
            config: Grid configuration
            reward_fn: Reward function R(state, action) -> float
            max_steps: Maximum episode length
        """
        super().__init__()
        
        self.field = field
        self.actor = actor
        self.config = config
        self.reward_fn = reward_fn or self._default_reward
        self.max_steps = max_steps
        
        # Action space: {-1, 0, +1} for vertical control
        self.action_space = spaces.Discrete(3)
        
        # Observation space: position + local displacement observation
        self.observation_space = spaces.Dict({
            'position': spaces.Box(
                low=np.array([1, 1, 1]), 
                high=np.array([config.n_x, config.n_y, config.n_z]),
                dtype=np.int32
            ),
            'local_displacement': spaces.Box(
                low=np.array([-config.d_max, -config.d_max]),
                high=np.array([config.d_max, config.d_max]),
                dtype=np.int32
            )
        })
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None, 
              options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset field and actor
        self.field.reset(seed=seed)
        self.actor.reset()
        
        # Reset episode counters
        self.step_count = 0
        self.done = False
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        if self.done:
            raise RuntimeError("Environment is done. Call reset() first.")
        
        # Convert action to vertical action
        vertical_action = VerticalAction(action - 1)  # Map {0,1,2} -> {-1,0,+1}
        
        # Sample horizontal displacement from field
        horizontal_displacement = self.field.sample_displacement(self.actor.position)
        
        # Apply horizontal displacement (with boundary enforcement)
        new_position_horizontal = self.field.enforce_boundaries(
            self.actor.position, horizontal_displacement)
        
        # Update actor position horizontally
        self.actor.position = GridPosition(
            new_position_horizontal.i, 
            new_position_horizontal.j, 
            self.actor.position.k
        )
        
        # Execute vertical action
        self.actor.position = self.actor.step_vertical(vertical_action)
        
        # Compute reward
        reward = self.reward_fn(self._get_state(), action)
        
        # Update episode counters
        self.step_count += 1
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_steps
        self.done = terminated or truncated
        
        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        info['horizontal_displacement'] = horizontal_displacement
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict[str, Any]:
        """Construct observation from current state."""
        # Sample displacement observation (what the agent observes)
        local_displacement = self.field.sample_displacement(self.actor.position)
        
        return {
            'position': np.array([self.actor.position.i, 
                                 self.actor.position.j, 
                                 self.actor.position.k]),
            'local_displacement': np.array([local_displacement.u, 
                                          local_displacement.v])
        }
    
    def _get_state(self) -> Dict[str, Any]:
        """Return complete environment state."""
        return {
            'actor': self.actor.get_state(),
            'field': self.field.get_field_state(),
            'step_count': self.step_count
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Return additional information."""
        return {
            'step_count': self.step_count,
            'position': self.actor.position,
            'is_terminal': self._is_terminated()
        }
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Override in subclasses for specific termination conditions
        return False
    
    def _default_reward(self, state: Dict[str, Any], action: int) -> float:
        """Default reward function (override for specific objectives)."""
        return 0.0
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            print(f"Step: {self.step_count}, Position: {self.actor.position}")
        elif mode == 'rgb_array':
            # Return RGB array representation (implement as needed)
            pass
    
    def close(self):
        """Clean up environment resources."""
        pass
```

# Basic Implementation Examples

## `grid_env/field/implementations/simple_field.py`

```python
import numpy as np
from typing import Dict, Any
from ..abstract_field import AbstractField
from ...utils.types import GridPosition, Displacement, GridConfig

class SimpleField(AbstractField):
    """Simple field implementation with uniform random displacements."""
    
    def __init__(self, config: GridConfig, seed: int = None):
        super().__init__(config, seed)
        self.reset()
    
    def reset(self, seed: int = None) -> None:
        """Reset field to new random configuration."""
        if seed is not None:
            self._rng = np.random.RandomState(seed)
    
    def sample_displacement(self, position: GridPosition) -> Displacement:
        """Sample uniform random displacement."""
        u = self._rng.randint(-self.config.d_max, self.config.d_max + 1)
        v = self._rng.randint(-self.config.d_max, self.config.d_max + 1)
        return Displacement(u, v)
    
    def get_field_state(self) -> Dict[str, Any]:
        """Return field state."""
        return {
            'type': 'simple',
            'config': self.config,
            'seed': self.seed
        }
    
    def get_displacement_pmf(self, position: GridPosition) -> np.ndarray:
        """Return uniform PMF over displacement space."""
        size = 2 * self.config.d_max + 1
        return np.ones((size, size)) / (size ** 2)
```

## `grid_env/actor/grid_actor.py`

```python
import numpy as np
from typing import Dict, Any
from .abstract_actor import AbstractActor
from ..utils.types import GridPosition, VerticalAction, GridConfig

class GridActor(AbstractActor):
    """Basic grid actor with simple vertical dynamics."""
    
    def __init__(self, config: GridConfig, initial_position: GridPosition,
                 noise_prob: float = 0.1):
        """
        Initialize grid actor.
        
        Args:
            config: Grid configuration
            initial_position: Starting position
            noise_prob: Probability of action noise
        """
        super().__init__(config, initial_position)
        self.noise_prob = noise_prob
        self._rng = np.random.RandomState()
    
    def step_vertical(self, action: VerticalAction) -> GridPosition:
        """Execute vertical action with noise."""
        # Determine vertical displacement
        if self._rng.random() < self.noise_prob:
            # Add noise: random displacement in {-1, 0, +1}
            z_displacement = self._rng.choice([-1, 0, 1])
        else:
            # Execute intended action
            z_displacement = action.action
        
        # Apply displacement with boundary enforcement
        new_k = self.enforce_vertical_boundaries(self.position.k + z_displacement)
        
        # Update position
        self.position = GridPosition(self.position.i, self.position.j, new_k)
        return self.position
    
    def get_vertical_displacement_pmf(self, action: VerticalAction) -> np.ndarray:
        """Return PMF over vertical displacements."""
        pmf = np.zeros(3)  # For displacements {-1, 0, +1}
        
        # Intended action probability
        intended_idx = action.action + 1  # Map {-1,0,+1} -> {0,1,2}
        pmf[intended_idx] = 1 - self.noise_prob
        
        # Noise probability distributed uniformly
        noise_prob_each = self.noise_prob / 3
        pmf += noise_prob_each
        
        return pmf
```

# Environment Registration and Usage

## `grid_env/__init__.py`

```python
from gymnasium.envs.registration import register
from .environment import GridEnvironment
from .field.implementations.simple_field import SimpleField
from .actor.grid_actor import GridActor
from .utils.types import GridConfig, GridPosition

# Register environment variants
register(
    id='GridEnv-v0',
    entry_point='grid_env:GridEnvironment',
    max_episode_steps=1000,
)

__all__ = [
    'GridEnvironment',
    'SimpleField', 
    'GridActor',
    'GridConfig',
    'GridPosition'
]
```

## `examples/basic_usage.py`

```python
import gymnasium as gym
import grid_env
from grid_env import SimpleField, GridActor, GridConfig, GridPosition

def main():
    # Create configuration
    config = GridConfig(n_x=5, n_y=5, n_z=3, d_max=1)
    
    # Create field and actor
    field = SimpleField(config, seed=42)
    actor = GridActor(config, GridPosition(3, 3, 2))
    
    # Create environment
    env = grid_env.GridEnvironment(
        field=field,
        actor=actor, 
        config=config,
        max_steps=100
    )
    
    # Run episode
    obs, info = env.reset()
    done = False
    
    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        print(f"Position: {info['position']}, Reward: {reward}")
    
    env.close()

if __name__ == "__main__":
    main()
```

# Key Design Principles

## Modularity
- **Separation of Concerns**: Field, actor, and environment are independent components
- **Interface-Based**: Abstract classes define clear contracts
- **Pluggable**: Easy to swap field/actor implementations

## Extensibility  
- **Field Implementations**: Add new field types (Gaussian, GP-based, etc.)
- **Actor Variants**: Different vertical dynamics, multi-agent support
- **Reward Functions**: Configurable objective functions

## Simplicity
- **Minimal Dependencies**: Only gymnasium, numpy required
- **Clear Hierarchy**: Simple inheritance structure
- **Type Safety**: Strong typing with NamedTuple and type hints

## Gymnasium Compliance
- **Standard Interface**: Full compatibility with gym ecosystem
- **Proper Observation/Action Spaces**: Well-defined spaces for RL algorithms
- **Episode Management**: Correct reset/step/termination handling

This template provides the minimal foundation to implement your grid environment while maintaining flexibility for future extensions toward the full POMDP formulation.