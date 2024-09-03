__credits__ = ["Carlos Luis"]

from typing import Optional

import gymnasium as gym
from gymnasium.envs.classic_control import utils
from gymnasium.envs.classic_control.pendulum import PendulumEnv as GymPendulumEnv
import numpy as np
import torch

from torchgym import spaces

DEFAULT_X = torch.pi
DEFAULT_Y = 1.0


class PendulumEnv(GymPendulumEnv, gym.Env):
    """
    ## Description

    The inverted pendulum swingup problem is based on the classic problem in control theory.
    The system consists of a pendulum attached at one end to a fixed point, and the other end being free.
    The pendulum starts in a random position and the goal is to apply torque on the free end to swing it
    into an upright position, with its center of gravity right above the fixed point.

    The diagram below specifies the coordinate system used for the implementation of the pendulum's
    dynamic equations.

    ![Pendulum Coordinate System](/_static/diagrams/pendulum.png)

    - `x-y`: cartesian coordinates of the pendulum's end in meters.
    - `theta` : angle in radians.
    - `tau`: torque in `N m`. Defined as positive _counter-clockwise_.

    ## Action Space

    The action is a `torch.tensor` with shape `(1,)` representing the torque applied to free end of the pendulum.

    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | Torque | -2.0 | 2.0 |

    ## Observation Space

    The observation is a `torch.tensor` with shape `(3,)` representing the x-y coordinates of the pendulum's free
    end and its angular velocity.

    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|
    | 0   | x = cos(theta)   | -1.0 | 1.0 |
    | 1   | y = sin(theta)   | -1.0 | 1.0 |
    | 2   | Angular Velocity | -8.0 | 8.0 |

    ## Rewards

    The reward function is defined as:

    *r = -(theta<sup>2</sup> + 0.1 * theta_dt<sup>2</sup> + 0.001 * torque<sup>2</sup>)*

    where `theta` is the pendulum's angle normalized between *[-pi, pi]* (with 0 being in the upright position).
    Based on the above equation, the minimum reward that can be obtained is
    *-(pi<sup>2</sup> + 0.1 * 8<sup>2</sup> + 0.001 * 2<sup>2</sup>) = -16.2736044*,
    while the maximum reward is zero (pendulum is upright with zero velocity and no torque applied).

    ## Starting State

    The starting state is a random angle in *[-pi, pi]* and a random angular velocity in *[-1,1]*.

    ## Episode Truncation

    The episode truncates at 200 time steps.

    ## Arguments

    - `g`: .

    Pendulum has two parameters for `gymnasium.make` with `render_mode` and `g` representing
    the acceleration of gravity measured in *(m s<sup>-2</sup>)* used to calculate the pendulum dynamics.
    The default value is `g = 10.0`.
    On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("Pendulum-v1", render_mode="rgb_array", g=9.81)  # default g=10.0
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<PendulumEnv<Pendulum-v1>>>>>
    >>> env.reset(seed=123, options={"low": -0.7, "high": 0.5})  # default low=-0.6, high=-0.5
    (tensor([ 0.4123625 ,  0.91101986, -0.89235795], dtype=float32), {})

    ```

    ## Version History

    * v1: Simplify the math equations, no difference in behavior.
    * v0: Initial versions release
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, g: float = 10.0, device: str = "cpu"):
        GymPendulumEnv.__init__(self, render_mode, g)
        self.device = device

        high = torch.tensor([1.0, 1.0, self.max_speed], device=self.device, dtype=torch.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high.numpy(), high=high.numpy(), dtype=np.float32)

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = torch.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        newthdot = thdot + (3 * g / (2 * l) * torch.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = torch.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt
        self.state = torch.stack([newth, newthdot])

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self._get_obs(), -costs, False, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        gym.Env.reset(self, seed=seed)
        if options is None:
            high = torch.tensor([DEFAULT_X, DEFAULT_Y])
        else:
            # Note that if you use custom reset bounds, it may lead to out-of-bound
            # state/observations.
            x = options.get("x_init") if "x_init" in options else DEFAULT_X
            y = options.get("y_init") if "y_init" in options else DEFAULT_Y
            x = utils.verify_number_and_cast(x)
            y = utils.verify_number_and_cast(y)
            high = torch.tensor([x, y])
        low = -high  # We enforce symmetric limits.
        self.state = torch.tensor(
            self.np_random.uniform(low=low, high=high),
            dtype=torch.float32,
            device=self.device,
        )
        self.last_u = None

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def _get_obs(self):
        theta, thetadot = self.state
        return torch.stack([torch.cos(theta), torch.sin(theta), thetadot])

def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi
