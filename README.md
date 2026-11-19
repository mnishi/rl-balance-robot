# Reinforcement Learning for a 1-DOF Balance Robot

This project investigates reinforcement learning for balancing an intentionally unstable robot in MuJoCo. The current experiment uses a **cylindrical ground-contact element** and **one actuated roll joint**. The aim is to learn dynamic balance by reducing both the gravitational moment about an estimated contact point and the system's angular momentum.

> **Status:** Experimental research project. Consistent long-duration balancing has not yet been demonstrated. The simulation has not been validated against physical hardware.

## Robot and Simulation

The MuJoCo model is defined in `balance.xml` and includes:

- A cylindrical ground-contact element (`ball`) with a free joint, allowing translation and rotation in the simulated world
- A rigid supporting structure (`foot`)
- One actuated hinge joint (`roll`) about the X axis
- An upper structure (`upper_body`) with an intentionally asymmetric mass distribution

**1-DOF refers to the number of controlled joints, not the number of physical degrees of freedom.** Because the contact body has a free joint, the robot can still translate, rotate, or lose contact with the ground.

The physics timestep is **0.002 s (2 ms)**. The environment advances **five physics steps per policy action**, giving a **0.010 s (10 ms) control interval**. An episode is truncated at **2,000 policy steps (20 seconds)** unless it terminates earlier.

## Reinforcement Learning

The environment is implemented with Gymnasium and MuJoCo. The policy is trained with **Proximal Policy Optimization (PPO)** from Stable-Baselines3.

### Observation Space

The policy receives a **3-dimensional** observation vector:

| Quantity | Dimensions | Description |
| --- | ---: | --- |
| COM Y offset | 1 | Combined robot center of mass minus estimated contact point, along world Y |
| COM Y offset velocity | 1 | Finite-difference rate of change of that offset |
| Roll joint angle | 1 | Relative angle of the actuated roll joint, in radians |
| **Total** | **3** | |

The observation does **not** currently include global robot displacement, roll joint velocity, or the base's absolute orientation. These omissions may limit the policy's ability to infer the full physical state.

### Action Space

The policy outputs **one normalized action** in `[-1, 1]`. The environment converts it to a roll-joint **position target**:

```text
roll_target = action[0] * radians(80)
```

The actuator is position-controlled. The MuJoCo model should have compatible joint limits and actuator control limits; consult `balance.xml` for the exact actuator gains and torque limits.

### Reward Function

The reward combines a measure of static gravitational balance with a measure of dynamic motion.

Let `c` be the combined center of mass, `p` the estimated contact point, `m` the combined mass, and `g` the gravity vector. The gravity-induced moment is:

```text
tau = (c - p) × (m * g)
```

Only the **X component**, corresponding to the controlled roll direction, is evaluated:

```text
R_torque = 1 / (1 + abs(tau_x) / 0.01)
```

The total angular momentum `H` about the estimated contact point includes both each body's rotational angular momentum and its orbital contribution. Again, only the X component is evaluated:

```text
R_momentum = 1 / (1 + (abs(H_x) / 0.001)^2)
```

The combined per-step reward is:

```text
R = R_torque * R_momentum
```

The contact point is currently **approximated** as a point one cylinder radius (0.02 m) below the `ball` body origin. This is not a measured MuJoCo contact location; it may become inaccurate when the cylinder tilts, moves, or leaves the ground.

> Minimizing gravitational moment and angular momentum does not guarantee dynamic stability. Contact dynamics, actuator saturation, partial observability, and recovery from perturbations still matter.

### Episode Termination

The current environment terminates an episode when **any** of the following conditions is met:

| Condition | Threshold |
| --- | --- |
| Absolute COM Y offset from the estimated contact point | Greater than **0.05 m** (5 cm) |
| Supporting structure (`foot`) tilt from world vertical | Greater than **80°** |
| Height of the `upper_body` body origin above the world ground plane | **0.05 m** (5 cm) or lower |
| Horizontal displacement of the `ball` body origin from its position at reset | Greater than **0.03 m** (3 cm), measured in world XY |

The last condition is part of the intended latest environment update; verify that it is present in your local `balance_env.py`. It requires storing the initial XY position during `reset()` and comparing it with the current XY position in `step()`.

The height condition uses the **body origin**, not necessarily the upper mass's center or its lowest surface. The movement condition terminates an episode; it does not physically constrain the robot to remain stationary.

An episode is **truncated** when it reaches **2,000 policy steps**.

## Requirements

- Python 3 and a virtual environment
- MuJoCo (`mujoco`)
- Gymnasium (`gymnasium`)
- Stable-Baselines3 (`stable-baselines3`)
- NumPy (`numpy`)
- pandas and Matplotlib (evaluation and plotting)

Example setup on Linux:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install mujoco gymnasium stable-baselines3 numpy pandas matplotlib
```

The interactive MuJoCo viewer requires a working graphics/OpenGL environment.

## Usage

Run commands from the project directory so `balance.xml` can be found.

### Check the environment

```bash
python test_env.py
```

### Train the PPO agent

Train for an additional one million policy steps (the default):

```bash
python train.py
```

Or specify the additional training steps:

```bash
python train.py 500000
```

`train.py` loads `balance_ppo.zip` if it exists, otherwise creates a new PPO model. When loading an existing model, it uses `reset_num_timesteps=False` to preserve the cumulative training-step counter. It saves the updated model to `balance_ppo.zip` after training. PPO's rollout size may cause the actual number of steps to exceed the requested increment slightly.

**Do not reuse an old policy without considering environment changes.** Changing the action scale from ±45° to ±80°, or changing the termination conditions, changes the learning problem even if observation and action dimensions remain the same. Keep earlier model files for comparison when starting a fresh experiment.

### Run staged training

```bash
bash train_loop.sh
```

The updated training-loop workflow is intended to:

1. Read the cumulative `num_timesteps` from an existing `balance_ppo.zip`, or start at zero if no model exists.
2. Train toward the next **1-million-step boundary** below 10 million cumulative steps.
3. Train toward the next **10-million-step boundary** at or above 10 million cumulative steps.
4. Save model backups in `models/` and continue toward a configurable final target (100 million steps in the current script).

Use the **revised** `train_loop.sh` that does not delete the model at startup. An older version of this script explicitly removes `balance_ppo.zip` and the `models/` directory; do not use that version when you want to resume training.

The current `train.py` also has a `CheckpointCallback` configured to save periodic checkpoints in `checkpoints/`. This is separate from the backups created by the training loop.

### Evaluate a trained model

```bash
python evaluate.py
```

Evaluate 100 episodes:

```bash
python evaluate.py --episodes 100
```

Or explicitly name the model:

```bash
python evaluate.py balance_ppo.zip --episodes 100
```

Evaluation uses deterministic policy actions in the current workflow and writes per-step results to `evaluation.csv`. Summary statistics include average, maximum, and minimum survival duration, plus the number of episodes that complete 2,000 steps.

### View the robot in MuJoCo

```bash
python evaluate_viewer.py
```

The viewer displays the trained policy controlling the robot. Inspect whether the cylinder actually contacts the ground and whether the robot translates or tips over; the free joint does not enforce continuous ground contact.

### Generate evaluation plots

```bash
python plot_evaluation.py
```

Plot a particular episode:

```bash
python plot_evaluation.py --episode 3
```

The **1-DOF plotting script** should display available CSV measurements such as roll target versus actual angle, COM Y offset and velocity, roll actuator torque, gravity torque, angular momentum, and reward components. Plots of horizontal travel or upper-body height require those columns to be exported by `evaluate.py`.

## Project Structure

```text
.
├── balance.xml             # MuJoCo 1-DOF robot model
├── balance_env.py          # Gymnasium environment, reward, termination
├── train.py                # PPO training and continuation
├── train_loop.sh           # Staged training toward cumulative targets
├── evaluate.py             # Numerical evaluation and CSV export
├── evaluate_viewer.py      # Interactive MuJoCo visualization
├── plot_evaluation.py      # Evaluation plots
├── test_env.py             # Basic environment test
├── balance_ppo.zip         # Current saved policy, when available
├── evaluation.csv          # Generated evaluation data
├── models/                 # Training-loop backups
├── checkpoints/            # Optional periodic PPO checkpoints
└── logs/                   # TensorBoard logs, when generated
```

Generated models, backups, CSV files, logs, and plots are created as needed and may not exist in a fresh checkout.

## Current Limitations and Future Work

- Improve consistency of long-duration balancing, not just transient survival.
- Investigate policy saturation at extreme roll position targets.
- Check whether adding joint velocity, base orientation, and base angular velocity improves observability.
- Use actual MuJoCo ground contacts rather than a fixed geometric contact-point approximation where appropriate.
- Monitor cylinder lift-off and horizontal translation during evaluation.
- Compare 45° and 80° action ranges using separate training runs.
- Validate the simulation's actuator response, control interval, and sensor assumptions before deploying to hardware.
- Compare PPO control with a conventional controller under the same initial conditions.

## Notes

This repository is intended for experimentation and learning. The model, reward, observations, termination criteria, and training scripts may change between experiments. Interpret results using the exact versions of the environment and checkpoint that produced them.
