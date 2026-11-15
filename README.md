# Reinforcement Learning for a Single-Contact Balance Robot

This project explores reinforcement learning for balancing an intentionally unstable robot in MuJoCo. The robot rests on a spherical contact point and uses two actuated joints (pitch and roll) to reposition its upper structure and maintain balance.

The objective is to investigate whether a PPO policy can learn dynamic balance by minimizing the gravitational moment about the ground-contact point, rather than relying solely on a conventional upright-angle reward.

> **Status:** Experimental research project. Stable long-duration balancing has not yet been demonstrated consistently, and the simulation is not a validated representation of physical hardware.

## Robot and Simulation

The MuJoCo model is defined in `balance.xml` and includes:

- A spherical ground-contact element (`ball`)
- A rigid supporting structure (`foot`)
- Two serially arranged, actuated joints (`pitch` and `roll`)
- An upper structure (`upper_body`) with an intentionally asymmetric mass distribution

The asymmetric design creates a nontrivial equilibrium posture and makes the balancing problem more representative of imperfect physical construction.

The current simulation configuration uses a MuJoCo physics timestep of **2 ms** and executes **five physics steps per environment action**, giving a **10 ms control interval**. Episodes are limited to **2,000 control steps (20 seconds)** unless terminated earlier.

## Reinforcement Learning

The environment is implemented with Gymnasium and MuJoCo, and the agent is trained using **Proximal Policy Optimization (PPO)** from Stable-Baselines3.

### Observation Space

The policy receives a 13-dimensional observation vector:

| Quantity | Dimensions |
| --- | ---: |
| Upper-body local Z axis expressed in world coordinates | 3 |
| Upper-body angular velocity | 3 |
| Pitch and roll joint angles | 2 |
| Pitch and roll joint velocities | 2 |
| Ball angular velocity | 3 |
| **Total** | **13** |

### Action Space

The policy outputs two normalized actions in the range `[-1, 1]`, corresponding to pitch and roll joint position targets. Each action is scaled to a target angle within **±30°**.

### Reward Function

The main reward encourages the robot to reduce the gravitational torque about an estimated ground-contact point.

Let:

- **c** be the combined center of mass of the robot,
- **p** be the estimated contact point,
- **m** be the total mass, and
- **g** be the gravity vector.

The gravitational torque is calculated as:

```text
tau = (c - p) × (m * g)
```

The torque-based reward is:

```text
R_torque = 1 / (1 + ||tau||² / 0.01²)
```

A small penalty discourages unnecessary joint movement:

```text
R = R_torque - 0.01 * mean_joint_angle_change
```

Here, `mean_joint_angle_change` is the running mean of the sum of absolute pitch and roll angle changes, using a window of up to 2,000 environment steps.

**Important:** Minimizing gravitational torque is not equivalent to guaranteeing dynamic stability. Angular momentum, contact dynamics, actuator limits, and delayed responses can still cause the robot to fall.

### Episode Termination

An episode terminates when the supporting structure tilts beyond the configured threshold (the world-frame Z component of the foot's local Z axis falls below approximately `0.707`). An episode is truncated after 2,000 control steps.

## Requirements

- Python 3 and a virtual environment
- MuJoCo Python package
- Gymnasium
- Stable-Baselines3
- NumPy
- pandas and Matplotlib (for evaluation plots)

Install dependencies in a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install mujoco gymnasium stable-baselines3 numpy pandas matplotlib
```

MuJoCo's interactive viewer also requires a working graphical/OpenGL environment.

## Usage

Run commands from the project directory so that `balance.xml` can be found.

### Check the environment

```bash
python test_env.py
```

To inspect the simulated response to a fixed motor command without PPO:

```bash
python test_zero_action.py
```

Despite its filename, `test_zero_action.py` currently applies a small **nonzero** fixed action.

### Train the PPO agent

```bash
python train.py
```

The training script manages model creation or continuation according to its current configuration. Check its settings before starting a long run. The model checkpoint used by the evaluation scripts is named `balance_ppo.zip` by default.

### Evaluate a trained model

```bash
python evaluate.py
```

This runs evaluation episodes and writes measurements to `evaluation.csv` for subsequent analysis.

### View the robot in MuJoCo

```bash
python evaluate_viewer.py
```

The viewer displays the policy controlling the robot. In the current workflow, the viewer can remain open after an episode ends and be closed manually to avoid a shutdown-related segmentation fault observed with immediate automatic teardown on one setup.

### Generate evaluation plots

```bash
python plot_evaluation.py
```

To plot a specific episode:

```bash
python plot_evaluation.py --episode 3
```

The plotting script reads `evaluation.csv` and saves PNG graphs of joint targets versus actual angles, gravitational torque, reward components, and joint movement. Its default episode selection is episode 10.

## Project Structure

```text
.
├── balance.xml              # MuJoCo robot model
├── balance_env.py           # Gymnasium environment and reward function
├── train.py                 # PPO training
├── evaluate.py              # Numerical policy evaluation and CSV export
├── evaluate_viewer.py       # Interactive MuJoCo visualization
├── plot_evaluation.py       # Evaluation plots
├── test_env.py              # Basic environment test
├── test_zero_action.py      # Fixed-action dynamics test
├── balance_ppo.zip          # Trained policy checkpoint (when available)
├── evaluation.csv           # Generated evaluation data
└── archive/                 # Optional location for past experiments
```

Generated checkpoints, CSV files, and PNG plots may not be present until the corresponding scripts have been run.

## Current Limitations and Future Work

- Improve the consistency and duration of balance without overfitting to short episodes.
- Study the relationship between static gravitational balance and dynamic stability.
- Evaluate policy robustness to variations in initial conditions, mass distribution, and external disturbances.
- Check whether the simulation's observations, control rates, and actuator behavior can be reproduced on physical hardware.
- Compare learned control with conventional control approaches when a suitable baseline is available.

## Notes

This repository is intended for experimentation and learning. The reward function, physical model, and training configuration may change as experiments progress. Results should be interpreted in the context of the exact model and checkpoint used.
