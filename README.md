# Active InterCONnect (AICON)

AICON is a framework for building robotic systems that estimate states and select actions through dynamic adjustment of component interactions in response to feedback and world regularities. It enables gradient-based optimization of actions while implicitly considering subgoals encoded in environmental regularities, enabling the solution of sequential tasks.

This is the specific implementation for the paper "No Plan but Everything Under Control: Robustly Solving Sequential Tasks with Dynamically Composed Gradient Descent" (Mengers & Brock, ICRA25). For more information, videos, and results, visit the project website: [https://www.tu.berlin/robotics/papers/noplan](https://www.tu.berlin/robotics/papers/noplan)

![Demo GIF](assets/demo.gif)

## Key Concepts

- **Recursive Estimators**: Components that estimate world state quantities and encode regularities in sensory inputs, actions, and time (S × A × t)
  - Each component maintains a differentiable state estimate
  - Supports probabilistic inference and uncertainty estimation
  - Enables robust perception through recursive updates

- **Active Interconnections**: Dynamic connections between components that change with system state
  - Bidirectional information flow enhances robustness
  - Encodes relationships between sensed, estimated, and actuated quantities
  - Enables identification of subgoals during action selection

- **Gradient-Based Action Selection**: Uses steepest gradient descent to pursue goals
  - Automatically identifies and pursues relevant subgoals
  - Adapts to uncertain and dynamic environments
  - Enables natural emergence of interactive perception and error recovery

## Features

- Component-based architecture with three main types:
  - Sensors: For obtaining measurements from the environment
  - Estimators: For recursive state estimation and uncertainty quantification
  - Actions: For executing gradient-based control commands
- Automatic differentiation support for gradient-based optimization
- Two middleware implementations:
  - Python sequential (synchronous)
  - ROS-based (asynchronous, multi-process)
- Example implementations:
  - Drawer opening experiment (ROS) - demonstrates robustness in uncertain environments
  - Blocksworld experiment (Python sequential) - shows solving sequential tasks without planning

## Installation

### Dependencies

- Python 3.12+
- PyTorch 2.4.0+ with CUDA 12.1 support
- ROS (for ROS middleware)
- Additional dependencies listed in `requirements.txt`

### Building

1. Clone the repository:
```bash
git clone https://github.com/yourusername/aicon.git
cd aicon
```

2. Create and activate a conda environment (recommended):
```bash
conda env create -f aicon.yml
conda activate aicon
```

3. Install Python dependencies:
```bash
pip install -e .
```

## Architecture

The framework is built around three main component types:

1. **Recursive Estimation**: Components estimate quantities through differentiable functions
   - State updates based on previous estimates and informative priors
   - Uncertainty quantification for robust information fusion
   - Support for various estimation approaches (Bayesian filtering, moving averages)

2. **Active Interconnections**: Dynamic connections between components
   - Implicit differentiable functions representing relationships
   - Bidirectional information flow for robust estimation
   - State-dependent information exchange

3. **Gradient-Based Action Selection**: Steepest gradient descent for goal pursuit
   - Automatic subgoal identification through regime changes
   - Conflict resolution through gradient magnitude comparison
   - Integrated estimation and action selection

## Examples

### Blocksworld Experiment (Python Sequential)

```python
from aicon.blocksworld_experiment import run_experiment_sync

# Run the experiment
run_experiment_sync()
```

### Drawer Opening Experiment (ROS)

```python
from aicon.drawer_experiment import run_experiment

# Run the experiment
run_experiment()
```

## Documentation

For detailed documentation of the API and components, please refer to the source code documentation. The main classes are:

- `Component`: Base class for all components
- `SensorComponent`: Base class for sensors
- `EstimationComponent`: Base class for state estimators
- `ActionComponent`: Base class for gradient-based actions
- `ActiveInterconnection`: Handles active interconnections between components

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{mengers2025noplan,
	author={Mengers, Vito and Brock, Oliver},
	booktitle={IEEE International Conference on Robotics and Automation (ICRA)}, 
	title={No Plan but Everything Under Control: Robustly Solving Sequential Tasks with Dynamically Composed Gradient Descent}, 
	year={2025},
	pages={XX--XX}
}
```