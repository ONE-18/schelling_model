# Schelling Model

This project implements the Schelling model of segregation, which demonstrates how individual preferences can lead to large-scale segregation in a population. The model is based on the work of economist Thomas Schelling and uses a grid-based approach to simulate the movement of individuals based on their neighbors' characteristics.

## Overview

The Schelling model is a simple agent-based model that illustrates how individuals with a preference for neighbors of the same type can lead to segregation, even if they are not strongly biased. The model allows for the simulation of different scenarios based on parameters such as the number of groups, the number of neighbors considered, the size of the board, the percentage of empty cells, and the tolerance threshold for individual satisfaction.

## Installation

To install the necessary dependencies, run:

```
pip install -r requirements.txt
```

## Usage

To use the `SchellingModel`, you can instantiate it with the desired parameters:

```python
from schelling import SchellingModel

model = SchellingModel(num_groups=3, num_neighbors=4, board_size=50, empty_percentage=0.2, tolerance_threshold=0.3)
convergence = model.init(max_generations=100)
```

The `init(max_generations)` method will run the model for a specified number of generations and return:
- `1` if the model converges (i.e., no individual is unhappy),
- `0` if it does not converge within the given generations.

## Running Tests

To run the unit tests for the `SchellingModel`, navigate to the `tests` directory and execute:

```
pytest test_schelling.py
```

This will run all the tests defined in the `test_schelling.py` file to ensure that the model behaves as expected.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.