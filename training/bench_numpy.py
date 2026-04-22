import time
import random
from src.schelling import SchellingModel
from src.schelling_numpy import SchellingModelNumPy

random.seed(0)

N = 80
steps = 50

print('Building pure-Python model...')
py = SchellingModel(num_groups=3, num_neighbors=2, board_size=N, empty_percentage=0.15, tolerance_threshold=0.4)
np_model = SchellingModelNumPy(num_groups=3, num_neighbors=2, board_size=N, empty_percentage=0.15, tolerance_threshold=0.4)

print('Warming up...')
# warmup
py.step(fast=True)
np_model.step(fast=True)

print('Timing pure-Python')
start = time.perf_counter()
for _ in range(steps):
    py.step(fast=True)
end = time.perf_counter()
print('Pure-Python steps:', steps, 'time:', end - start)

print('Timing NumPy')
start = time.perf_counter()
for _ in range(steps):
    np_model.step(fast=True)
end = time.perf_counter()
print('NumPy steps:', steps, 'time:', end - start)
