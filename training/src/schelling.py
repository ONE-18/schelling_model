import math
import random
from collections import deque


class SchellingModel:
    """Schelling segregation model on an odd-r hexagonal grid with toroidal wrap.

    Cells hold integers: 0 = empty, 1..G = groups
    """

    def __init__(self, num_groups=3, num_neighbors=1, board_size=35,
                 empty_percentage=0.15, tolerance_threshold=0.4,
                 stall_window=500, stall_delta=2):
        self.num_groups = int(num_groups)
        self.num_neighbors = int(num_neighbors)
        self.N = int(board_size)
        self.empty_percentage = float(empty_percentage)
        self.tolerance_threshold = float(tolerance_threshold)
        self.stall_window = int(stall_window)
        self.stall_delta = int(stall_delta)

        # board as list of lists: 0 = empty, 1..G = groups
        self.board = [[0 for _ in range(self.N)] for _ in range(self.N)]
        self.gen = 0
        self.unhappy_history = []
        self.segregation_history = []
        self.satisfaction_history = []

        # neighbor coords cache (computed after board/init)
        self._neighbor_coords = None

        self._init_board()
        # precompute neighbor coordinates for configured radius
        self._compute_neighbor_coords_cache()

    def _init_board(self):
        total = self.N * self.N
        num_empty = int(round(total * self.empty_percentage))
        num_nonempty = total - num_empty
        counts = [num_nonempty // self.num_groups] * self.num_groups
        # distribute remainder
        for i in range(num_nonempty - sum(counts)):
            counts[i % self.num_groups] += 1

        values = []
        for g, c in enumerate(counts, start=1):
            values.extend([g] * c)
        values.extend([0] * num_empty)
        random.shuffle(values)
        # convert flat values list into 2D Python list
        arr = [values[i * self.N:(i + 1) * self.N] for i in range(self.N)]
        self.board = arr
        # If empty percentage was set to 0, force at least one empty cell
        if num_empty == 0:
            ei = random.randrange(self.N)
            ej = random.randrange(self.N)
            self.board[ei][ej] = 0
            # ensure counts/histories consistent
            num_empty = 1
        self.gen = 0
        self.unhappy_history.clear()
        self.segregation_history.clear()
        self.satisfaction_history.clear()

    def converges(self):
        # converged when there are no unhappy agents
        for i in range(self.N):
            for j in range(self.N):
                if self.board[i][j] != 0 and not self.is_happy(i, j):
                    return False
        return True

    def init(self, max_generations=1000):
        for _ in range(max_generations):
            self.gen += 1
            self.step()
            if self.converges():
                return 1
        return 0

    # coordinate helpers (port from the JS odd-r / axial helpers)
    def oddr_to_axial(self, row, col):
        row_parity = abs(row % 2)
        q = col - (row - row_parity) // 2
        r = row
        return q, r

    def axial_to_oddr(self, q, r):
        row_parity = abs(r % 2)
        row = r
        col = q + (r - row_parity) // 2
        return row, col

    def wrap_index(self, v):
        return v % self.N

    def _compute_neighbor_coords_once(self, row, col, radius):
        # compute neighbor coords for a single cell (with wrapping)
        if radius <= 0:
            return []
        origin_q, origin_r = self.oddr_to_axial(row, col)
        coords = []
        seen = set()
        for dq in range(-radius, radius + 1):
            dr_min = max(-radius, -dq - radius)
            dr_max = min(radius, -dq + radius)
            for dr in range(dr_min, dr_max + 1):
                if dq == 0 and dr == 0:
                    continue
                q = origin_q + dq
                r = origin_r + dr
                rr, cc = self.axial_to_oddr(q, r)
                rr = rr % self.N
                cc = cc % self.N
                key = (rr, cc)
                if key in seen:
                    continue
                seen.add(key)
                coords.append(key)
        return coords

    def _compute_neighbor_coords_cache(self):
        # compute and store neighbor coordinate lists for every cell
        self._neighbor_coords = [[[] for _ in range(self.N)] for _ in range(self.N)]
        if self.num_neighbors <= 0:
            return
        for i in range(self.N):
            for j in range(self.N):
                self._neighbor_coords[i][j] = self._compute_neighbor_coords_once(i, j, self.num_neighbors)

    def hex_neighbors_in_radius(self, row, col, radius):
        # use cached neighbor coordinates when possible
        if radius <= 0:
            return []
        if self._neighbor_coords is not None and radius == self.num_neighbors:
            coords = self._neighbor_coords[row][col]
            return [self.board[r][c] for (r, c) in coords]
        # fallback: compute on the fly
        coords = self._compute_neighbor_coords_once(row, col, radius)
        return [self.board[r][c] for (r, c) in coords]

    def neighbors_coords(self, row, col, radius):
        # return list of (r,c) coords for neighbors in radius
        if radius <= 0:
            return []
        # return cached coords when radius matches configured num_neighbors
        if self._neighbor_coords is not None and radius == self.num_neighbors:
            return list(self._neighbor_coords[row][col])
        # otherwise compute on the fly
        return self._compute_neighbor_coords_once(row, col, radius)

    def is_happy(self, row, col):
        v = int(self.board[row][col])
        if v == 0:
            return True
        thresh = float(self.tolerance_threshold)
        nbrs = [int(x) for x in self.hex_neighbors_in_radius(row, col, self.num_neighbors) if x != 0]
        if not nbrs:
            return True
        same = sum(1 for x in nbrs if x == v)
        return (same / len(nbrs)) >= thresh

    def step(self):
        unhappy = []
        empty = []
        for i in range(self.N):
            for j in range(self.N):
                if self.board[i][j] == 0:
                    empty.append((i, j))
                else:
                    if not self.is_happy(i, j):
                        unhappy.append((i, j))

        random.shuffle(unhappy)
        random.shuffle(empty)
        moves = min(len(unhappy), len(empty))
        for k in range(moves):
            ui, uj = unhappy[k]
            ei, ej = empty[k]
            self.board[ei][ej] = self.board[ui][uj]
            self.board[ui][uj] = 0

        self.gen += 1
        unhappy_count = sum(1 for i in range(self.N) for j in range(self.N) if self.board[i][j] != 0 and not self.is_happy(i, j))
        total_nonempty = sum(1 for row in self.board for cell in row if cell != 0)
        unhappy_pct = 0 if total_nonempty == 0 else round(unhappy_count / total_nonempty * 100)

        segregation = self.segregation_index()

        self.unhappy_history.append(unhappy_pct)
        self.segregation_history.append(segregation)
        self.satisfaction_history.append(100 - unhappy_pct)

        return unhappy_count

    def segregation_index(self):
        same = 0
        total = 0
        for i in range(self.N):
            for j in range(self.N):
                v = int(self.board[i][j])
                if v == 0:
                    continue
                # use cached neighbor coords when available
                coords = self.neighbors_coords(i, j, self.num_neighbors)
                nbrs = [int(self.board[r][c]) for (r, c) in coords if self.board[r][c] != 0]
                if not nbrs:
                    continue
                same += sum(1 for x in nbrs if x == v)
                total += len(nbrs)
        return 0 if total == 0 else round(same / total * 100)

    def should_stop_by_stagnation(self):
        n = int(self.stall_window)
        delta = int(self.stall_delta)
        if n <= 0 or delta <= 0:
            return False
        if len(self.satisfaction_history) <= n:
            return False
        current = self.satisfaction_history[-1]
        past = self.satisfaction_history[-1 - n]
        return (current - past) < delta

    def run(self, max_generations=10000, stop_on_empty_unhappy=True):
        for _ in range(max_generations):
            unhappy = self.step()
            if unhappy == 0 and stop_on_empty_unhappy:
                break
            if self.should_stop_by_stagnation():
                break

    def reset(self):
        self._init_board()
        # recompute neighbor cache in case parameters changed
        self._compute_neighbor_coords_cache()
