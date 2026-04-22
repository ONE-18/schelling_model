import random
import numpy as np
import math
import os
import concurrent.futures
import numba as nb

class SchellingModelNumPy:
    """NumPy-accelerated Schelling model (opt-in alternative).

    API is intentionally similar to `src.schelling.SchellingModel` but implemented
    over NumPy arrays for hot loops.
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

        self.gen = 0
        self.unhappy_history = []
        self.segregation_history = []
        self.satisfaction_history = []

        self._neighbor_coords = None
        self._neighbor_index = None

        self._init_board()
        self._compute_neighbor_coords_cache()

    def _init_board(self):
        total = self.N * self.N
        num_empty = int(round(total * self.empty_percentage))
        num_nonempty = total - num_empty
        counts = [num_nonempty // self.num_groups] * self.num_groups
        for i in range(num_nonempty - sum(counts)):
            counts[i % self.num_groups] += 1

        values = []
        for g, c in enumerate(counts, start=1):
            values.extend([g] * c)
        values.extend([0] * num_empty)
        random.shuffle(values)
        arr = np.array(values, dtype=np.int64).reshape((self.N, self.N))
        # ensure at least one empty
        if num_empty == 0:
            ei = random.randrange(self.N)
            ej = random.randrange(self.N)
            arr[ei, ej] = 0
        self.board = arr
        self.gen = 0
        self.unhappy_history.clear()
        self.segregation_history.clear()
        self.satisfaction_history.clear()

    # coordinate helpers (copy of odd-r axial helpers)
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

    def _compute_neighbor_coords_once(self, row, col, radius):
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
        self._neighbor_coords = [[[] for _ in range(self.N)] for _ in range(self.N)]
        if self.num_neighbors <= 0:
            return
        for i in range(self.N):
            for j in range(self.N):
                self._neighbor_coords[i][j] = self._compute_neighbor_coords_once(i, j, self.num_neighbors)

        # Flatten to neighbor-index array
        total = self.N * self.N
        max_nei = 0
        for i in range(self.N):
            for j in range(self.N):
                max_nei = max(max_nei, len(self._neighbor_coords[i][j]))
        if max_nei == 0:
            self._neighbor_index = None
            return
        arr = np.full((total, max_nei), -1, dtype=np.int64)
        for i in range(self.N):
            for j in range(self.N):
                idx = i * self.N + j
                neigh = self._neighbor_coords[i][j]
                if not neigh:
                    continue
                flat = [r * self.N + c for (r, c) in neigh]
                arr[idx, : len(flat)] = flat
        self._neighbor_index = arr

    def hex_neighbors_in_radius(self, row, col, radius):
        if radius <= 0:
            return []
        if self._neighbor_coords is not None and radius == self.num_neighbors:
            coords = self._neighbor_coords[row][col]
            return [int(self.board[r, c]) for (r, c) in coords]
        coords = self._compute_neighbor_coords_once(row, col, radius)
        return [int(self.board[r, c]) for (r, c) in coords]

    def is_happy(self, row, col):
        v = int(self.board[row, col])
        if v == 0:
            return True
        nbrs = [int(x) for x in self.hex_neighbors_in_radius(row, col, self.num_neighbors) if x != 0]
        if not nbrs:
            return True
        same = sum(1 for x in nbrs if x == v)
        return (same / len(nbrs)) >= float(self.tolerance_threshold)

    def step(self, fast=False):
        """Vectorized step using NumPy internally but deterministic shuffling via
        Python's `random.shuffle` to remain reproducible with `random.seed()`.
        """
        if self._neighbor_index is None:
            # fallback to cheap scan
            unhappy = []
            empty = []
            for i in range(self.N):
                for j in range(self.N):
                    v = int(self.board[i, j])
                    if v == 0:
                        empty.append((i, j))
                    else:
                        if not self.is_happy(i, j):
                            unhappy.append((i, j))
        else:
            board_flat = self.board.ravel()
            nei = self._neighbor_index
            # if numba is available, use JITed computation for same_counts/totals
            if nb is not None:
                same_counts, totals = _numba_same_totals(nei, board_flat)
            else:
                nei_safe = np.where(nei == -1, 0, nei)
                nbr_vals = board_flat[nei_safe]
                valid_mask = nei != -1
                mask_nonzero = valid_mask & (nbr_vals != 0)
                same_counts = np.sum((nbr_vals == board_flat[:, None]) & mask_nonzero, axis=1)
                totals = np.sum(mask_nonzero, axis=1)
            frac = np.zeros_like(same_counts, dtype=float)
            nonzero = totals > 0
            frac[nonzero] = same_counts[nonzero] / totals[nonzero]
            is_happy_mask = (board_flat == 0) | (frac >= float(self.tolerance_threshold))

            unhappy_idx = np.nonzero((~is_happy_mask) & (board_flat != 0))[0]
            empty_idx = np.nonzero(board_flat == 0)[0]

            # convert flat indices to (row, col) tuples and deterministic shuffle
            N = self.N
            unhappy = [(int(x) // N, int(x) % N) for x in unhappy_idx.tolist()]
            empty = [(int(x) // N, int(x) % N) for x in empty_idx.tolist()]
            random.shuffle(unhappy)
            random.shuffle(empty)

        # perform moves
        moves = min(len(unhappy), len(empty))
        for k in range(moves):
            ui, uj = unhappy[k]
            ei, ej = empty[k]
            self.board[ei, ej] = self.board[ui, uj]
            self.board[ui, uj] = 0

        self.gen += 1

        # recompute unhappy_count
        if self._neighbor_index is None:
            unhappy_count = 0
            for i in range(self.N):
                for j in range(self.N):
                    v = int(self.board[i, j])
                    if v != 0 and not self.is_happy(i, j):
                        unhappy_count += 1
        else:
            board_flat = self.board.ravel()
            nei = self._neighbor_index
            nei_safe = np.where(nei == -1, 0, nei)
            nbr_vals = board_flat[nei_safe]
            valid_mask = nei != -1
            mask_nonzero = valid_mask & (nbr_vals != 0)
            same_counts = np.sum((nbr_vals == board_flat[:, None]) & mask_nonzero, axis=1)
            totals = np.sum(mask_nonzero, axis=1)
            frac = np.zeros_like(same_counts, dtype=float)
            nonzero = totals > 0
            frac[nonzero] = same_counts[nonzero] / totals[nonzero]
            is_happy_mask = (board_flat == 0) | (frac >= float(self.tolerance_threshold))
            unhappy_count = int(np.sum((board_flat != 0) & (~is_happy_mask)))

        total_nonempty = int(np.sum(self.board != 0))
        unhappy_pct = 0 if total_nonempty == 0 else int(round(unhappy_count / total_nonempty * 100))

        if not fast:
            # segregation index vectorized
            occupied = (self.board.ravel() != 0)
            if np.any(occupied):
                board_flat = self.board.ravel()
                nei = self._neighbor_index
                nei_safe = np.where(nei == -1, 0, nei)
                nbr_vals = board_flat[nei_safe]
                valid_mask = nei != -1
                mask_nonzero = valid_mask & (nbr_vals != 0)
                same_counts = np.sum((nbr_vals == board_flat[:, None]) & mask_nonzero, axis=1)
                totals = np.sum(mask_nonzero, axis=1)
                same_total = int(np.sum(same_counts[occupied & (totals > 0)]))
                total_total = int(np.sum(totals[occupied & (totals > 0)]))
                segregation = 0 if total_total == 0 else int(round(same_total / total_total * 100))
            else:
                segregation = 0
            self.unhappy_history.append(unhappy_pct)
            self.segregation_history.append(segregation)
            self.satisfaction_history.append(100 - unhappy_pct)

        return unhappy_count

    def segregation_index(self):
        same = 0
        total = 0
        for i in range(self.N):
            for j in range(self.N):
                v = int(self.board[i, j])
                if v == 0:
                    continue
                coords = self._neighbor_coords[i][j]
                nbrs = [int(self.board[r, c]) for (r, c) in coords if self.board[r, c] != 0]
                if not nbrs:
                    continue
                same += sum(1 for x in nbrs if x == v)
                total += len(nbrs)
        return 0 if total == 0 else int(round(same / total * 100))

    def converges(self):
        for i in range(self.N):
            for j in range(self.N):
                if int(self.board[i, j]) != 0 and not self.is_happy(i, j):
                    return False
        return True

    def init(self, max_generations=1000):
        for _ in range(max_generations):
            self.gen += 1
            self.step()
            if self.converges():
                return 1
        return 0

    def run(self, max_generations=10000, stop_on_empty_unhappy=True, fast=False):
        for _ in range(max_generations):
            unhappy = self.step(fast=fast)
            if unhappy == 0 and stop_on_empty_unhappy:
                break
            if (not fast) and len(self.satisfaction_history) > 0 and self.should_stop_by_stagnation():
                break

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

    def reset(self):
        self._init_board()
        self._compute_neighbor_coords_cache()

    def run_simulations(self, runs=10, max_generations=1000, parallel=False, workers=None, base_seed=None, fast=False):
        """Run `runs` independent simulations starting from the current board snapshot.

        Returns the fraction of simulations that converged (float between 0 and 1).
        """
        if runs <= 0:
            return 0.0
        snapshot = self.board.copy() if isinstance(self.board, np.ndarray) else [row.copy() for row in self.board]
        params = dict(
            num_groups=self.num_groups,
            num_neighbors=self.num_neighbors,
            board_size=self.N,
            empty_percentage=self.empty_percentage,
            tolerance_threshold=self.tolerance_threshold,
            stall_window=self.stall_window,
            stall_delta=self.stall_delta,
        )
        if base_seed is None:
            base_seed = random.randrange(2 ** 30)

        results = []
        if parallel and runs > 1:
            max_workers = workers or min(runs, (os.cpu_count() or 1))
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_worker_run_schelling_numpy, snapshot, params, base_seed + i, max_generations, fast) for i in range(runs)]
                for f in concurrent.futures.as_completed(futures):
                    results.append(int(f.result()))
        else:
            state = random.getstate()
            try:
                for i in range(runs):
                    random.seed(base_seed + i)
                    m = SchellingModelNumPy(**params)
                    m.board = snapshot.copy() if isinstance(snapshot, np.ndarray) else [row.copy() for row in snapshot]
                    m.gen = 0
                    m.unhappy_history.clear()
                    m.segregation_history.clear()
                    m.satisfaction_history.clear()
                    m._compute_neighbor_coords_cache()
                    results.append(int(m.init(max_generations=max_generations)))
            finally:
                random.setstate(state)

        converged = sum(results)
        return float(converged) / float(runs)


def _worker_run_schelling_numpy(snapshot, params, seed, max_generations, fast):
    import random
    from training.src.schelling_numpy import SchellingModelNumPy
    import time
    m = SchellingModelNumPy(**params)
    m.board = snapshot.copy() if isinstance(snapshot, np.ndarray) else [row.copy() for row in snapshot]
    m.gen = 0
    m.unhappy_history.clear()
    m.segregation_history.clear()
    m.satisfaction_history.clear()
    m._compute_neighbor_coords_cache()
    random.seed(seed)
    t0 = time.perf_counter()
    res = int(m.init(max_generations=max_generations))
    t1 = time.perf_counter()
    return (res, t1 - t0)


if nb is not None:
    # Numba-jitted helper: compute same_counts and totals per flat index
    @nb.njit(parallel=True)
    def _numba_same_totals(nei, board_flat):
        total_cells, max_nei = nei.shape
        same = np.zeros(total_cells, dtype=np.int64)
        totals = np.zeros(total_cells, dtype=np.int64)
        for i in nb.prange(total_cells):
            bi = board_flat[i]
            s = 0
            t = 0
            for j in range(max_nei):
                idx = nei[i, j]
                if idx == -1:
                    continue
                val = board_flat[idx]
                if val != 0:
                    t += 1
                    if val == bi:
                        s += 1
            same[i] = s
            totals[i] = t
        return same, totals
else:
    _numba_same_totals = None

