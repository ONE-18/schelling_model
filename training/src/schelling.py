import math
import random
from collections import deque
import os
import concurrent.futures


def _worker_run_schelling(snapshot, params, seed, max_generations, fast):
    """Worker function for running a single simulation in a separate process.

    Returns 1 if converged, 0 otherwise.
    """
    import random
    # create a fresh model and overwrite its board with the provided snapshot
    m = SchellingModel(**params)
    # snapshot is list of lists
    m.board = [row.copy() for row in snapshot]
    m.gen = 0
    m.unhappy_history.clear()
    m.segregation_history.clear()
    m.satisfaction_history.clear()
    m._compute_neighbor_coords_cache()
    random.seed(seed)
    import time
    t0 = time.perf_counter()
    res = int(m.init(max_generations=max_generations))
    t1 = time.perf_counter()
    return (res, t1 - t0)


class SchellingModel:
    """Schelling segregation model on an odd-r hexagonal grid with toroidal wrap.

    Cells hold integers: 0 = empty, 1..G = groups
    """

    def __init__(self, num_groups=3, num_neighbors=1, board_size=35,
                 empty_percentage=0.15, tolerance_threshold=0.4,
                 stall_window=500, stall_delta=2, use_numpy=False):
        """Inicializa el modelo con los parámetros dados y prepara el tablero.

        Parámetros:
        - `num_groups`: número de grupos de agentes.
        - `num_neighbors`: radio de vecinos hexagonales.
        - `board_size`: tamaño N del tablero (N x N).
        - `empty_percentage`: fracción de celdas vacías.
        - `tolerance_threshold`: umbral de tolerancia para estar satisfecho.
        - `stall_window`, `stall_delta`: parámetros para detección de estancamiento.
        """
        self.num_groups = int(num_groups)
        self.num_neighbors = int(num_neighbors)
        self.N = int(board_size)
        self.empty_percentage = float(empty_percentage)
        self.tolerance_threshold = float(tolerance_threshold)
        self.stall_window = int(stall_window)
        self.stall_delta = int(stall_delta)
        # control whether to use NumPy-accelerated path (default off)
        self.use_numpy = bool(use_numpy)

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
        """Genera y baraja la distribución inicial de agentes y celdas vacías.

        Construye `self.board` como lista 2D y reinicia contadores/historiales.
        """
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
        """Devuelve True si no hay agentes insatisfechos en el tablero.

        Recorre el tablero y comprueba `is_happy` para cada agente no vacío.
        """
        for i in range(self.N):
            for j in range(self.N):
                if self.board[i][j] != 0 and not self.is_happy(i, j):
                    return False
        return True

    def init(self, max_generations=1000):
        """Ejecuta el modelo hasta `max_generations` o hasta convergencia.

        Incrementa `self.gen` y llama a `step()` repetidamente. Devuelve 1 si
        converge antes de agotar las generaciones, 0 en caso contrario.
        """
        for _ in range(max_generations):
            self.gen += 1
            self.step()
            if self.converges():
                return 1
        return 0

    # coordinate helpers (port from the JS odd-r / axial helpers)
    def oddr_to_axial(self, row, col):
        """Convierte coordenadas odd-r (fila, columna) a coordenadas axiales (q, r).

        Útil para calcular vecinos en rejilla hexagonal con offset impar.
        """
        row_parity = abs(row % 2)
        q = col - (row - row_parity) // 2
        r = row
        return q, r

    def axial_to_oddr(self, q, r):
        """Convierte coordenadas axiales (q, r) a odd-r (fila, columna)."""
        row_parity = abs(r % 2)
        row = r
        col = q + (r - row_parity) // 2
        return row, col

    def wrap_index(self, v):
        """Aplica envoltura toroidal al índice `v` basándose en `self.N`.

        Devuelve el índice equivalente en el rango [0, N).
        """
        return v % self.N

    def _compute_neighbor_coords_once(self, row, col, radius):
        """Calcula las coordenadas (fila,col) de todos los vecinos dentro de
        `radius` para una celda dada, aplicando envoltura toroidal.

        Evita duplicados usando `seen` y excluye la propia celda.
        """
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
        """Precomputa y almacena la lista de coordenadas de vecinos para cada celda.

        Esta cache acelera consultas repetidas de vecinos cuando `num_neighbors`
        es fija.
        """
        self._neighbor_coords = [[[] for _ in range(self.N)] for _ in range(self.N)]
        if self.num_neighbors <= 0:
            return
        for i in range(self.N):
            for j in range(self.N):
                self._neighbor_coords[i][j] = self._compute_neighbor_coords_once(i, j, self.num_neighbors)
    def hex_neighbors_in_radius(self, row, col, radius):
        """Devuelve los valores de las celdas vecinas dentro del `radius`.

        Usa la cache cuando `radius` coincide con `num_neighbors`, sino calcula
        las coordenadas al vuelo.
        """
        if radius <= 0:
            return []
        if self._neighbor_coords is not None and radius == self.num_neighbors:
            coords = self._neighbor_coords[row][col]
            return [self.board[r][c] for (r, c) in coords]
        coords = self._compute_neighbor_coords_once(row, col, radius)
        return [self.board[r][c] for (r, c) in coords]

    def neighbors_coords(self, row, col, radius):
        """Devuelve una lista de coordenadas (fila,col) de vecinos dentro del radio.

        Retorna la cache cuando es aplicable, o calcula las coordenadas.
        """
        if radius <= 0:
            return []
        if self._neighbor_coords is not None and radius == self.num_neighbors:
            return list(self._neighbor_coords[row][col])
        return self._compute_neighbor_coords_once(row, col, radius)

    def _scan_chunk(self, row_start, row_end):
        """Escanea un rango de filas y devuelve listas de celdas insatisfechas y vacías.

        Diseñado para usarse por hilos/trabajadores en paralelización.
        """
        unhappy = []
        empty = []
        for i in range(row_start, row_end):
            for j in range(self.N):
                if self.board[i][j] == 0:
                    empty.append((i, j))
                else:
                    if not self.is_happy(i, j):
                        unhappy.append((i, j))
        return unhappy, empty

    def _find_unhappy_and_empty_parallel(self):
        """Escanea en paralelo el tablero y concatena listas de celdas insatisfechas y vacías.

        Divide las filas en trozos, lanza tareas con `ThreadPoolExecutor` y
        combina resultados ordenándolos en orden fila-columna.
        """
        workers = min(32, (os.cpu_count() or 1), self.N)
        if workers <= 1:
            return self._scan_chunk(0, self.N)

        chunk = int(math.ceil(self.N / workers))
        ranges = []
        for w in range(workers):
            start = w * chunk
            end = min((w + 1) * chunk, self.N)
            if start < end:
                ranges.append((start, end))

        unhappy = []
        empty = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(self._scan_chunk, s, e) for (s, e) in ranges]
            for f in concurrent.futures.as_completed(futures):
                u, e = f.result()
                unhappy.extend(u)
                empty.extend(e)
        # ensure deterministic row-major ordering before any random shuffle
        unhappy.sort(key=lambda x: (x[0], x[1]))
        empty.sort(key=lambda x: (x[0], x[1]))
        return unhappy, empty

    def is_happy(self, row, col):
        """Determina si el agente en (row,col) está satisfecho según el umbral.

        Retorna True para celdas vacías o cuando la fracción de vecinos del
        mismo grupo >= `tolerance_threshold`.
        """
        v = int(self.board[row][col])
        if v == 0:
            return True
        thresh = float(self.tolerance_threshold)
        nbrs = [int(x) for x in self.hex_neighbors_in_radius(row, col, self.num_neighbors) if x != 0]
        if not nbrs:
            return True
        same = sum(1 for x in nbrs if x == v)
        return (same / len(nbrs)) >= thresh

    def step(self, fast=False):
        """Ejecuta una generación del modelo.

        - Encuentra celdas insatisfechas y vacías (paralelizado).
        - Reubica aleatoriamente agentes insatisfechos en celdas vacías.
        - Actualiza contadores y, salvo `fast=True`, actualiza historiales y
          calcula el índice de segregación.
        Devuelve el número de agentes insatisfechos tras el paso.
        """
        unhappy, empty = self._find_unhappy_and_empty_parallel()

        random.shuffle(unhappy)
        random.shuffle(empty)
        moves = min(len(unhappy), len(empty))
        for k in range(moves):
            ui, uj = unhappy[k]
            ei, ej = empty[k]
            self.board[ei][ej] = self.board[ui][uj]
            self.board[ui][uj] = 0

        self.gen += 1
        unhappy_count = sum(1 for i in range(self.N) for j in range(self.N)
                           if self.board[i][j] != 0 and not self.is_happy(i, j))
        total_nonempty = sum(1 for row in self.board for cell in row if cell != 0)
        unhappy_pct = 0 if total_nonempty == 0 else round(unhappy_count / total_nonempty * 100)

        if not fast:
            segregation = self.segregation_index()
            self.unhappy_history.append(unhappy_pct)
            self.segregation_history.append(segregation)
            self.satisfaction_history.append(100 - unhappy_pct)

        return unhappy_count

    def is_happy(self, row, col):
        """Determina si el agente en (row,col) está satisfecho según el umbral.

        Retorna True para celdas vacías o cuando la fracción de vecinos del
        mismo grupo >= `tolerance_threshold`.
        """
        v = int(self.board[row][col])
        if v == 0:
            return True
        thresh = float(self.tolerance_threshold)
        nbrs = [int(x) for x in self.hex_neighbors_in_radius(row, col, self.num_neighbors) if x != 0]
        if not nbrs:
            return True
        same = sum(1 for x in nbrs if x == v)
        return (same / len(nbrs)) >= thresh

    def step(self, fast=False):
        """Ejecuta una generación del modelo.

        - Encuentra celdas insatisfechas y vacías (paralelizado).
        - Reubica aleatoriamente agentes insatisfechos en celdas vacías.
        - Actualiza contadores y, salvo `fast=True`, actualiza historiales y
          calcula el índice de segregación.
        Devuelve el número de agentes insatisfechos tras el paso.
        """
        # If NumPy acceleration is available and neighbor index built, use it
        if self.use_numpy:
            import numpy as np
        if self.use_numpy and np is not None and self._neighbor_index is not None:
            return self._step_numpy(fast=fast)

        unhappy, empty = self._find_unhappy_and_empty_parallel()

        random.shuffle(unhappy)
        random.shuffle(empty)
        moves = min(len(unhappy), len(empty))
        for k in range(moves):
            ui, uj = unhappy[k]
            ei, ej = empty[k]
            self.board[ei][ej] = self.board[ui][uj]
            self.board[ui][uj] = 0

        self.gen += 1
        unhappy_count = sum(1 for i in range(self.N) for j in range(self.N)
                           if self.board[i][j] != 0 and not self.is_happy(i, j))
        total_nonempty = sum(1 for row in self.board for cell in row if cell != 0)
        unhappy_pct = 0 if total_nonempty == 0 else round(unhappy_count / total_nonempty * 100)

        if not fast:
            segregation = self.segregation_index()
            self.unhappy_history.append(unhappy_pct)
            self.segregation_history.append(segregation)
            self.satisfaction_history.append(100 - unhappy_pct)

        return unhappy_count

    def segregation_index(self):
        """Calcula un índice simple de segregación.

        Para cada agente suma cuántos vecinos son del mismo grupo y normaliza
        por el total de vecinos considerandos, devolviendo un porcentaje.
        """
        same = 0
        total = 0
        for i in range(self.N):
            for j in range(self.N):
                v = int(self.board[i][j])
                if v == 0:
                    continue
                coords = self.neighbors_coords(i, j, self.num_neighbors)
                nbrs = [int(self.board[r][c]) for (r, c) in coords if self.board[r][c] != 0]
                if not nbrs:
                    continue
                same += sum(1 for x in nbrs if x == v)
                total += len(nbrs)
        return 0 if total == 0 else round(same / total * 100)

    def should_stop_by_stagnation(self):
        """Devuelve True si la satisfacción no ha mejorado suficientemente en la ventana.

        Compara la satisfacción actual con la de `n` generaciones atrás y
        devuelve True si la diferencia es menor que `stall_delta`.
        """
        n = int(self.stall_window)
        delta = int(self.stall_delta)
        if n <= 0 or delta <= 0:
            return False
        if len(self.satisfaction_history) <= n:
            return False
        current = self.satisfaction_history[-1]
        past = self.satisfaction_history[-1 - n]
        return (current - past) < delta

    def run(self, max_generations=10000, stop_on_empty_unhappy=True, fast=False):
        """Ejecuta el modelo durante hasta `max_generations` generaciones.

        Si `fast=True` evita comprobaciones y cálculos adicionales para maximizar
        el rendimiento. Puede parar si no hay agentes insatisfechos.
        """
        for _ in range(max_generations):
            unhappy = self.step(fast=fast)
            if unhappy == 0 and stop_on_empty_unhappy:
                break
            if (not fast) and self.should_stop_by_stagnation():
                break

    def run_simulations(self, runs=10, max_generations=1000, parallel=False, workers=None, base_seed=None, fast=False, mode='process', verbose=True):
        """Run `runs` independent simulations starting from the current board snapshot.

        Returns a dict with statistics: {
            'runs', 'converged', 'fraction', 'total_time', 'mean_time', 'std_time', 'times'
        }

        - `parallel`: if True uses pool execution according to `mode` ('process' or 'thread').
        - `mode`: 'process' or 'thread' (ignored when parallel=False).
        - `workers`: number of workers for parallel execution.
        - `base_seed`: optional base seed; if None a random base seed is chosen.
        - `verbose`: if True prints a short summary to stdout.
        """
        import time
        import statistics

        if runs <= 0:
            return {'runs': 0, 'converged': 0, 'fraction': 0.0, 'total_time': 0.0, 'mean_time': 0.0, 'std_time': 0.0, 'times': []}

        snapshot = [row.copy() for row in self.board]
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

        times = []
        results = []
        t0 = time.perf_counter()

        if parallel and runs > 1:
            max_workers = workers or min(runs, (os.cpu_count() or 1))
            Executor = concurrent.futures.ProcessPoolExecutor if mode == 'process' else concurrent.futures.ThreadPoolExecutor
            with Executor(max_workers=max_workers) as ex:
                futures = [ex.submit(_worker_run_schelling, snapshot, params, base_seed + i, max_generations, fast) for i in range(runs)]
                for f in concurrent.futures.as_completed(futures):
                    res, dur = f.result()
                    results.append(int(res))
                    times.append(float(dur))
        else:
            # sequential: preserve RNG state
            state = random.getstate()
            try:
                for i in range(runs):
                    s = base_seed + i
                    random.seed(s)
                    t1 = time.perf_counter()
                    m = SchellingModel(**params)
                    m.board = [row.copy() for row in snapshot]
                    m.gen = 0
                    m.unhappy_history.clear()
                    m.segregation_history.clear()
                    m.satisfaction_history.clear()
                    m._compute_neighbor_coords_cache()
                    res = int(m.init(max_generations=max_generations))
                    t2 = time.perf_counter()
                    results.append(res)
                    times.append(t2 - t1)
            finally:
                random.setstate(state)

        t_total = time.perf_counter() - t0
        converged = int(sum(results))
        fraction = float(converged) / float(runs)
        mean_t = float(statistics.mean(times)) if times else 0.0
        std_t = float(statistics.pstdev(times)) if times else 0.0

        stats = {
            'runs': runs,
            'converged': converged,
            'fraction': fraction,
            'total_time': t_total,
            'mean_time': mean_t,
            'std_time': std_t,
            'times': times,
        }

        if verbose:
            print(f'Ran {runs} simulations: {converged} converged ({fraction:.3f})')
            print(f'Total wall time: {t_total:.4f}s, mean per-run: {mean_t:.4f}s, std: {std_t:.4f}s')

        return stats

    def reset(self):
        self._init_board()
        # recompute neighbor cache in case parameters changed
        self._compute_neighbor_coords_cache()
