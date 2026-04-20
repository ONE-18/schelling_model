import random
import tkinter as tk
from tkinter import ttk


EMPTY = 0
A = 1
B = 2


class SchellingApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Modelo de segregación de Schelling")
        self.root.geometry("860x760")
        self.root.minsize(760, 680)

        self.grid_size_var = tk.IntVar(value=30)
        self.empty_var = tk.IntVar(value=15)
        self.threshold_var = tk.IntVar(value=40)
        self.speed_var = tk.IntVar(value=5)

        self.running = False
        self.after_id = None
        self.generation = 0
        self.grid = []

        self._build_ui()
        self.init_grid()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        container = ttk.Frame(self.root, padding=16)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)

        controls = ttk.Frame(container)
        controls.grid(row=0, column=0, sticky="ew")
        for i in range(4):
            controls.columnconfigure(i, weight=1)

        self._add_slider(
            controls,
            0,
            "Tamaño de cuadrícula",
            self.grid_size_var,
            10,
            60,
            5,
            self._on_grid_size_change,
            suffix=" × {value}",
        )
        self._add_slider(
            controls,
            1,
            "% celdas vacías",
            self.empty_var,
            5,
            40,
            5,
            self._on_rebuild_change,
            suffix="%",
        )
        self._add_slider(
            controls,
            2,
            "Umbral de tolerancia",
            self.threshold_var,
            0,
            100,
            5,
            self._on_rebuild_change,
            suffix="%",
        )
        self._add_slider(
            controls,
            3,
            "Velocidad",
            self.speed_var,
            1,
            10,
            1,
            self._on_speed_change,
        )

        buttons = ttk.Frame(container)
        buttons.grid(row=1, column=0, sticky="ew", pady=(14, 10))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        buttons.columnconfigure(2, weight=1)
        buttons.columnconfigure(3, weight=1)

        ttk.Button(buttons, text="Reiniciar", command=self.reset).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(buttons, text="Paso", command=self.step).grid(row=0, column=1, sticky="ew", padx=8)
        self.play_button = ttk.Button(buttons, text="▶ Ejecutar", command=self.toggle_play)
        self.play_button.grid(row=0, column=2, sticky="ew", padx=8)

        canvas_frame = ttk.Frame(container)
        canvas_frame.grid(row=2, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.canvas_size = 540
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_size, height=self.canvas_size, highlightthickness=1)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        stats = ttk.Frame(container)
        stats.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        for i in range(3):
            stats.columnconfigure(i, weight=1)

        self.gen_value = self._make_stat(stats, 0, "Generación", "0")
        self.unhappy_value = self._make_stat(stats, 1, "Insatisfechos", "—")
        self.segr_value = self._make_stat(stats, 2, "Segregación", "—")

        self.status_var = tk.StringVar(value="Listo")
        status = ttk.Label(container, textvariable=self.status_var, anchor="w")
        status.grid(row=4, column=0, sticky="ew", pady=(10, 0))

    def _add_slider(self, parent, col, label, var, from_, to, resolution, callback, suffix=""):
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=col, sticky="ew", padx=6)
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w")
        scale = ttk.Scale(frame, from_=from_, to=to, orient="horizontal")
        scale.grid(row=1, column=0, sticky="ew", pady=4)
        value_label = ttk.Label(frame, text="")
        value_label.grid(row=2, column=0, sticky="w")

        def update_from_scale(_event=None):
            value = int(round(float(scale.get()) / resolution) * resolution)
            value = max(from_, min(to, value))
            if var.get() != value:
                var.set(value)
            self._update_slider_label(label, value_label, value, suffix)
            callback()

        def update_from_var(*_args):
            value = var.get()
            scale.set(value)
            self._update_slider_label(label, value_label, value, suffix)

        scale.configure(command=lambda _value: update_from_scale())
        var.trace_add("write", lambda *_args: update_from_var())
        scale.set(var.get())
        update_from_var()
        return frame

    def _update_slider_label(self, label, widget, value, suffix) -> None:
        if label == "Tamaño de cuadrícula":
            widget.configure(text=f"{value} × {value}")
        else:
            widget.configure(text=f"{value}{suffix}")

    def _make_stat(self, parent, col, label, value):
        frame = ttk.LabelFrame(parent, text=label, padding=10)
        frame.grid(row=0, column=col, sticky="ew", padx=6)
        value_label = ttk.Label(frame, text=value, font=("Segoe UI", 18, "bold"))
        value_label.pack(anchor="center")
        return value_label

    def init_grid(self) -> None:
        self.stop()
        self.n = self.grid_size_var.get()
        empty_pct = self.empty_var.get() / 100.0
        self.grid = []
        for _ in range(self.n):
            row = []
            for _ in range(self.n):
                r = random.random()
                if r < empty_pct:
                    row.append(EMPTY)
                elif r < empty_pct + (1 - empty_pct) / 2:
                    row.append(A)
                else:
                    row.append(B)
            self.grid.append(row)
        self.generation = 0
        self.status_var.set("Simulación reiniciada")
        self.draw()
        self.update_stats()

    def neighbors(self, i, j):
        res = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.n and 0 <= nj < self.n:
                    res.append(self.grid[ni][nj])
        return res

    def is_happy(self, i, j):
        value = self.grid[i][j]
        if value == EMPTY:
            return True
        threshold = self.threshold_var.get() / 100.0
        nbrs = [x for x in self.neighbors(i, j) if x != EMPTY]
        if not nbrs:
            return True
        same = sum(1 for x in nbrs if x == value)
        return same / len(nbrs) >= threshold

    def step(self) -> None:
        unhappy = []
        empty = []
        for i in range(self.n):
            for j in range(self.n):
                cell = self.grid[i][j]
                if cell == EMPTY:
                    empty.append((i, j))
                elif not self.is_happy(i, j):
                    unhappy.append((i, j))

        random.shuffle(unhappy)
        random.shuffle(empty)
        moves = min(len(unhappy), len(empty))
        for k in range(moves):
            ui, uj = unhappy[k]
            ei, ej = empty[k]
            self.grid[ei][ej] = self.grid[ui][uj]
            self.grid[ui][uj] = EMPTY

        self.generation += 1
        self.draw()
        self.update_stats(len(unhappy))

        if len(unhappy) == 0 and self.running:
            self.stop()
            self.status_var.set("La simulación se estabilizó")

    def segregation_index(self) -> int:
        same = 0
        total = 0
        for i in range(self.n):
            for j in range(self.n):
                value = self.grid[i][j]
                if value == EMPTY:
                    continue
                nbrs = [x for x in self.neighbors(i, j) if x != EMPTY]
                if not nbrs:
                    continue
                same += sum(1 for x in nbrs if x == value)
                total += len(nbrs)
        return 0 if total == 0 else round(same / total * 100)

    def update_stats(self, unhappy_count=None) -> None:
        self.gen_value.configure(text=str(self.generation))
        if unhappy_count is not None:
            occupied = sum(1 for row in self.grid for x in row if x != EMPTY)
            pct = 0 if occupied == 0 else round(unhappy_count / occupied * 100)
            self.unhappy_value.configure(text=f"{unhappy_count} ({pct}%)")
        self.segr_value.configure(text=f"{self.segregation_index()}%")

    def draw(self) -> None:
        self.canvas.delete("all")
        width = int(self.canvas.winfo_width() or self.canvas_size)
        height = int(self.canvas.winfo_height() or self.canvas_size)
        size = min(width, height)
        cell_size = size / self.n
        bg = "#f5f5f3"
        self.canvas.configure(background=bg)

        for i in range(self.n):
            for j in range(self.n):
                x1 = j * cell_size
                y1 = i * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                cell = self.grid[i][j]
                if cell == EMPTY:
                    color = bg
                    outline = bg
                elif cell == A:
                    color = "#B5D4F4"
                    outline = "#185FA5"
                else:
                    color = "#F5C4B3"
                    outline = "#993C1D"

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=outline if cell != EMPTY and cell_size > 6 else color, width=1 if cell != EMPTY and cell_size > 6 else 0)

    def stop(self) -> None:
        self.running = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.play_button.configure(text="▶ Ejecutar")

    def toggle_play(self) -> None:
        if self.running:
            self.stop()
            self.status_var.set("Pausado")
            return
        self.running = True
        self.play_button.configure(text="⏸ Pausar")
        self.status_var.set("Ejecutando")
        self._schedule_next_step()

    def _schedule_next_step(self) -> None:
        if not self.running:
            return
        delay_ms = (11 - self.speed_var.get()) * 60
        self.after_id = self.root.after(delay_ms, self._run_step_and_reschedule)

    def _run_step_and_reschedule(self) -> None:
        self.step()
        if self.running:
            self._schedule_next_step()

    def reset(self) -> None:
        self.init_grid()

    def _on_grid_size_change(self) -> None:
        self.reset()

    def _on_rebuild_change(self) -> None:
        self.reset()

    def _on_speed_change(self) -> None:
        if self.running:
            self.stop()
            self.toggle_play()


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    SchellingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
