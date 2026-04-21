import argparse
import math
import traceback
import sys
from pathlib import Path

# Ensure repository root is on sys.path so `training` package can be imported
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.src.schelling import SchellingModel

GROUP_COLORS = [
    {'fill': '#B5D4F4', 'stroke': '#185FA5', 'name': 'Grupo A'},
    {'fill': '#F5C4B3', 'stroke': '#993C1D', 'name': 'Grupo B'},
    {'fill': '#C0DD97', 'stroke': '#3B6D11', 'name': 'Grupo C'},
    {'fill': '#F4C0D1', 'stroke': '#993556', 'name': 'Grupo D'},
    {'fill': '#FAC775', 'stroke': '#854F0B', 'name': 'Grupo E'},
]


def get_hex_layout(N, r=1.0):
    sqrt3 = math.sqrt(3)
    hexW = sqrt3 * r
    gridW = hexW * (N + 0.5)
    gridH = r * (1.5 * N + 0.5)
    return {'r': r, 'hexW': hexW, 'offsetX': 0, 'offsetY': 0, 'gridW': gridW, 'gridH': gridH}


def get_hex_center(i, j, layout):
    return (
        layout['offsetX'] + layout['hexW'] * (j + 0.5 + (i % 2) * 0.5),
        layout['offsetY'] + layout['r'] * (1 + i * 1.5),
    )


def animate_model(model: SchellingModel, interval=200, max_steps=1000, nogui=False, out_path=None):
    N = model.N
    # choose r so grid fits comfortably in unit coords
    r = 1.0
    layout = get_hex_layout(N, r)

    if nogui:
        # run model and save a snapshot + stats chart
        model.run(max_generations=max_steps)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import patches

        fig = plt.figure(figsize=(10, 5))
        ax_grid = fig.add_subplot(1, 2, 1)
        ax_chart = fig.add_subplot(2, 2, 2)
        ax_chart2 = fig.add_subplot(2, 2, 4)
    else:
        import matplotlib.pyplot as plt
        from matplotlib import patches
        from matplotlib.animation import FuncAnimation

        fig = plt.figure(figsize=(10, 5))
        ax_grid = fig.add_subplot(1, 2, 1)
        ax_chart = fig.add_subplot(2, 2, 2)
        ax_chart2 = fig.add_subplot(2, 2, 4)

    ax_grid.set_aspect('equal')
    ax_grid.set_xlim(-0.5, layout['gridW'] + 0.5)
    ax_grid.set_ylim(layout['gridH'] + 0.5, -0.5)
    ax_grid.axis('off')

    patches_list = [[None for _ in range(N)] for _ in range(N)]

    # create patches once
    for i in range(N):
        for j in range(N):
            x, y = get_hex_center(i, j, layout)
            poly = patches.RegularPolygon((x, y), numVertices=6, radius=layout['r'] * 0.98,
                                          orientation=math.radians(30), ec='gray', lw=0.3)
            ax_grid.add_patch(poly)
            patches_list[i][j] = poly

    line_unhappy, = ax_chart.plot([], [], '-o', color='#d97706')
    line_segr, = ax_chart2.plot([], [], '-o', color='#2563eb')
    ax_chart.set_ylim(0, 100)
    ax_chart2.set_ylim(0, 100)
    ax_chart.set_title('Insatisfechos (%)')
    ax_chart2.set_title('Segregación (%)')

    txt_gen = ax_grid.text(0.02, 0.02, '', transform=ax_grid.transAxes, fontsize=10,
                            bbox=dict(facecolor='white', alpha=0.7))

    anim_obj = {'obj': None}

    def update(frame):
        # perform a step for frames > 0 and capture unhappy count
        unhappy = None
        if frame > 0:
            unhappy = model.step()

        board = model.board
        for i in range(N):
            for j in range(N):
                v = int(board[i][j])
                poly = patches_list[i][j]
                if v == 0:
                    poly.set_facecolor('#e8e6e0')
                    poly.set_edgecolor('#cfcfcf')
                else:
                    c = GROUP_COLORS[(v - 1) % len(GROUP_COLORS)]
                    poly.set_facecolor(c['fill'])
                    poly.set_edgecolor(c['stroke'])
        txt_gen.set_text(f'Generación: {model.gen}')

        xs = list(range(max(1, len(model.unhappy_history))))
        ys = model.unhappy_history
        zs = model.segregation_history
        line_unhappy.set_data(xs, ys)
        ax_chart.set_xlim(0, max(10, len(xs)))
        line_segr.set_data(xs, zs)
        ax_chart2.set_xlim(0, max(10, len(xs)))
        # stop conditions: no unhappy agents or stagnation
        try:
            stopped = False
            if unhappy == 0:
                stopped = True
            elif model.converges():
                stopped = True
            elif model.should_stop_by_stagnation():
                stopped = True
            if stopped and anim_obj['obj'] is not None:
                anim_obj['obj'].event_source.stop()
                txt_gen.set_text(f'Generación: {model.gen} — detenido')
        except Exception:
            # don't let the animation crash; re-raise after printing
            traceback.print_exc()
            raise

        return []

    if nogui:
        # finalize and save
        plt.tight_layout()
        path = out_path or 'schelling_snapshot.png'
        fig.savefig(path, dpi=150)
        print('Saved snapshot to', path)
    else:
        from matplotlib.animation import FuncAnimation
        anim = FuncAnimation(fig, update, frames=range(max_steps), interval=interval, blit=False)
        anim_obj['obj'] = anim
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Run Schelling model animation')
    parser.add_argument('--groups', type=int, default=3)
    parser.add_argument('--radius', type=int, default=1)
    parser.add_argument('--size', type=int, default=35)
    parser.add_argument('--empty', type=float, default=0.15)
    parser.add_argument('--thresh', type=float, default=0.4)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--nogui', action='store_true', help='Run without GUI and save a snapshot')
    parser.add_argument('--out', type=str, default=None, help='Output path for snapshot when --nogui')
    parser.add_argument('--text', action='store_true', help='Run in text-only mode, printing stats to console')
    args = parser.parse_args()

    model = SchellingModel(num_groups=args.groups, num_neighbors=args.radius,
                           board_size=args.size, empty_percentage=args.empty,
                           tolerance_threshold=args.thresh)
    if args.text:
        # run in text-only mode (no matplotlib)
        for _ in range(args.steps):
            unhappy = model.step()
            total_nonempty = sum(1 for row in model.board for cell in row if cell != 0)
            unhappy_pct = 0 if total_nonempty == 0 else model.unhappy_history[-1]
            segr = model.segregation_history[-1]
            print(f'Gen={model.gen} unhappy={unhappy} ({unhappy_pct}%) segr={segr}%')
            if unhappy == 0:
                print('All agents happy — stopping')
                break
        return

    animate_model(model, interval=200, max_steps=args.steps, nogui=args.nogui, out_path=args.out)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
