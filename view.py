import argparse
import queue
import struct
import subprocess
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
 
VALID_OBSTACLES = ('none', 'cylinder', 'rectangle', 'ellipse', 'nozzle', 'airfoil', 'venturi', 'cylinder-array')

def read_exact(stream, count):
    chunks = []
    while count:
        chunk = stream.read(count)
        if not chunk:
            return None
        chunks.append(chunk)
        count -= len(chunk)
    return b''.join(chunks)

def main():
    parser = argparse.ArgumentParser(description='View the smoke simulation from the C program')
    parser.add_argument('--exe', type=Path, default=Path(__file__).with_name('fluid.exe'))
    parser.add_argument('--scene', choices=VALID_OBSTACLES, default='cylinder')
    parser.add_argument('--frames', type=int, default=20000)
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error('--frames must be a positive integer')
    if not args.exe.is_file():
        parser.error(f'Cannot find {args.exe}; compile fluid.c first')

    process = subprocess.Popen(
        [str(args.exe.resolve()), '--frames', str(args.frames), '--scene', args.scene],
        stdout=subprocess.PIPE,
        stderr=None,
        bufsize=0,
    )
    header = read_exact(process.stdout, 16)
    if header is None or header[:4] != b'SMK1':
        rc = process.wait()
        raise RuntimeError(f'Could not read header from C (exit code: {rc})')
    nx, ny, channels = struct.unpack('=III', header[4:])
    if channels != 4 or not (1 <= nx <= 2048 and 1 <= ny <= 2048):
        process.terminate()
        raise RuntimeError(f'Invalid header: {nx}x{ny}x{channels}')

    inbox = queue.Queue(maxsize=2)
    state = {'done': False, 'error': '', 'received': 0}
    frame_bytes = channels * nx * ny * np.dtype(np.float32).itemsize

    def receive():
        try:
            while True:
                raw = read_exact(process.stdout, frame_bytes)
                if raw is None:
                    break
                frame = np.frombuffer(raw, dtype=np.float32).reshape(channels, ny, nx).copy()
                state['received'] += 1
                if inbox.full():
                    try:
                        inbox.get_nowait() # Drop the old frame if C runs faster than the GU
                    except queue.Empty:
                        pass
                inbox.put_nowait((state['received'], frame))
            rc = process.wait()
            if rc != 0:
                state['error'] = f'C stopped with error code {rc}'
            elif state['received'] != args.frames:
                state['error'] = 'Data stream ended prematurely'
        except Exception as exc:
            state['error'] = str(exc)
        finally:
            state['done'] = True

    worker = threading.Thread(target=receive, daemon=True)
    worker.start()

    smoke_map = LinearSegmentedColormap.from_list('smoke', [(0, '#08101d'), (0.2, '#19365d'), (0.55, '#58a9c0'), (1, '#fff4d1')]).copy()
    # frame[1] = dv/dx - du/dy: negative is clockwise, positive is counterclockwise.
    vortex_map = plt.colormaps['RdBu'].copy()
    speed_map = plt.colormaps['magma'].copy()
    for color_map in (smoke_map, vortex_map, speed_map):
        color_map.set_bad('#343434')  # Solid cells sent from C have dye=-1
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#08101d')
    ax.set_facecolor('#08101d')
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    image = ax.imshow(
        np.zeros((ny, nx)), origin='lower', extent=(0, nx, 0, ny),
        interpolation='bilinear', cmap=smoke_map, vmin=0, vmax=0.9,
    )

    vector_spacing = max(6, min(nx, ny) // 20)
    vector_i = np.arange(vector_spacing // 2, nx, vector_spacing)
    vector_j = np.arange(vector_spacing // 2, ny, vector_spacing)
    vector_x, vector_y = np.meshgrid(vector_i + 0.5, vector_j + 0.5)
    vector_zeros = np.zeros_like(vector_x, dtype=np.float32)
    velocity_arrows = ax.quiver(vector_x, vector_y, vector_zeros, vector_zeros, color='white', angles='xy', scale_units='xy', scale=1.0, width=0.0022, headwidth=3.2, headlength=4.2, pivot='mid', zorder=3)
    velocity_arrows.set_visible(False)

    view = {
        'mode': 0,
        'paused': False,
        'index': 0,
        'data': None,
        'fps_t': time.perf_counter(),
        'fps_received': 0,
        'fps_drawn': 0,
        'sim_fps': 0,
        'view_fps': 0,        
    }
    def draw():
        data = view['data']
        labels = ('smoke', 'vorticity', 'velocity field')
        if data is not None:
            solid = data[0] < 0
            if view['mode'] == 0:
                values, color_map, limits = data[0], smoke_map, (0, 0.9)
                velocity_arrows.set_visible(False)
            elif view['mode'] == 1:
                values, color_map, limits = data[1], vortex_map, (-0.22, 0.22)
                velocity_arrows.set_visible(False)
            else:
                ux = data[2]
                vy = data[3]
                values = np.hypot(ux, vy)
                fluid_speed = values[~solid]
                speed_limit = max(
                    float(np.percentile(fluid_speed, 99.0)) if fluid_speed.size else 1.0,
                    1.0e-6,
                )
                color_map, limits = speed_map, (0, speed_limit)

                sampled_ux = ux[np.ix_(vector_j, vector_i)]
                sampled_vy = vy[np.ix_(vector_j, vector_i)]
                sampled_speed = np.hypot(sampled_ux, sampled_vy)
                sampled_solid = solid[np.ix_(vector_j, vector_i)]

                maximum_arrow_length = 0.72 * vector_spacing
                arrow_length = maximum_arrow_length * np.clip(
                    sampled_speed / speed_limit, 0.0, 1.0
                )
                display_u = np.zeros_like(sampled_ux)
                display_v = np.zeros_like(sampled_vy)
                moving = sampled_speed > 1.0e-8
                display_u[moving] = (sampled_ux[moving] / sampled_speed[moving] * arrow_length[moving])
                display_v[moving] = (sampled_vy[moving] / sampled_speed[moving] * arrow_length[moving])

                velocity_arrows.set_UVC(np.ma.array(display_u, mask=sampled_solid), np.ma.array(display_v, mask=sampled_solid))
                velocity_arrows.set_visible(True)
            image.set_data(np.ma.array(values, mask=solid))
            image.set_cmap(color_map)
            image.set_clim(*limits)

        status = state['error'] or (
            'Finished' if state['done'] else
            'Display paused' if view['paused'] else 'Running'
        )
        ax.set_title(
            f"{args.scene} | {labels[view['mode']]} | frame {view['index']}/{args.frames} | "
            f"sim {view['sim_fps']:.1f} fps | view {view['view_fps']:.1f} fps | {status}\n"
            '1: smoke   2: vorticity   3: velocity field   Space: pause display   Q: quit',
            color='white', fontsize=12,
        )
    def animate(_):
        now = time.perf_counter()
        view['fps_drawn'] += 1
        elapsed = now - view['fps_t']
        if elapsed >= 0.5:
            view['sim_fps'] = (state['received'] - view['fps_received']) / elapsed
            view['view_fps'] = view['fps_drawn'] / elapsed
            view['fps_t'] = now
            view['fps_received'] = state['received']
            view['fps_drawn'] = 0
        if not view['paused']:
            latest = None
            while True:
                try:
                    latest = inbox.get_nowait()
                except queue.Empty:
                    break
            if latest is not None:
                view['index'], view['data'] = latest
        draw()
        return (image, velocity_arrows)
    def on_key(event):
        if event.key in ('q', 'escape'):
            plt.close(fig)
        elif event.key in ('1', '2', '3'):
            view['mode'] = int(event.key) - 1
            draw()
            fig.canvas.draw_idle()
        elif event.key == ' ':
            view['paused'] = not view['paused']
    def on_close(_):
        if process.poll() is None:
            process.terminate()
        process.stdout.close()
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('close_event', on_close)
    animation = FuncAnimation(fig, animate, interval=16, blit=False, cache_frame_data=False)
    try:
        plt.show()
    finally:
        if process.poll() is None:
            process.terminate()
        if not process.stdout.closed:
            process.stdout.close()
    _ = animation

if __name__ == '__main__':
    main()
