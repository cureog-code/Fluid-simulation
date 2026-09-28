import argparse
import queue
import struct
import subprocess
import threading
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
    parser.add_argument('--frames', type=int, default=5000)
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
    if channels != 3 or not (1 <= nx <= 2048 and 1 <= ny <= 2048):
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
                frame = np.frombuffer(raw, dtype=np.float32).reshape(3, ny, nx).copy()
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

    smoke_map = LinearSegmentedColormap.from_list('smoke', [(0, '#08101d'), (0.2, '#19365d'),
                                                (0.55, '#58a9c0'), (1, '#fff4d1')]).copy()
    vortex_map = plt.colormaps['RdBu_r'].copy()  # Weak: blue; strong: red
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
    view = {'mode': 0, 'paused': False, 'index': 0, 'data': None}
    def draw():
        data = view['data']
        labels = ('smoke', 'vorticity magnitude |ω|', 'speed |v|')
        if data is not None:
            # C marks every obstacle with dye=-1. Do not draw any additional fixed shapes
            solid = data[0] < 0
            if view['mode'] == 0:
                values, color_map, limits = data[0], smoke_map, (0, 0.9)
            elif view['mode'] == 1:
                values, color_map, limits = np.abs(data[1]), vortex_map, (0, 0.22)
            else:
                values, color_map, limits = data[2], speed_map, (0, 2.2)
            image.set_data(np.ma.array(values, mask=solid))
            image.set_cmap(color_map)
            image.set_clim(*limits)

        status = state['error'] or (
            'Finished' if state['done'] else
            'Display paused' if view['paused'] else 'Running'
        )
        ax.set_title(
            f"{args.scene} · {labels[view['mode']]} · frame {view['index']}/{args.frames} · {status}\n"
            '1: smoke   2: vorticity   3: speed   Space: pause display   Q: quit',
            color='white', fontsize=12,
        )
    def animate(_):
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
        return (image,)
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
    animation = FuncAnimation(fig, animate, interval=35, blit=False, cache_frame_data=False)
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