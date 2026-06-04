import os
import shutil
from pathlib import Path

class SimRenderer:
    @staticmethod
    def replay(sim, record = False, record_path = None, make_video = False):
        if record:
            temp_folder_name = os.path.basename(record_path) + '_tmp'
            record_folder = os.path.join(Path(record_path).parent, temp_folder_name)
            os.makedirs(record_folder, exist_ok = True)
            sim.viewer_options.record = True
            sim.viewer_options.record_folder = record_folder
            loop = sim.viewer_options.loop
            infinite = sim.viewer_options.infinite
            sim.viewer_options.loop = False
            sim.viewer_options.infinite = False
            # Each viewer tick advances `num_steps = max(1, int(dt*speed/h + 0.5))`
            # sim steps from q_his. With defaults (fps=30, speed=1.0) and the
            # sim's h=1e-3 this is ~33 steps/tick, so short trajectories (e.g.
            # the gravity precheck's ~40-frame q_his) finish in 1-2 ticks and
            # only one PNG is captured — which `os.remove('0.png')` below then
            # deletes, leaving ffmpeg with an empty folder. For short q_his,
            # bump fps so we get ~30 frames captured regardless of length; for
            # long trajectories (>=900 entries) the defaults are fine.
            saved_fps = sim.viewer_options.fps
            saved_speed = sim.viewer_options.speed
            try:
                q_his_len = len(sim.get_q_his())
            except Exception:
                q_his_len = 0
            if 0 < q_his_len < 900:
                # num_steps ≈ q_his_len / 30 frames target. Solve:
                #   num_steps = max(1, int(dt*speed/h + 0.5)) ≈ q_his_len/30
                # With speed=1.0, h=1e-3 → dt = num_steps * 1e-3 → fps = 1/dt.
                target_steps = max(1, q_his_len // 30)
                sim.viewer_options.fps = int(round(1.0 / (target_steps * 1e-3)))
                sim.viewer_options.speed = 1.0

        sim.replay()

        if record:
            sim.viewer_options.fps = saved_fps
            sim.viewer_options.speed = saved_speed
            images_path = os.path.join(record_folder, r"%d.png")
            zero_png = os.path.join(record_folder, "0.png")
            if os.path.exists(zero_png):
                os.remove(zero_png)

            if make_video:
                os.system(f"ffmpeg -y -framerate 30 -i {images_path} -c:v libx264 -pix_fmt yuv420p {record_path} -hide_banner -loglevel error")
            else:
                palette_path = os.path.join(record_folder, 'palette.png')
                os.system("ffmpeg -y -i {} -vf palettegen {} -hide_banner -loglevel error".format(images_path, palette_path))
                os.system("ffmpeg -y -i {} -i {} -lavfi paletteuse {} -hide_banner -loglevel error".format(images_path, palette_path, record_path))

            shutil.rmtree(record_folder)

            sim.viewer_options.record = False
            sim.viewer_options.loop = loop
            sim.viewer_options.infinite = infinite
            
