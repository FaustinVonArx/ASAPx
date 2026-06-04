"""Off-screen pyvista renders of single-part-highlighted subassemblies.

Used by LLMDFASequencePlanner to build the images shown to a vision LLM during
frontier selection. Results are cached on disk by (subassembly, highlighted
part, image size) so re-runs and duplicate sub-states don't re-render.
"""

import hashlib
from pathlib import Path

import numpy as np
import pyvista as pv

from assets.load import load_assembly


# Cache trimesh-loaded assembly per assembly_dir to avoid hitting disk per call.
_ASSEMBLY_CACHE = {}


def render_part_in_context(assembly_dir, subassembly, highlighted_part,
                           cache_dir, size=(512, 512), pose=None):
    """Render `subassembly` with `highlighted_part` in red, others in gray.

    pose: optional 4x4 homogeneous transform applied to every mesh before
    rendering, so the picture shows the assembly in the step-specific stable
    orientation (e.g. lying on its side) rather than the canonical config.
    Identity / None is treated as no extra transform.

    Returns the PNG file path. Re-uses a cached PNG keyed by
    (frozenset(subassembly), highlighted_part, size, pose).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    parts_sorted = sorted(map(str, subassembly))

    # Normalise pose into a hashable signature, treating identity as "no pose"
    # so calls without a pose hit the same cache entries they did pre-change.
    pose_arr = None
    pose_key = 'identity'
    if pose is not None:
        try:
            pose_arr = np.asarray(pose, dtype=float)
            if pose_arr.shape != (4, 4):
                pose_arr = None
            elif not np.allclose(pose_arr, np.eye(4)):
                # Quantize to 6 decimals so floating noise doesn't fragment the cache.
                pose_key = np.round(pose_arr, 6).tobytes().hex()[:32]
            else:
                pose_arr = None
        except (TypeError, ValueError):
            pose_arr = None

    key_input = f"{parts_sorted}|{highlighted_part}|{size[0]}x{size[1]}|pose:{pose_key}"
    key = hashlib.sha256(key_input.encode()).hexdigest()[:16]
    out = cache_dir / f"part_{highlighted_part}_n{len(parts_sorted)}_{key}.png"
    if out.exists():
        return str(out)

    if assembly_dir not in _ASSEMBLY_CACHE:
        _ASSEMBLY_CACHE[assembly_dir] = load_assembly(assembly_dir)
    assembly = _ASSEMBLY_CACHE[assembly_dir]

    no_highlight = highlighted_part is None
    plotter = pv.Plotter(off_screen=True, window_size=tuple(size))
    for p in parts_sorted:
        if p not in assembly:
            continue
        mesh_tm = assembly[p].get('mesh') if isinstance(assembly[p], dict) else None
        if mesh_tm is None:
            continue
        if pose_arr is not None:
            # Copy so the long-lived assembly cache isn't mutated.
            mesh_tm = mesh_tm.copy()
            mesh_tm.apply_transform(pose_arr)
        mesh_pv = pv.wrap(mesh_tm)
        if no_highlight:
            # Full-assembly view, no part singled out — used by the initial-pose
            # picker to show the whole assembly in each candidate orientation.
            plotter.add_mesh(mesh_pv, color='lightgray', opacity=1.0, show_edges=False)
        elif str(p) == str(highlighted_part):
            plotter.add_mesh(mesh_pv, color='crimson', opacity=1.0, show_edges=False)
        else:
            plotter.add_mesh(mesh_pv, color='lightgray', opacity=0.35, show_edges=False)

    plotter.camera_position = 'iso'
    plotter.screenshot(str(out))
    plotter.close()
    return str(out)


def render_unstable_parts(assembly_dir, parts, observed_fallen, save_path,
                          size=(1024, 1024), pose=None):
    """Render the full assembly with every part id in `observed_fallen`
    coloured crimson and the rest in light gray. Used by the planner to
    show the user which parts the gravity precheck observed falling when
    no stable initial pose could be verified.

    pose: optional 4x4 transform applied to each mesh (per-call copy) so
    the image reflects the orientation in which the parts fell. Identity
    or None renders the canonical orientation.

    Returns the saved PNG path on success, or None on failure (e.g.
    assembly load / pyvista error)."""
    try:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if assembly_dir not in _ASSEMBLY_CACHE:
            _ASSEMBLY_CACHE[assembly_dir] = load_assembly(assembly_dir)
        assembly = _ASSEMBLY_CACHE[assembly_dir]

        fallen_set = {str(p) for p in (observed_fallen or [])}

        pose_arr = None
        if pose is not None:
            try:
                _p = np.asarray(pose, dtype=float)
                if _p.shape == (4, 4) and not np.allclose(_p, np.eye(4)):
                    pose_arr = _p
            except (TypeError, ValueError):
                pose_arr = None

        plotter = pv.Plotter(off_screen=True, window_size=tuple(size))
        for p in sorted(map(str, parts)):
            if p not in assembly:
                continue
            data = assembly[p]
            mesh_tm = data.get('mesh') if isinstance(data, dict) else None
            if mesh_tm is None:
                continue
            if pose_arr is not None:
                mesh_tm = mesh_tm.copy()  # don't mutate the cached mesh
                mesh_tm.apply_transform(pose_arr)
            mesh_pv = pv.wrap(mesh_tm)
            if p in fallen_set:
                plotter.add_mesh(mesh_pv, color='crimson', opacity=1.0, show_edges=False)
            else:
                plotter.add_mesh(mesh_pv, color='lightgray', opacity=0.45, show_edges=False)
        plotter.camera_position = 'iso'
        plotter.screenshot(str(save_path))
        plotter.close()
        return str(save_path)
    except Exception as e:
        print(f'[render_unstable_parts] failed: {e}')
        return None
