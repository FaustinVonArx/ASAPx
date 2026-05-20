"""Off-screen pyvista renders of single-part-highlighted subassemblies.

Used by LLMDFASequencePlanner to build the images shown to a vision LLM during
frontier selection. Results are cached on disk by (subassembly, highlighted
part, image size) so re-runs and duplicate sub-states don't re-render.
"""

import hashlib
from pathlib import Path

import pyvista as pv

from assets.load import load_assembly


# Cache trimesh-loaded assembly per assembly_dir to avoid hitting disk per call.
_ASSEMBLY_CACHE = {}


def render_part_in_context(assembly_dir, subassembly, highlighted_part,
                           cache_dir, size=(512, 512)):
    """Render `subassembly` with `highlighted_part` in red, others in gray.

    Returns the PNG file path. Re-uses a cached PNG keyed by
    (frozenset(subassembly), highlighted_part, size).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    parts_sorted = sorted(map(str, subassembly))
    key_input = f"{parts_sorted}|{highlighted_part}|{size[0]}x{size[1]}"
    key = hashlib.sha256(key_input.encode()).hexdigest()[:16]
    out = cache_dir / f"part_{highlighted_part}_n{len(parts_sorted)}_{key}.png"
    if out.exists():
        return str(out)

    if assembly_dir not in _ASSEMBLY_CACHE:
        _ASSEMBLY_CACHE[assembly_dir] = load_assembly(assembly_dir)
    assembly = _ASSEMBLY_CACHE[assembly_dir]

    plotter = pv.Plotter(off_screen=True, window_size=tuple(size))
    for p in parts_sorted:
        if p not in assembly:
            continue
        mesh_tm = assembly[p].get('mesh') if isinstance(assembly[p], dict) else None
        if mesh_tm is None:
            continue
        mesh_pv = pv.wrap(mesh_tm)
        if str(p) == str(highlighted_part):
            plotter.add_mesh(mesh_pv, color='crimson', opacity=1.0, show_edges=False)
        else:
            plotter.add_mesh(mesh_pv, color='lightgray', opacity=0.55, show_edges=False)

    plotter.camera_position = 'iso'
    plotter.screenshot(str(out))
    plotter.close()
    return str(out)
