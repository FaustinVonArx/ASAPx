import os
import shutil
import sys
import tempfile
from pathlib import Path

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

from time import time
import numpy as np
import redmax_py as redmax
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

from assets.load import load_assembly, load_part_ids
from assets.transform import transform_pt_by_matrix
from plan_path.run_connect import ConnectPathPlanner
from plan_sequence.sim_string import get_path_sim_string, get_stability_sim_string, get_contact_sim_string
from utils.renderer import SimRenderer


# Parameters for physics-based planner

FORCE_MAG = 1e2
FRAME_SKIP = 100
MAX_TIME = 180
CONTACT_EPS = 1e-1
# Both thresholds are now FRACTIONS of the assembly's initial bounding-box
# diagonal (computed once at t=0 over the bbox of all part centroids — moving
# and fixed). A part is flagged as fallen if its COG-relative displacement
# exceeds `POS_FAR_THRESHOLD × initial_diag`; it counts as settled when the
# last `NEAR_STEP` frames stay within a `POS_NEAR_THRESHOLD × initial_diag`
# band. Was previously in absolute world-units (m) which only made sense for
# assemblies near one scale.
POS_FAR_THRESHOLD = 0.4
POS_NEAR_THRESHOLD = 0.03
NEAR_STEP = 20
MAX_STEP = 100
CHECK_FREQ = 20
COL_TH_PATH = 0.01
COL_TH_STABLE = 0.00
MIN_SEP = 0.5
DEBUG_SIM = False
DEBUG_STABILITY = False

DOF_MAX_TIME = 5.0  # per-direction probe timeout (s); shorter than path-planner's MAX_TIME


class State:

    def __init__(self, q, qdot):
        self.q = q
        self.qdot = qdot

    def __repr__(self):
        return f'[State object at {hex(id(self))}]'


class MultiPartPathPlanner:

    frame_skip = FRAME_SKIP
    max_time = MAX_TIME
    contact_eps = CONTACT_EPS
    col_th_base = COL_TH_PATH
    min_sep = MIN_SEP

    def __init__(self, asset_folder, assembly_dir, parts_fix, part_move, parts_removed=[], pose=None, force_mag=FORCE_MAG, save_sdf=False,
        camera_pos=None, camera_lookat=None, floor=False):
        model_string = get_contact_sim_string(assembly_dir, parts_fix + [part_move] + parts_removed, save_sdf=save_sdf)
        sim = redmax.Simulation(model_string, asset_folder)
        col_th_dict = self._compute_col_th_dict(sim, parts_fix + parts_removed, part_move)

        model_string = get_path_sim_string(assembly_dir, parts_fix, part_move, parts_removed=parts_removed,
            save_sdf=save_sdf, pose=pose, col_th=col_th_dict, floor=floor)
        self.sim = redmax.Simulation(model_string, asset_folder)
        if camera_pos is not None: self.sim.viewer_options.camera_pos = camera_pos
        if camera_lookat is not None: self.sim.viewer_options.camera_lookat = camera_lookat
        self.asset_folder = asset_folder
        self.assembly_dir = assembly_dir
        self.parts_fix = parts_fix
        self.part_move = part_move
        self.parts_removed = parts_removed
        self.pose = pose
        self.force_mag = force_mag

        # connect path planner
        self.connect_path_planner = ConnectPathPlanner(assembly_dir, min_sep=self.min_sep)
        
    def _compute_col_th_dict(self, sim, parts_fix, part_move):
        col_th_dict = {}
        for part_fix in parts_fix:
            d_mf = sim.get_body_distance(f'part{part_move}', f'part{part_fix}')
            d_fm = sim.get_body_distance(f'part{part_fix}', f'part{part_move}')
            col_th_dict[part_fix] = max(0, max(-d_mf, -d_fm)) + self.col_th_base
        col_th_dict[part_move] = max(col_th_dict.values())
        return col_th_dict

    def get_state(self):
        q = self.sim.get_joint_qm(f'part{self.part_move}')
        qdot = self.sim.get_joint_qmdot(f'part{self.part_move}')
        return State(q, qdot)

    def set_state(self, state):
        qm = state.q
        self.sim.set_joint_qm(f'part{self.part_move}', qm)
        self.sim.zero_joint_qdot(f'part{self.part_move}')
        self.sim.update_robot()

    def apply_action(self, action):
        assert len(action) == 3
        action = action * self.force_mag
        force = np.concatenate([np.zeros(3), action])
        self.sim.set_body_external_force(f'part{self.part_move}', force)

    def is_disassembled(self, min_sep=None):
        in_contact = False
        if min_sep is None: min_sep = self.min_sep
        for part_fix in self.parts_fix: # if any part in contact, then not fully disassembled
            in_contact = in_contact or self.sim.body_in_contact(f'part{part_fix}', f'part{self.part_move}', min_sep)
        return not in_contact # if all movable parts are not in contact with fixed parts, then fully disassembled
    
    def check_success(self, action, return_path=False, min_sep=None):

        self.sim.reset()
        self.apply_action(action)

        t_start = time()
        step = 0
        path = []

        while True:

            self.set_state(self.get_state())
            state = self.get_state()
            last_qdot = state.qdot[:3]
            path.append(state.q)

            for _ in range(self.frame_skip):
                self.sim.forward(1, verbose=False)
                new_state = self.get_state()
                path.append(new_state.q)

                t_plan = time() - t_start
                if t_plan > self.max_time:
                    if return_path:
                        return False, path
                    else:
                        return False

            if self.is_disassembled(min_sep):
                break

            qdot = new_state.qdot[:3] # measure translation qdot only
            qdotdot = (qdot - last_qdot) / self.sim.options.h / self.frame_skip
            # if self.pose is not None:
            #     qdotdot = np.dot(qdotdot, self.pose[:3, :3].T) # revert to local frame
            # qdotdot = np.dot(qdotdot, action) 

            if np.linalg.norm(qdotdot) < 0.01 * self.force_mag:
                if return_path:
                    return False, path
                else:
                    return False

            step += 1

        if return_path:
            return True, path
        else:
            return True

    def compute_dof(self, max_time=DOF_MAX_TIME):
        '''
        Probe each of 6 assembly-local axis-aligned directions using the same
        logic as check_success: run frame_skip forward steps per chunk, repeat
        until the part is fully disassembled from its neighbours (DoF=1) or
        qdotdot drops below 0.01*force_mag / times out (DoF=0).
        Local directions are rotated by self.pose[:3,:3] before being applied
        so the returned 6-vector is pose-invariant.
        Returns a length-6 numpy int array in local-frame order
        [+Z, -Z, +X, -X, +Y, -Y].
        '''
        t_start_all = time()
        local_directions = [
            np.array([0, 0, 1]),
            np.array([0, 0, -1]),
            np.array([1, 0, 0]),
            np.array([-1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, -1, 0]),
        ]
        R = self.pose[:3, :3] if self.pose is not None else np.eye(3)
        directions = [R @ d for d in local_directions]
        dof = np.zeros(6, dtype=int)

        for i, direction in enumerate(directions):
            self.sim.reset()
            self.apply_action(direction)
            t_probe = time()
            free = False

            while True:
                self.set_state(self.get_state())
                state = self.get_state()
                last_qdot = state.qdot[:3]

                timed_out = False
                for _ in range(self.frame_skip):
                    self.sim.forward(1, verbose=False)
                    if time() - t_probe > max_time:
                        timed_out = True
                        break

                if timed_out:
                    break

                if self.is_disassembled():
                    free = True
                    break

                new_state = self.get_state()
                qdot = new_state.qdot[:3]
                qdotdot = (qdot - last_qdot) / self.sim.options.h / self.frame_skip
                if np.linalg.norm(qdotdot) < 0.01 * self.force_mag:
                    break

            dof[i] = 1 if free else 0

        self.sim.reset()
        elapsed = time() - t_start_all
        #print(f'[compute_dof] elapsed: {elapsed:.3f}s  dof: {dof.tolist()}')
        return dof

    def check_tool(self, tools, asset_folder_bfs=None, output_dir=None, show=False, verbose=False):
        """Tool-based feasibility check for the current (parts_fix + part_move) sub-assembly.

        Tries each tool in `tools` (a list of `Tool` instances with cached
        `direction`/`contact_point`) to determine whether any of them can be
        geometrically applied to `self.part_move` and whether the resulting
        tool — and the tool+part union — can be path-planned out of the
        assembly. No VLM/LLM calls are made.

        Args:
            tools: list of `Tool` instances (already scaled to the assembly).
            asset_folder_bfs: redmax asset folder used by the BFS path planner
                invoked inside the pipeline. Defaults to `<repo>/ATA/assets`.
            output_dir: optional directory for tool-placement screenshots.
            show, verbose: forwarded.

        Returns:
            dict {'tool_id', 'tool_mesh', 'inverted'} on success, or None when
            no tool in `tools` is feasible.
        """
        return check_tool(
            asset_folder=self.asset_folder,
            assembly_dir=self.assembly_dir,
            parts_fix=self.parts_fix,
            part_move=self.part_move,
            tools=tools,
            asset_folder_bfs=asset_folder_bfs,
            output_dir=output_dir,
            show=show,
            verbose=verbose,
        )

    def plan_path(self, action, rotation=False, connect_path=False):
        success, path = self.check_success(action, return_path=True)
        if success and connect_path:
            cp = self.connect_path_planner.plan(self.part_move, self.parts_fix, self.parts_removed,
                rotation=rotation, final_state=path[-1])
            if cp is not None:
                path += cp
        return success, path

    def render(self, path=None, reverse=False, record_path=None, make_video=False):
        q_his, qdot_his = self.sim.get_q_his(), self.sim.get_qdot_his()
        if path is not None:
            # assume path is global coordinate
            path = [self.sim.get_joint_q_from_qm(f'part{self.part_move}', qm) for qm in path]
        if reverse:
            path = q_his[::-1] if path is None else path[::-1]
            self.sim.set_state_his(path, [np.zeros(6) for _ in range(len(path))])
        else:
            if path is not None:
                self.sim.set_state_his(path, [np.zeros(6) for _ in range(len(path))])
        SimRenderer.replay(self.sim, record=record_path is not None, record_path=record_path, make_video=make_video)
        self.sim.set_state_his(q_his, qdot_his)
        return self.sim.export_replay_matrices()

    def render_with_tool(self, tool_mesh, path=None, reverse=False, record_path=None,
                         make_video=False, tool_color=None, camera_pos=None, camera_lookat=None,
                         body_color_dict=None):
        '''
        Like ``render`` but with ``tool_mesh`` rigidly attached to the moving part so it
        rides along the planned trajectory. ``tool_mesh`` must already be positioned in
        the moving part's OBJ frame (e.g. the output of
        ``ToolAnalyzer._apply_tool_geometric`` is exactly this).

        Internally builds a fresh redmax sim using ``get_path_sim_string(tool_attach=...)``,
        which nests the tool as a fixed child-link of the moving part. The path's joint
        states are copied over so the visual matches a regular ``render`` call.
        '''
        tmp_dir = Path(tempfile.mkdtemp(prefix='tool_render_'))
        tmp_obj = tmp_dir / 'tool.obj'
        tool_mesh.export(str(tmp_obj))

        try:
            # Recompute collision thresholds the same way __init__ does (the contact sim
            # is cheap and stateless, so it's fine to spin a fresh one).
            contact_string = get_contact_sim_string(
                self.assembly_dir,
                self.parts_fix + [self.part_move] + self.parts_removed,
            )
            contact_sim = redmax.Simulation(contact_string, self.asset_folder)
            col_th_dict = self._compute_col_th_dict(
                contact_sim, self.parts_fix + self.parts_removed, self.part_move,
            )

            tool_attach = {
                'filename': str(tmp_obj),
                'color': tool_color or '0.9 0.45 0.1 1.0',
            }
            model_string = get_path_sim_string(
                self.assembly_dir, self.parts_fix, self.part_move,
                parts_removed=self.parts_removed, pose=self.pose, col_th=col_th_dict,
                tool_attach=tool_attach,
            )
            render_sim = redmax.Simulation(model_string, self.asset_folder)

            # Match camera framing with the original sim (or any override).
            cam_pos = camera_pos if camera_pos is not None else self.sim.viewer_options.camera_pos
            cam_look = camera_lookat if camera_lookat is not None else self.sim.viewer_options.camera_lookat
            render_sim.viewer_options.camera_pos = cam_pos
            render_sim.viewer_options.camera_lookat = cam_look
            if body_color_dict is not None:
                render_sim.set_body_color_map(body_color_dict)

            if path is not None:
                path_q = [render_sim.get_joint_q_from_qm(f'part{self.part_move}', qm) for qm in path]
            else:
                path_q = list(self.sim.get_q_his())

            if reverse:
                path_q = path_q[::-1]
            render_sim.set_state_his(path_q, [np.zeros(6) for _ in range(len(path_q))])

            SimRenderer.replay(
                render_sim,
                record=record_path is not None,
                record_path=record_path,
                make_video=make_video,
            )
            return render_sim.export_replay_matrices()
        finally:
            shutil.rmtree(str(tmp_dir), ignore_errors=True)


class StabilityChecker:

    contact_eps = CONTACT_EPS
    pos_far_threshold = POS_FAR_THRESHOLD
    pos_near_threshold = POS_NEAR_THRESHOLD
    near_step = NEAR_STEP

    def __init__(self, allow_gap):
        self.sim = None
        self.parts_move = None
        self.parts_fix = None
        self.G = None
        self.pos_his_map = None
        self.dist_his_map = None
        self.n_step = 0
        # When set by the caller, every check_fallen_parts invocation appends a
        # diagnostic line to this file: live contact graph edges, parts_move /
        # parts_fix, and per-part fall reason (distance-threshold or
        # contact-lost). See MultiPartStabilityPlanner.check_success for where
        # this gets populated from record_path.
        self.log_path = None
        if allow_gap:
            self.pos_far_threshold = np.inf

    def _log(self, msg):
        if not self.log_path:
            return
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, 'a') as fp:
                fp.write(msg + '\n')
        except OSError:
            pass  # silently drop log writes if filesystem is being uncooperative

    def _render_contact_graph(self, live_G, parts_fallen, fall_reasons):
        """Save a PNG of the live contact graph alongside self.log_path:
        <log_stem>_check{step:04d}.png. parts_fix shown in dark gray, fallen
        parts in red, the rest of parts_move in sky blue. Layout is
        deterministic (spring with fixed seed) so successive frames are
        visually comparable."""
        if not self.log_path or live_G is None:
            return
        try:
            stem, _ = os.path.splitext(self.log_path)
            png_path = f'{stem}_check{self.n_step:04d}.png'
            os.makedirs(os.path.dirname(png_path), exist_ok=True)

            fixed_set = self._fixed_set
            fallen_set = set(parts_fallen)
            node_colors = []
            for n in live_G.nodes():
                if n in fixed_set:
                    node_colors.append('#404040')          # dark gray: held
                elif n in fallen_set:
                    node_colors.append('#d62728')          # red: fallen
                else:
                    node_colors.append('#7fb3d5')          # sky blue: moving + OK

            pos = nx.spring_layout(live_G, seed=0)
            fig, ax = plt.subplots(figsize=(6, 6))
            nx.draw(
                live_G, pos, ax=ax, with_labels=True,
                node_color=node_colors, edge_color='#999999',
                node_size=550, font_size=8, font_color='white',
            )
            title = (f'step={self.n_step}   '
                     f'|move|={len(self.parts_move)}   '
                     f'|fix|={len(self.parts_fix)}   '
                     f'|fall|={len(parts_fallen)}')
            if fall_reasons:
                # Compact one-line summary of why each part was flagged.
                reasons = '  '.join(f'{p}:{r}' for p, r in fall_reasons.items())
                title += f'\n{reasons}'
            ax.set_title(title, fontsize=9)
            fig.tight_layout()
            fig.savefig(png_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
        except Exception as _e:
            # Don't kill a real run for a viz hiccup.
            try:
                self._log(f'[step={self.n_step}] WARN graph render failed: {_e}')
            except Exception:
                pass

    def get_part_pos(self, part):
        return self.sim.get_joint_qm(f'part{part}')[:3]

    def _get_any_part_pos(self, part):
        """World position for a part regardless of joint type.

        `get_joint_qm` works only for free-DoF joints (free3d-exp on
        `parts_move`); on fixed-DoF joints (`parts_fix`) the C++ binding
        prints `[Error] get_joint_qm: joint ndof not supported: 0` to stderr
        before raising. We route by membership in `_fixed_set` (populated in
        update_sim) so the C++ error path is never reached. Fixed parts use
        the body-to-world transform's translation column instead, which is
        defined for all bodies regardless of joint DoF."""
        if part in self._fixed_set:
            E0j = np.asarray(self.sim.get_body_E0j(f'part{part}'))
            return E0j[:3, 3].copy()
        return self.sim.get_joint_qm(f'part{part}')[:3]

    def derive_contact_graph(self):
        G = nx.Graph()
        for i in range(len(self.parts_move)):
            G.add_node(self.parts_move[i])
            for j in range(i + 1, len(self.parts_move)):
                in_contact = self.sim.body_in_contact(f'part{self.parts_move[i]}', f'part{self.parts_move[j]}', self.contact_eps)
                if in_contact:
                    G.add_edge(self.parts_move[i], self.parts_move[j])
            for part_fix in self.parts_fix:
                in_contact = self.sim.body_in_contact(f'part{self.parts_move[i]}', f'part{part_fix}', self.contact_eps)
                if in_contact:
                    G.add_edge(self.parts_move[i], part_fix)
        return G

    def update_sim(self, sim, parts_move, parts_fix):
        self.sim = sim
        self.parts_move = parts_move
        self.parts_fix = parts_fix
        # Frozen set for fast O(1) membership lookup in `_get_any_part_pos`,
        # which routes the position fetch to `get_body_E0j` (works on fixed
        # joints) vs `get_joint_qm` (free joints only). Rebuilt each
        # update_sim call since the adaptive planner shifts parts move→fix.
        self._fixed_set = frozenset(parts_fix)
        self.G = self.derive_contact_graph()
        # Mass-weighted COG frame: pull each body's mass once (gravity-dependent
        # cost on sim init, but a no-op afterwards). Cover both `parts_move`
        # and `parts_fix` so the COG includes the (stationary) held parts as an
        # anchor — without them the COG can drift away with a falling cluster.
        if not hasattr(self, '_mass_map') or self._mass_map is None:
            self._mass_map = {}
        for p in list(self.parts_move) + list(self.parts_fix):
            if p not in self._mass_map:
                self._mass_map[p] = float(self.sim.get_body_mass(f'part{p}'))
        # Reference geometry, taken once at the first update_sim call (= sim
        # t=0). Later update_sim calls (e.g., the adaptive planner shifting
        # parts move→fix mid-run) intentionally keep the same reference so
        # the metric stays "displacement vs t=0," not "displacement vs the
        # most recent re-init."
        if self.pos_his_map is None:
            self.pos_his_map = {part: [self.get_part_pos(part)] for part in self.parts_move}
            self._cog_initial = self._compute_cog()
            self._r_initial = {
                p: float(np.linalg.norm(self.get_part_pos(p) - self._cog_initial))
                for p in self.parts_move
            }
            # Initial bounding-box diagonal of the assembly (centroid AABB
            # over parts_move + parts_fix at t=0). Used to scale the fall /
            # settle thresholds so the same constants behave the same on a
            # 0.1 m IKEA shelf and a 2 m chassis. Caveat: this is the AABB
            # of part *centroids*, not full mesh extents — slightly
            # underestimates true assembly size for parts with off-centre
            # geometry. Adequate for threshold-scaling.
            all_parts = list(self.parts_move) + list(self.parts_fix)
            if all_parts:
                positions = np.stack(
                    [self._get_any_part_pos(p) for p in all_parts], axis=0,
                )
                diag = float(np.linalg.norm(
                    positions.max(axis=0) - positions.min(axis=0)
                ))
            else:
                diag = 0.0
            # Guard against degenerate cases (single part / coincident
            # centroids): falling back to 1.0 keeps the threshold equal to
            # the raw constant, which is the closest analogue of the old
            # absolute-units behaviour.
            self._initial_diag = diag if diag > 1e-6 else 1.0
        if self.dist_his_map is None:
            self.dist_his_map = {part: [0.0] for part in self.parts_move}

    def _compute_cog(self):
        """Mass-weighted centre of gravity over `parts_move + parts_fix`.
        Uses `_get_any_part_pos` so fixed (0-DoF) parts are included via
        their body-to-world transform rather than via `get_joint_qm`."""
        weighted = np.zeros(3, dtype=float)
        total_mass = 0.0
        for p in self.parts_move:
            m = self._mass_map.get(p, 0.0)
            weighted += m * self._get_any_part_pos(p)
            total_mass += m
        for p in self.parts_fix:
            m = self._mass_map.get(p, 0.0)
            weighted += m * self._get_any_part_pos(p)
            total_mass += m
        if total_mass <= 0.0:
            return weighted  # degenerate; preserve zero vector
        return weighted / total_mass

    def update_status(self):
        # COG-relative L2 metric: each moving part's distance to the CURRENT
        # mass-weighted COG, compared (absolute Δ) to its initial distance to
        # the t=0 COG. Rigid-body motion of the whole assembly preserves
        # ||pos − cog|| → Δ ≈ 0 → won't trip the fall/stability thresholds.
        # An outlier part drifting away from the bulk has its ||pos − cog||
        # grow, so |Δ| crosses the threshold while the bulk parts don't.
        cog_now = self._compute_cog()
        for part_move in self.parts_move:
            pos = self.get_part_pos(part_move)
            self.pos_his_map[part_move].append(pos)
            r_now = float(np.linalg.norm(pos - cog_now))
            delta_r = abs(r_now - self._r_initial[part_move])
            self.dist_his_map[part_move].append(delta_r)
        self.n_step += 1

    def check_disconnected_parts(self):
        parts_disconnected = []
        for part_move in self.parts_move:
            if self.G.degree(part_move) == 0:
                parts_disconnected.append(part_move)
        return parts_disconnected
    
    def check_fallen_parts(self, group=True):
        # Live contact graph + per-part fall reasons. Render the graph as a
        # PNG sidecar next to the GIF so the connectivity is actually
        # inspectable; the .log file gets a one-line per-check summary with
        # the fall reasons only (the human-readable bit). Both emitted only
        # when self.log_path is set (= caller supplied a record_path).
        live_G = self.derive_contact_graph() if self.log_path else None

        # Diagonal-scaled fall threshold. allow_gap sets
        # pos_far_threshold = inf, which survives the multiplication.
        far_abs = self.pos_far_threshold * self._initial_diag
        parts_fallen = []
        fall_reasons = {}
        for part_move in self.parts_move:
            if self.dist_his_map[part_move][-1] > far_abs: # check distance
                parts_fallen.append(part_move)
                fall_reasons[part_move] = (
                    f'distance({self.dist_his_map[part_move][-1]:.3f}'
                    f' > {far_abs:.3f}'
                    f' = {self.pos_far_threshold:g}×diag)'
                )
                continue
            for part_other in self.G.neighbors(part_move): # check connectivity
                in_contact = self.sim.body_in_contact(f'part{part_other}', f'part{part_move}', self.contact_eps)
                if not in_contact:
                    parts_fallen.append(part_move)
                    fall_reasons[part_move] = f'contact_lost:{part_other}'
                    break

        if self.log_path:
            self._log(f'[step={self.n_step}] fallen={fall_reasons}')
            self._render_contact_graph(live_G, parts_fallen, fall_reasons)
        if group and len(parts_fallen) > 1:
            parts_fallen_grouped = []
            G = self.derive_contact_graph()
            G = G.subgraph([x for x in parts_fallen])
            for G_sub in nx.connected_components(G):
                G_sub = list(G_sub)
                if len(G_sub) > 1:
                    G_sub_sorted = sorted(G_sub, key=lambda x: self.sim.get_body_mass(f'part{x}'), reverse=True)
                    parts_fallen_grouped.append(G_sub_sorted[0])
                else:
                    parts_fallen_grouped.append(G_sub[0])
            return parts_fallen_grouped
        else:
            return parts_fallen

    def check_stable_parts(self):
        # Diagonal-scaled settle band, matching the fall check.
        near_abs = self.pos_near_threshold * self._initial_diag
        parts_stable = []
        if self.n_step >= self.near_step:
            for part_move in self.parts_move:
                dist_interval = self.dist_his_map[part_move][self.n_step - self.near_step:self.n_step]
                if np.max(dist_interval) - np.min(dist_interval) < near_abs:
                    parts_stable.append(part_move)
        return parts_stable

    def plot_his(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, len(self.parts_move), figsize=(3 * len(self.parts_move), 3 * 3))
        axis_names = ['X', 'Y', 'Z']
        for i, part_id in enumerate(self.parts_move):
            for j in range(3):
                axes[j][i].plot(list(range(len(self.pos_his_map[part_id]))), np.array(self.pos_his_map[part_id])[:, j].round(3))
                if i == 0:
                    axes[j][i].set_ylabel(f'{axis_names[j]}')
                if j == 0:
                    axes[j][i].set_title(f'Part {part_id}')
            axes[3][i].plot(list(range(len(self.dist_his_map[part_id]))), np.array(self.dist_his_map[part_id]).round(3))
            axes[3][i].set_ylabel(f'Dist')
            axes[3][i].set_xlabel('Time step')
        plt.tight_layout()
        plt.show()


class MultiPartStabilityPlanner:

    max_step = MAX_STEP
    col_th = COL_TH_STABLE

    def __init__(self, asset_folder, assembly_dir, parts_fix, parts_move, pose=None, save_sdf=False, allow_gap=False, ignore_unstable=()):
        model_string = get_stability_sim_string(assembly_dir, parts_fix, parts_move, pose=pose, save_sdf=save_sdf, col_th=self.col_th)
        self.sim = redmax.Simulation(model_string, asset_folder)
        self.parts_fix = parts_fix.copy()
        self.parts_move = parts_move.copy()
        self.allow_gap = allow_gap
        # Parts whose falls should be tolerated rather than treated as failure.
        # Populated by SequencePlanner when settings.no_stable_pose_action ==
        # 'ignore_unstable' and the precheck observed inherently unstable parts.
        self.ignore_unstable = frozenset(ignore_unstable)

    def check_success(self, max_step=MAX_STEP, timeout=None, progress=False, progress_desc='stability', record_path=None):

        t_start = time()

        # Render gate: replay the gravity sim to a GIF when the caller
        # supplied a path AND the user opted in via settings.debug_stability.
        # The render uses self.sim.replay() which reads the already-populated
        # q_his / qdot_his — no physics is re-run.
        try:
            import settings as _user_settings
            _record_enabled = bool(getattr(_user_settings, 'debug_stability', False))
        except ImportError:
            _record_enabled = False
        do_record = _record_enabled and record_path is not None

        # initialize sim and stability checker
        self.sim.reset()
        checker = StabilityChecker(self.allow_gap)
        # Route the per-check diagnostic to a sidecar log next to the would-be
        # GIF (e.g. <step_label>_fix0.gif → <step_label>_fix0.log). Decoupled
        # from `do_record` so logs are always written when the caller supplied
        # a record_path, regardless of whether GIF rendering is enabled.
        if record_path is not None:
            checker.log_path = os.path.splitext(record_path)[0] + '.log'
        checker.update_sim(self.sim, self.parts_move, self.parts_fix)

        # check initial connectivity
        parts_disconnected = checker.check_disconnected_parts()
        if self.ignore_unstable:
            parts_disconnected = [p for p in parts_disconnected if p not in self.ignore_unstable]
        if len(parts_disconnected) > 0:
            return False, parts_disconnected

        # iterate until max step
        if DEBUG_SIM or progress:
            iterator = tqdm(range(max_step), desc=progress_desc, leave=False)
        else:
            iterator = range(max_step)
        for i in iterator:

            # simulate and update status
            self.sim.forward(1, verbose=DEBUG_SIM)
            checker.update_status()

            if (i + 1) % CHECK_FREQ == 0:
                # Check fallen parts. group=False returns ALL fallen parts (not
                # just one representative per contact-connected cluster), so the
                # greedy `get_stable_plan_1pose_serial` adds the whole cluster
                # to `parts_fix` in a single iteration instead of peeling it off
                # one rep per iteration (which inflates max_fix usage on
                # multi-part tumbles).
                parts_fall = checker.check_fallen_parts(group=False)
                if self.ignore_unstable:
                    parts_fall = [p for p in parts_fall if p not in self.ignore_unstable]
                if len(parts_fall) > 0:
                    if do_record:
                        print(f'[MultiPartStabilityPlanner] fallen parts: {parts_fall} at step {i}; replaying to {record_path}')
                        os.makedirs(os.path.dirname(record_path), exist_ok=True)
                        self.render(record_path=record_path)
                    return False, parts_fall

            if timeout is not None and time() - t_start > timeout:
                return False, None

        if do_record:
            print(f'[MultiPartStabilityPlanner] stable for {max_step} steps; replaying to {record_path}')
            os.makedirs(os.path.dirname(record_path), exist_ok=True)
            self.render(record_path=record_path)

        return True, None

    def render(self, record_path=None, make_video=False):
        SimRenderer.replay(self.sim, record=record_path is not None, record_path=record_path, make_video=make_video)


class MultiPartNoForceStabilityPlanner(MultiPartStabilityPlanner):

    def __init__(self, asset_folder, assembly_dir, parts, save_sdf=False, allow_gap=False):
        model_string = get_stability_sim_string(assembly_dir, [], parts, gravity=False, save_sdf=save_sdf, col_th=self.col_th)
        self.sim = redmax.Simulation(model_string, asset_folder)
        self.parts_fix = []
        self.parts_move = parts.copy()
        self.allow_gap = allow_gap

    def check_success(self, max_step=MAX_STEP, timeout=None):

        t_start = time()

        # initialize sim and stability checker
        self.sim.reset()
        checker = StabilityChecker(self.allow_gap)
        checker.update_sim(self.sim, self.parts_move, self.parts_fix)

        # check initial connectivity
        parts_fall = checker.check_disconnected_parts()

        # iterate until max step
        iterator = tqdm(range(max_step)) if DEBUG_SIM else range(max_step)
        for i in iterator:

            # simulate and update status
            self.sim.forward(1, verbose=DEBUG_SIM)
            checker.update_status()

            if (i + 1) % CHECK_FREQ == 0:
                # check fallen parts
                parts_fall_i = checker.check_fallen_parts()
                for part_fall in parts_fall_i:
                    if part_fall not in parts_fall:
                        parts_fall.append(part_fall)

            if timeout is not None and time() - t_start > timeout:
                return False, nx.Graph()

        # get result graph
        for part_fall in parts_fall:
            self.parts_move.remove(part_fall)
        checker.update_sim(self.sim, self.parts_move, self.parts_fix)

        success = (len(parts_fall) == 0)
        return success, checker.G


class MultiPartAdaptiveStabilityPlanner(MultiPartStabilityPlanner):

    def __init__(self, asset_folder, assembly_dir, parts_fix, parts_move, pose=None, save_sdf=False, allow_gap=False):
        self.asset_folder = asset_folder
        self.assembly_dir = assembly_dir
        self.parts_fix = parts_fix.copy()
        self.parts_move = parts_move.copy()
        self.pose = pose
        self.save_sdf = save_sdf
        self.allow_gap = allow_gap

    def check_success(self, max_step=MAX_STEP, timeout=None):

        t_start = time()

        # initialize sim and stability checker
        model_string = get_stability_sim_string(self.assembly_dir, self.parts_fix, self.parts_move, gravity=True, 
            save_sdf=self.save_sdf, pose=self.pose, col_th=self.col_th)
        self.sim = redmax.Simulation(model_string, self.asset_folder)
        self.sim.reset()
        checker = StabilityChecker(self.allow_gap)
        checker.update_sim(self.sim, self.parts_move, self.parts_fix)

        parts_stable_all = []

        # check initial connectivity
        parts_disconnected = checker.check_disconnected_parts()
        if len(parts_disconnected) > 0:
            return False, parts_disconnected, parts_stable_all

        # iterate until max step
        iterator = tqdm(range(max_step)) if DEBUG_SIM else range(max_step)
        for i in iterator:

            # simulate and update status
            self.sim.forward(1, verbose=DEBUG_SIM)
            checker.update_status()

            if (i + 1) % CHECK_FREQ == 0:
                # check fallen parts
                parts_fall = checker.check_fallen_parts()
                if len(parts_fall) > 0:
                    return False, parts_fall, parts_stable_all

                # check stable parts
                parts_stable = checker.check_stable_parts()
                if len(parts_stable) > 0:
                    parts_stable_all.extend(parts_stable)

                    # fix stable parts
                    self.parts_fix.extend(parts_stable)
                    for part_stable in parts_stable:
                        self.parts_move.remove(part_stable)
                    
                    # re-initialize simulation
                    mat_dict = self.sim.get_body_E0j_map()
                    mat_dict = {key.replace('part', ''): val for key, val in mat_dict.items()}
                    q_map = self.sim.get_q_map()
                    qdot_map = self.sim.get_qdot_map()

                    model_string = get_stability_sim_string(self.assembly_dir, self.parts_fix, self.parts_move, gravity=True, 
                        save_sdf=self.save_sdf, pose=self.pose, mat_dict=mat_dict, col_th=self.col_th)
                    self.sim = redmax.Simulation(model_string, self.asset_folder)
                    self.sim.reset()
                    self.sim.set_q_map(q_map)
                    self.sim.set_qdot_map(qdot_map)

                    # self.render()

                    # update checker with new sim
                    checker.update_sim(self.sim, self.parts_move, self.parts_fix)

            if timeout is not None and time() - t_start > timeout:
                return False, None, None # NOTE

        return True, None, parts_stable_all


def plot_stability_curve(pos_list_map):
    fig, axes = plt.subplots(3, len(pos_list_map), figsize=(3 * len(pos_list_map), 2 * 3))
    axis_names = ['X', 'Y', 'Z']
    for i, part_id in enumerate(pos_list_map.keys()):
        for j in range(3):
            axes[j][i].plot(list(range(len(pos_list_map[part_id]))), np.array(pos_list_map[part_id])[:, j].round(3))
            if i == 0:
                axes[j][i].set_ylabel(f'{axis_names[j]}')
            if j == 0:
                axes[j][i].set_title(f'Part {part_id}')
            if j == 2:
                axes[j][i].set_xlabel('Time step')
    plt.tight_layout()
    plt.show()


def _render_pose_debug_pyvista(assembly_dir, parts, poses, save_dir, size=(720, 540)):
    '''
    Off-screen pyvista screenshot per candidate stable pose. Each part is
    colored distinctly, a light ground plane is drawn at z=0, and the camera
    is set to iso so you can visually confirm the resting orientation.
    Saves <save_dir>/pose_XX.png and returns the list of paths.
    '''
    import pyvista as pv
    import matplotlib.cm as cm

    os.makedirs(save_dir, exist_ok=True)
    assembly = load_assembly(assembly_dir)
    part_meshes = [(p, assembly[p]['mesh']) for p in parts if p in assembly]

    out_paths = []
    for i, pose in enumerate(poses):
        plotter = pv.Plotter(off_screen=True, window_size=size)
        cmap = cm.tab20(np.linspace(0, 1, max(len(part_meshes), 1)))

        xy_min = np.full(2, np.inf)
        xy_max = np.full(2, -np.inf)
        for j, (part_id, m) in enumerate(part_meshes):
            m_posed = m.copy()
            m_posed.apply_transform(pose)
            color = tuple(float(c) for c in cmap[j % len(cmap)][:3])
            plotter.add_mesh(pv.wrap(m_posed), color=color, show_edges=False, opacity=1.0)
            xy_min = np.minimum(xy_min, m_posed.vertices[:, :2].min(axis=0))
            xy_max = np.maximum(xy_max, m_posed.vertices[:, :2].max(axis=0))

        size_xy = float(max((xy_max - xy_min).max() * 1.5, 1.0))
        center_xy = ((xy_min + xy_max) / 2).tolist()
        ground = pv.Plane(center=(center_xy[0], center_xy[1], 0.0),
                          direction=(0, 0, 1), i_size=size_xy, j_size=size_xy)
        plotter.add_mesh(ground, color='lightgray', opacity=0.3, show_edges=True)

        plotter.add_axes()
        plotter.camera_position = 'iso'
        plotter.add_text(f'pose {i + 1}/{len(poses)}', position='upper_left', font_size=10)

        out = os.path.join(save_dir, f'pose_{i:02d}.png')
        plotter.screenshot(out)
        plotter.close()
        out_paths.append(out)

    return out_paths


def _check_pose_standalone(asset_folder, assembly_dir, parts, pose,
                           save_sdf, allow_gap, max_step, timeout, pose_idx,
                           record_path=None):
    '''
    Picklable worker for parallel `find_stable_initial_poses`. Builds a fresh
    MultiPartStabilityPlanner (with parts_fix=[]) and runs the gravity sim.
    Returns {'pose_idx', 'success', 'fallen', '_dt_build', '_dt_run'}.

    record_path: when settings.debug_stability is True, the per-pose gravity
    sim is replayed to this GIF after the forward() loop. The replay uses
    the sim's already-populated q_his/qdot_his — no re-simulation.
    '''
    t_build = time()
    planner = MultiPartStabilityPlanner(
        asset_folder, assembly_dir,
        parts_fix=[], parts_move=list(parts),
        pose=pose, save_sdf=save_sdf, allow_gap=allow_gap,
    )
    dt_build = time() - t_build

    t_run = time()
    success, fallen = planner.check_success(
        max_step=max_step, timeout=timeout, progress=False,
        record_path=record_path,
    )
    dt_run = time() - t_run

    return {
        'pose_idx': pose_idx,
        'success': bool(success),
        'fallen': fallen,
        '_dt_build': dt_build,
        '_dt_run': dt_run,
    }


def find_stable_initial_poses(asset_folder, assembly_dir, parts, max_poses=5,
                              save_sdf=False, allow_gap=False,
                              max_step=MAX_STEP, timeout=None, progress=True,
                              num_proc=1, first_only=True, debug_dir=None,
                              stability_debug_dir=None):
    '''
    Pre-flight check for the full assembly. Compute trimesh stable poses for
    the combined mesh and keep only those for which the assembly stands
    self-supported under gravity (no parts need fixing).

    Returns (valid_poses, observed_fallen, per_pose) where
      - valid_poses is the subset of `get_stable_poses` outputs (in original
        order) that pass MultiPartStabilityPlanner with parts_fix=[] and
        parts_move=parts.
      - observed_fallen is a frozenset of part IDs that were observed falling
        in at least one candidate pose check (union across all poses).
      - per_pose is a list of dicts, one per attempted candidate pose:
            {'pose_idx': int, 'pose': 4x4 ndarray, 'success': bool, 'fallen': list[str]}
        Caller uses this to render one diagnostic image per pose (e.g.
        precheck_unstable_<i>.png) instead of a single union image.

    When `progress` is True (default), prints per-pose timing breakdowns
    (mesh, candidate enumeration, sim build, sim run) and shows a tqdm bar
    over the inner stability-check forward steps (serial mode only).

    When `num_proc > 1` the candidate poses are simulated in parallel via
    `utils.parallel.parallel_execute`. If `first_only=True` (default) the
    pool terminates remaining workers as soon as one stable pose is found.
    '''
    from plan_sequence.stable_pose import get_combined_mesh, get_stable_poses

    t0 = time()
    if progress:
        print(f'[find_stable_initial_poses] start: {len(parts)} parts, max_poses={max_poses}, max_step={max_step}, num_proc={num_proc}, first_only={first_only}')

    t_mesh = time()
    mesh = get_combined_mesh(assembly_dir, parts)
    dt_mesh = time() - t_mesh

    t_poses = time()
    # prob_th=1.0 disables the cumulative-probability early-exit; the only cap
    # is `max_num=max_poses`, so the caller's pose budget is actually honored.
    candidate_poses = get_stable_poses(mesh, prob_th=1.0, max_num=max_poses)
    dt_poses = time() - t_poses

    if progress:
        print(f'[find_stable_initial_poses] mesh: {dt_mesh:.2f}s  '
              f'trimesh stable poses ({len(candidate_poses)}): {dt_poses:.2f}s')

    if not candidate_poses:
        if progress:
            print(f'[find_stable_initial_poses] done in {time() - t0:.2f}s  kept 0/0 pose(s)')
        return [], frozenset()

    if debug_dir is not None:
        try:
            paths = _render_pose_debug_pyvista(assembly_dir, parts, candidate_poses, debug_dir)
            if progress:
                print(f'[find_stable_initial_poses] debug renders saved: {paths}')
        except Exception as e:
            print(f'[find_stable_initial_poses] WARNING: debug render failed: {e}')

    # Pre-compute per-pose stability-replay paths if the caller asked for
    # them. The render gate inside check_success is settings.debug_stability,
    # so producing the paths here is harmless when the flag is off.
    def _stability_record_path(i):
        if stability_debug_dir is None:
            return None
        return os.path.join(stability_debug_dir, f'pose_{i:02d}.gif')

    if num_proc > 1:
        from utils.parallel import parallel_execute

        worker_args = [
            (asset_folder, assembly_dir, list(parts), pose,
             save_sdf, allow_gap, max_step, timeout, i,
             _stability_record_path(i))
            for i, pose in enumerate(candidate_poses)
        ]

        def _terminate(result):
            return first_only and result['success']

        valid_idx = []
        observed_fallen = set()
        per_pose_map = {}
        for result in parallel_execute(
            _check_pose_standalone, worker_args, num_proc,
            show_progress=progress, desc='initial stable pose check',
            terminate_func=_terminate,
        ):
            i = result['pose_idx']
            if progress:
                outcome = 'STABLE' if result['success'] else f"unstable (fallen={result['fallen']})"
                print(f"[find_stable_initial_poses] pose {i + 1}/{len(candidate_poses)}: "
                      f"build {result['_dt_build']:.2f}s  run {result['_dt_run']:.2f}s  → {outcome}")
            per_pose_map[i] = {
                'pose_idx': i,
                'pose': candidate_poses[i],
                'success': bool(result['success']),
                'fallen': list(result['fallen']) if result['fallen'] else [],
            }
            if result['success']:
                valid_idx.append(i)
            elif result['fallen']:
                observed_fallen.update(result['fallen'])

        valid_poses = [candidate_poses[i] for i in sorted(valid_idx)]
        per_pose = [per_pose_map[i] for i in sorted(per_pose_map.keys())]
        if progress:
            print(f'[find_stable_initial_poses] done in {time() - t0:.2f}s  '
                  f'kept {len(valid_poses)}/{len(candidate_poses)} pose(s)  '
                  f'observed_fallen={sorted(observed_fallen)}')
        return valid_poses, frozenset(observed_fallen), per_pose

    # Serial path: keeps the inner per-step tqdm bar for live progress.
    valid_poses = []
    observed_fallen = set()
    per_pose = []
    for i, pose in enumerate(candidate_poses):
        t_build = time()
        planner = MultiPartStabilityPlanner(
            asset_folder, assembly_dir,
            parts_fix=[], parts_move=list(parts),
            pose=pose, save_sdf=save_sdf, allow_gap=allow_gap,
        )
        dt_build = time() - t_build

        if progress:
            print(f'[find_stable_initial_poses] pose {i + 1}/{len(candidate_poses)}: '
                  f'sim build {dt_build:.2f}s, running {max_step} steps...')

        t_run = time()
        success, fallen = planner.check_success(
            max_step=max_step, timeout=timeout,
            progress=progress, progress_desc=f'pose {i + 1}/{len(candidate_poses)}',
            record_path=_stability_record_path(i),
        )
        dt_run = time() - t_run

        if progress:
            outcome = 'STABLE' if success else f'unstable (fallen={fallen})'
            print(f'[find_stable_initial_poses] pose {i + 1}/{len(candidate_poses)}: '
                  f'run {dt_run:.2f}s  → {outcome}')

        per_pose.append({
            'pose_idx': i,
            'pose': pose,
            'success': bool(success),
            'fallen': list(fallen) if fallen else [],
        })

        if success:
            valid_poses.append(pose)
            if first_only:
                break
        elif fallen:
            observed_fallen.update(fallen)

    if progress:
        print(f'[find_stable_initial_poses] done in {time() - t0:.2f}s  '
              f'kept {len(valid_poses)}/{len(candidate_poses)} pose(s)  '
              f'observed_fallen={sorted(observed_fallen)}')
    return valid_poses, frozenset(observed_fallen), per_pose


def get_contact_graph(asset_folder, assembly_dir, parts=None, contact_eps=CONTACT_EPS, save_sdf=False):
    '''
    Get contact graph for assembly
    '''
    if parts is None: parts = load_part_ids(assembly_dir)

    model_string = get_contact_sim_string(assembly_dir, parts, save_sdf=save_sdf)
    sim = redmax.Simulation(model_string, asset_folder)

    G = nx.Graph()
    for i in range(len(parts)):
        G.add_node(parts[i])
        for j in range(i + 1, len(parts)):
            in_contact = sim.body_in_contact(f'part{parts[i]}', f'part{parts[j]}', contact_eps)
            if in_contact:
                G.add_edge(parts[i], parts[j])
    return G


def get_distance_all_bodies(asset_folder, assembly_dir, parts=None, save_sdf=False):
    if parts is None: parts = load_part_ids(assembly_dir)

    model_string = get_contact_sim_string(assembly_dir, parts, save_sdf=save_sdf)
    sim = redmax.Simulation(model_string, asset_folder)

    distance = {}
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            dist_ij = sim.get_body_distance(f'part{parts[i]}', f'part{parts[j]}')
            dist_ji = sim.get_body_distance(f'part{parts[j]}', f'part{parts[i]}')
            distance[(parts[i], parts[j])] = min(dist_ij, dist_ji)

    return distance


def get_body_mass(asset_folder, assembly_dir, parts=None, save_sdf=False):
    if parts is None: parts = load_part_ids(assembly_dir)

    model_string = get_contact_sim_string(assembly_dir, parts, save_sdf=save_sdf)
    sim = redmax.Simulation(model_string, asset_folder)

    mass_dict = {}
    for part in parts:
        mass_dict[part] = sim.get_body_mass(f'part{part}')

    return mass_dict


_VERIFY_DIRECTIONS = [
    np.array([1, 0, 0]), np.array([-1, 0, 0]),
    np.array([0, 1, 0]), np.array([0, -1, 0]),
    np.array([0, 0, 1]), np.array([0, 0, -1]),
]


def verify_separation(asset_folder, assembly_dir, parts_S, parts_R,
                      save_sdf=False, max_time=30, force_mag=FORCE_MAG):
    '''
    Verify that the union of parts_R can be physically separated from the union
    of parts_S. The two part lists are fused into a single combined mesh each
    (in their world-frame final pose), written to a temp dir, and then run
    through MultiPartPathPlanner.check_success along each of the 6 world-frame
    axis-aligned directions. Returns True iff any direction yields full
    disassembly.

    Pose is identity; gravity / stability are not considered.
    '''
    import shutil
    import tempfile

    from plan_sequence.stable_pose import get_combined_mesh

    if not parts_S or not parts_R:
        return False

    tmp_dir = tempfile.mkdtemp(prefix='verify_sep_')
    try:
        mesh_S = get_combined_mesh(assembly_dir, list(parts_S))
        mesh_R = get_combined_mesh(assembly_dir, list(parts_R))
        mesh_S.export(os.path.join(tmp_dir, 'S.obj'))
        mesh_R.export(os.path.join(tmp_dir, 'R.obj'))

        planner = MultiPartPathPlanner(
            asset_folder=asset_folder,
            assembly_dir=tmp_dir,
            parts_fix=['S'],
            part_move='R',
            pose=None,
            force_mag=force_mag,
            save_sdf=save_sdf,
        )
        planner.max_time = max_time

        for direction in _VERIFY_DIRECTIONS:
            if planner.check_success(direction):
                return True
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _verify_standalone(asset_folder, assembly_dir, parts_S, parts_R,
                       save_sdf, max_time, force_mag):
    '''
    Multiprocessing worker for verify_separation. Catches and silences
    exceptions so a failing partition just returns verified=False.
    '''
    try:
        ok = verify_separation(
            asset_folder=asset_folder,
            assembly_dir=assembly_dir,
            parts_S=parts_S,
            parts_R=parts_R,
            save_sdf=save_sdf,
            max_time=max_time,
            force_mag=force_mag,
        )
    except Exception:
        import traceback
        traceback.print_exc()
        ok = False
    return {'verified': ok}


# Project root used to import the top-level tool_analyzer module from this nested ASAPx
# location. tool_analyzer lives at <repo>/tool_analyzer.py.
_ASSEMBLY_EVAL_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
)


def check_tool(asset_folder, assembly_dir, parts_fix, part_move, tools,
               asset_folder_bfs=None, output_dir=None, show=False, verbose=False,
               diagnostics=None, failure_record_dir=None):
    '''
    Generic tool-feasibility check for a sub-assembly during sequence planning.

    Given the set of parts currently in the assembly (`parts_fix + [part_move]`) and a
    list of candidate tools, attempt to find any tool that can be geometrically applied
    to `part_move` and whose tool / tool+part union can be path-planned out. No VLM/LLM
    calls are made — tools are tried in order and the first feasible one is returned.

    Args:
        asset_folder: redmax asset root (only used to align the planner with the rest
            of the call site; not directly consumed by the tool pipeline).
        assembly_dir: directory containing the active parts' .obj files (matched by id).
        parts_fix: list of part IDs that must remain in place during removal.
        part_move: ID of the part the tool needs to remove.
        tools: list of `Tool` instances (already scaled to the assembly).
        asset_folder_bfs: redmax asset folder used by the BFSPlanner invoked inside the
            pipeline. Defaults to <repo>/ATA/assets.
        output_dir, show, verbose: forwarded.

    Returns:
        dict {'tool_id', 'tool_mesh', 'inverted'} on success, or None when no tool works.
    '''
    if _ASSEMBLY_EVAL_ROOT not in sys.path:
        sys.path.insert(0, _ASSEMBLY_EVAL_ROOT)
    from tool_analyzer import ToolAnalyzer

    if asset_folder_bfs is None:
        asset_folder_bfs = os.path.join(_ASSEMBLY_EVAL_ROOT, 'ATA', 'assets')

    parts_active = list(parts_fix) + [part_move]
    return ToolAnalyzer.check_tool_pipeline(
        asset_folder=asset_folder_bfs,
        assembly_dir=assembly_dir,
        parts=parts_active,
        part_move=part_move,
        tools=tools,
        output_dir=output_dir,
        show=show,
        verbose=verbose,
        diagnostics=diagnostics,
        failure_record_dir=failure_record_dir,
    )
