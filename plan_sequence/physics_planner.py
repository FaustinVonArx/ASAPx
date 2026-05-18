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
MAX_TIME = 60
CONTACT_EPS = 1e-1
POS_FAR_THRESHOLD = 0.2
POS_NEAR_THRESHOLD = 0.05
NEAR_STEP = 100
MAX_STEP = 1000
CHECK_FREQ = 20
COL_TH_PATH = 0.01
COL_TH_STABLE = 0.00
MIN_SEP = 0.5
DEBUG_SIM = False

DOF_NUM_STEPS = 50
DOF_RETURN_EPS = 1e-3


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

    def compute_dof(self, num_steps=DOF_NUM_STEPS, return_eps=DOF_RETURN_EPS):
        '''
        Probe each of 6 assembly-local axis-aligned directions to estimate
        translational DoFs. Local directions are rotated by self.pose[:3,:3]
        before being applied, so the returned 6-vector is pose-invariant.
        For each direction, apply a constant force for up to num_steps single
        forward steps, stopping early when qdotdot drops below 0.01 * force_mag.
        DoF[i] = 1 if the trajectory never returned within return_eps of an
        earlier visited position; else 0.
        Returns a length-6 numpy int array in local-frame order
        [+Z, -Z, +X, -X, +Y, -Y].
        '''
        t_start = time()
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
            init_state = self.get_state()
            positions = [init_state.q[:3].copy()]
            prev_qdot = init_state.qdot[:3].copy()
            for _ in range(num_steps):
                self.sim.forward(1, verbose=False)
                new_state = self.get_state()
                positions.append(new_state.q[:3].copy())
                cur_qdot = new_state.qdot[:3].copy()
                qdotdot = (cur_qdot - prev_qdot) / self.sim.options.h
                prev_qdot = cur_qdot
                if np.linalg.norm(qdotdot) < 0.01 * self.force_mag:
                    break
            final_pos = positions[-1]
            returned = any(
                np.linalg.norm(final_pos - prev_pos) < return_eps
                for prev_pos in positions[:-1]
            )
            dof[i] = 0 if returned else 1
        self.sim.reset()
        elapsed = time() - t_start
        print(f'[compute_dof] elapsed: {elapsed:.3f}s  dof: {dof.tolist()}')
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
        if allow_gap:
            self.pos_far_threshold = np.inf

    def get_part_pos(self, part):
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
        self.G = self.derive_contact_graph()
        if self.pos_his_map is None:
            self.pos_his_map = {part: [self.get_part_pos(part)] for part in self.parts_move}
        if self.dist_his_map is None:
            self.dist_his_map = {part: [0.0] for part in self.parts_move}

    def update_status(self):
        for part_move in self.parts_move:
            pos = self.get_part_pos(part_move)
            self.pos_his_map[part_move].append(pos)
            self.dist_his_map[part_move].append(np.linalg.norm(pos - self.pos_his_map[part_move][0]))
        self.n_step += 1

    def check_disconnected_parts(self):
        parts_disconnected = []
        for part_move in self.parts_move:
            if self.G.degree(part_move) == 0:
                parts_disconnected.append(part_move)
        return parts_disconnected
    
    def check_fallen_parts(self, group=True):
        parts_fallen = []
        for part_move in self.parts_move:
            if self.dist_his_map[part_move][-1] > self.pos_far_threshold: # check distance
                parts_fallen.append(part_move)
                continue
            for part_other in self.G.neighbors(part_move): # check connectivity
                in_contact = self.sim.body_in_contact(f'part{part_other}', f'part{part_move}', self.contact_eps)
                if not in_contact:
                    parts_fallen.append(part_move)
                    break
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
        parts_stable = []
        if self.n_step >= self.near_step:
            for part_move in self.parts_move:
                dist_interval = self.dist_his_map[part_move][self.n_step - self.near_step:self.n_step]
                if np.max(dist_interval) - np.min(dist_interval) < self.pos_near_threshold:
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

    def __init__(self, asset_folder, assembly_dir, parts_fix, parts_move, pose=None, save_sdf=False, allow_gap=False):
        model_string = get_stability_sim_string(assembly_dir, parts_fix, parts_move, pose=pose, save_sdf=save_sdf, col_th=self.col_th)
        self.sim = redmax.Simulation(model_string, asset_folder)
        self.parts_fix = parts_fix.copy()
        self.parts_move = parts_move.copy()
        self.allow_gap = allow_gap

    def check_success(self, max_step=MAX_STEP, timeout=None, progress=False, progress_desc='stability'):

        t_start = time()

        # initialize sim and stability checker
        self.sim.reset()
        checker = StabilityChecker(self.allow_gap)
        checker.update_sim(self.sim, self.parts_move, self.parts_fix)

        # check initial connectivity
        parts_disconnected = checker.check_disconnected_parts()
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
                # check fallen parts
                parts_fall = checker.check_fallen_parts()
                if len(parts_fall) > 0:
                    # checker.plot_his()
                    # self.render()
                    return False, parts_fall

            if timeout is not None and time() - t_start > timeout:
                return False, None

        # checker.plot_his()
        # self.render()

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


def find_stable_initial_poses(asset_folder, assembly_dir, parts, max_poses=3,
                              save_sdf=False, allow_gap=False,
                              max_step=MAX_STEP, timeout=None, progress=True):
    '''
    Pre-flight check for the full assembly. Compute trimesh stable poses for
    the combined mesh and keep only those for which the assembly stands
    self-supported under gravity (no parts need fixing).

    Returns the subset of `get_stable_poses` outputs (in original order) that
    pass MultiPartStabilityPlanner with parts_fix=[] and parts_move=parts.

    When `progress` is True (default), prints per-pose timing breakdowns
    (mesh, candidate enumeration, sim build, sim run) and shows a tqdm bar
    over the inner stability-check forward steps.
    '''
    from plan_sequence.stable_pose import get_combined_mesh, get_stable_poses

    t0 = time()
    if progress:
        print(f'[find_stable_initial_poses] start: {len(parts)} parts, max_poses={max_poses}, max_step={max_step}')

    t_mesh = time()
    mesh = get_combined_mesh(assembly_dir, parts)
    dt_mesh = time() - t_mesh

    t_poses = time()
    candidate_poses = get_stable_poses(mesh, max_num=max_poses)
    dt_poses = time() - t_poses

    if progress:
        print(f'[find_stable_initial_poses] mesh: {dt_mesh:.2f}s  '
              f'trimesh stable poses ({len(candidate_poses)}): {dt_poses:.2f}s')

    valid_poses = []
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
        )
        dt_run = time() - t_run

        if progress:
            outcome = 'STABLE' if success else f'unstable (fallen={fallen})'
            print(f'[find_stable_initial_poses] pose {i + 1}/{len(candidate_poses)}: '
                  f'run {dt_run:.2f}s  → {outcome}')

        if success:
            valid_poses.append(pose)

    if progress:
        print(f'[find_stable_initial_poses] done in {time() - t0:.2f}s  '
              f'kept {len(valid_poses)}/{len(candidate_poses)} pose(s)')
    return valid_poses


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
               asset_folder_bfs=None, output_dir=None, show=False, verbose=False):
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
    )
