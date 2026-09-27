"""Recursive subassembly plan: prefix -> unified split -> S -> R, where S and R
are themselves planned the same way.

This runs alongside the flat sequence logic rather than replacing it. The flat
minimum-cost sequence is still computed first (see BaseSequenceOptimizer.
optimize_scored); this module then asks whether that same tree also contains a
sequence whose ORDER respects a nested subassembly decomposition, and only
adopts it when one exists.

The decomposition is found with the DivideOptimizer's obstruction graph, which
is built once on the full tree and then searched under a `restrict_parts`
universe per block (see DivideOptimizer.find_locally_free_subassemblies). No
re-planning is involved: every step in the resulting order is still an edge of
the original tree, so the renderer, per-step poses, tool decisions and the arm
pipeline all keep working on it unchanged.

Block schema (JSON-serialisable, nested):

    {'kind': 'split',
     'path': ['S'],                 # '' at the root, then S/R per level
     'parts': [...],                # every part in this block
     'prefix': [...],               # removed before the split, in order
     'split': {'S': [...], 'R': [...], 'score': float},
     'S': <block>, 'R': <block>}

    {'kind': 'leaf',
     'path': [...], 'parts': [...], 'sequence': [...]}

A block's `prefix` is what makes "diminished" splits usable: the DivideOptimizer
propagates cuts forward along a disassembly sequence, so the best cut for a
block is often one that only exists after a few parts have come off. Those parts
become the block's prefix.
"""
import numpy as np


DEFAULT_MAX_DEPTH = 2
DEFAULT_MIN_BLOCK_PARTS = 4
# A side smaller than this is not worth calling a subassembly.
MIN_SIDE_PARTS = 2


def _settings():
    try:
        import settings as user_settings
        return user_settings
    except ImportError:
        return None


def _find_block_split(div, parts, sequence, threshold, timeout, top_k,
                      num_proc, verbose, sweep_states=True,
                      min_block_parts=DEFAULT_MIN_BLOCK_PARTS, max_states=None,
                      known_verified=None):
    """Best physically-verified cut of `parts`, or None.

    `parts` is the block's universe; `sequence` is a representative disassembly
    order for it, used to walk the block's prefix states (and to propagate cuts
    forward at the first one). Returns (S, R, score) with S and R disjoint
    subsets of `parts` (their union may be a strict subset — the remainder is
    the block's prefix, which is exactly the set that has to come off before
    the cut is free).

    `known_verified` is an (S, R, score) already physically verified for this
    block — seq_plan's top-level cut, at the root. Candidates that cannot beat
    its score are dropped before verification, so when the search turns up
    nothing better the physics cost is zero and the known cut is returned
    unchanged. That is the common case: a cut needing a prefix has smaller
    sides, so the common-basis score ranks it below a cut that was already free
    in the whole block."""
    # The DivideOptimizer is shared across every block (its obstruction graph is
    # what makes a restricted search cheap), and both calls below overwrite its
    # result attributes. Snapshot them so the caller's own full-assembly results
    # -- which seq_plan has already read but other callers may not have -- are
    # still there afterwards.
    saved_free = div.locally_free
    saved_verified = getattr(div, 'verified_locally_free', None)
    try:
        if sweep_states:
            # Searches every prefix state of the block, not just the whole
            # block, so a side that is locked until some part comes off is
            # still found. Pools the initial state's propagated cuts too, so
            # this is strictly additive.
            found = div.sweep_sequence_states(
                list(sequence),
                restrict_parts=set(parts),
                timeout=timeout,
                min_block_parts=min_block_parts,
                max_states=max_states,
                verbose=verbose,
            )
        else:
            found = div.find_locally_free_subassemblies(
                timeout=timeout,
                propagate=True,
                disassembly_sequence=list(sequence),
                restrict_parts=set(parts),
                verbose=verbose,
            )
        if not found:
            return None

        floor = threshold
        if known_verified is not None:
            # Only a strictly better cut is worth a physics call.
            floor = max(floor, float(known_verified[2]))
        viable = [
            (S, R, score) for (S, R, score) in found
            if score > floor or (known_verified is None and score >= floor)
            if len(S) >= MIN_SIDE_PARTS and len(R) >= MIN_SIDE_PARTS
        ]
        if not viable:
            if known_verified is not None:
                if verbose:
                    print('[split_plan] nothing outranks the already-verified '
                          'cut; keeping it (no extra verification)')
                return (set(known_verified[0]), set(known_verified[1]),
                        float(known_verified[2]))
            return None

        # verify_locally_free reads self.locally_free, so narrow it to the
        # viable candidates before verifying — otherwise the top-k budget gets
        # spent on cuts we have already rejected on score.
        div.locally_free = viable
        verified = div.verify_locally_free(
            top_k=top_k, num_proc=num_proc, verbose=verbose,
        )
        if not verified:
            if known_verified is not None:
                return (set(known_verified[0]), set(known_verified[1]),
                        float(known_verified[2]))
            return None
        S, R, score = verified[0]
        if known_verified is not None and float(score) <= float(known_verified[2]):
            return (set(known_verified[0]), set(known_verified[1]),
                    float(known_verified[2]))
        return set(S), set(R), float(score)
    finally:
        div.locally_free = saved_free
        div.verified_locally_free = saved_verified


def build_split_plan(tree, asset_folder, assembly_dir, sequence,
                     divide_optimizer=None, root_split=None,
                     threshold=0.1, max_depth=DEFAULT_MAX_DEPTH,
                     min_parts=DEFAULT_MIN_BLOCK_PARTS, timeout=100,
                     top_k=10, num_proc=1, sweep_states=True,
                     sweep_max_states=None, debug=0):
    """Build the recursive subassembly plan for `sequence`.

    `divide_optimizer` should be a DivideOptimizer whose obstruction graph has
    already been built on the full tree; one is built here when omitted.
    `root_split` is the already-verified top-level (S, R) from seq_plan. With
    the sweep off it is taken as the root cut directly. With the sweep on the
    root block is still swept — it is the block where a cut locked inside the
    full assembly matters most — but `root_split` is carried in as the cut to
    beat, so the sweep only spends physics on candidates that outrank it.

    `sweep_states` searches every prefix state of each block rather than only
    the whole block (see DivideOptimizer.sweep_sequence_states).

    Returns the root block, or None when no split was found at all (in which
    case the caller keeps the flat sequence).
    """
    div = divide_optimizer
    if div is None:
        from .divide import DivideOptimizer
        div = DivideOptimizer(tree, asset_folder=asset_folder,
                              assembly_dir=assembly_dir)
        if div.build_obstruction_graph() is None:
            return None
    elif div.obstruction_graph is None:
        if div.build_obstruction_graph() is None:
            return None

    root_parts = set(sequence)
    root_node = getattr(div, 'root', None)
    if root_node:
        root_parts |= set(root_node)

    verbose = debug > 1

    def leaf(parts, seq_local, path):
        return {
            'kind': 'leaf',
            'path': list(path),
            'parts': sorted(parts),
            'sequence': list(seq_local),
        }

    def plan_block(parts, seq_local, depth, path):
        if depth >= max_depth or len(parts) < min_parts:
            return leaf(parts, seq_local, path)

        split = None
        known = None
        if depth == 0 and root_split is not None:
            S = {p for p in root_split.get('S', []) if p in parts}
            R = {p for p in root_split.get('R', []) if p in parts}
            if (len(S) >= MIN_SIDE_PARTS and len(R) >= MIN_SIDE_PARTS
                    and not (S & R)):
                known = (S, R, float(root_split.get('score', 0.0)))
                if not sweep_states:
                    # Already verified by seq_plan; no need to re-scan.
                    split = known

        if split is None:
            split = _find_block_split(
                div, parts, seq_local, threshold, timeout, top_k, num_proc,
                verbose, sweep_states=sweep_states, min_block_parts=min_parts,
                max_states=sweep_max_states, known_verified=known,
            )
        if split is None:
            return leaf(parts, seq_local, path)

        S, R, score = split
        # Order by the representative sequence, but derive membership from the
        # block's part set: the tree's sequence is one part short (the leaf node
        # keeps the last part), and that part still has to land in a block or it
        # would vanish from the plan.
        order = {p: i for i, p in enumerate(seq_local)}
        tail = len(seq_local)

        def in_seq_order(subset):
            return sorted(subset, key=lambda q: order.get(q, tail))

        prefix = in_seq_order(p for p in parts if p not in S and p not in R)
        s_seq = in_seq_order(S)
        r_seq = in_seq_order(R)

        if debug > 0:
            label = '.'.join(path) or 'root'
            print(f'[split_plan] {label}: |prefix|={len(prefix)} '
                  f'|S|={len(S)} |R|={len(R)} score={score:.4f}')

        return {
            'kind': 'split',
            'path': list(path),
            'parts': sorted(parts),
            'prefix': prefix,
            'split': {'S': sorted(S), 'R': sorted(R), 'score': score},
            'S': plan_block(S, s_seq, depth + 1, path + ('S',)),
            'R': plan_block(R, r_seq, depth + 1, path + ('R',)),
        }

    plan = plan_block(root_parts, list(sequence), 0, ())
    if plan['kind'] != 'split':
        return None
    return plan


def retarget_split_plan(plan, sequence):
    """Re-derive every block's `prefix` / leaf `sequence` ordering from
    `sequence`.

    The plan is first built against the flat sequence; once the constrained
    sequence has been chosen the stored plan has to describe THAT order, or the
    persisted stats would disagree with the rendered steps."""
    pos = {p: i for i, p in enumerate(sequence)}
    last = len(sequence)

    def key(p):
        return pos.get(p, last)

    def walk(block):
        if block['kind'] != 'split':
            return {**block, 'sequence': sorted(block['parts'], key=key)}
        return {
            **block,
            'prefix': sorted(block['prefix'], key=key),
            'S': walk(block['S']),
            'R': walk(block['R']),
        }

    return walk(plan)


def split_order_constraint(plan):
    """Build ``is_allowed(sequence) -> bool`` enforcing the plan's block order.

    Per split block: every prefix part must come off before any part of S or R,
    and every S part before every R part. Applied recursively, that is exactly
    "prefix -> unified split -> S -> R" at every level.

    A part absent from the sequence is the one the tree leaves over at the leaf
    node; it is treated as removed last, which correctly forces it into the
    deepest R block."""
    pairs = []

    def collect(block):
        if block['kind'] != 'split':
            return
        S = set(block['split']['S'])
        R = set(block['split']['R'])
        prefix = set(block['prefix'])
        if prefix:
            pairs.append((prefix, S | R))
        pairs.append((S, R))
        collect(block['S'])
        collect(block['R'])

    collect(plan)

    def is_allowed(sequence):
        pos = {p: i for i, p in enumerate(sequence)}
        last = len(sequence)
        for earlier, later in pairs:
            if not earlier or not later:
                continue
            if max(pos.get(p, last) for p in earlier) >= min(
                pos.get(p, last) for p in later
            ):
                return False
        return True

    return is_allowed


def flatten_split_plan(plan, sequence):
    """Flatten the plan into the ordered step list the pipeline consumes.

    Entries are, in disassembly order:
        {'kind': 'remove', 'part': id, 'group': ['S', 'R'], 'depth': n}
        {'kind': 'join', 'group': [...], 'depth': n, 'S': [...], 'R': [...],
         'score': float}

    `group` is the block path: [] for parts that belong to no subassembly (a
    prefix at the root), ['S'] for the top-level S block, ['S', 'R'] for the R
    half of S, and so on. The manual reads it to colour and label pages; the
    'join' entries are the unified S/R separation steps, which are NOT tree
    edges and so have no per-part render of their own."""
    pos = {p: i for i, p in enumerate(sequence)}
    last = len(sequence)

    def key(p):
        return pos.get(p, last)

    def walk(block, path):
        depth = len(path)
        if block['kind'] != 'split':
            return [
                {'kind': 'remove', 'part': p, 'group': list(path), 'depth': depth}
                for p in sorted(block['parts'], key=key)
            ]
        out = [
            {'kind': 'remove', 'part': p, 'group': list(path), 'depth': depth}
            for p in sorted(block['prefix'], key=key)
        ]
        out.append({
            'kind': 'join',
            'group': list(path),
            'depth': depth,
            'S': list(block['split']['S']),
            'R': list(block['split']['R']),
            'score': float(block['split']['score']),
        })
        out += walk(block['S'], path + ('S',))
        out += walk(block['R'], path + ('R',))
        return out

    return walk(plan, ())


def derive_split_sequence(plan, sequence):
    """The plan's own disassembly order over EVERY part, as a flat list.

    This is the fallback for when the tree contains no path that respects the
    plan (the common case: a block ordering is a narrow slice of the orderings a
    budget-limited search explores). Parts keep their relative order from
    `sequence` within each block, so each removal still happens in the relative
    context the planner validated -- and it happens with strictly FEWER parts
    present, since the other side has been lifted away by then.

    Unlike a constrained sequence this is not a root-to-leaf path of the tree,
    so it must not be handed to the renderer, which resolves each step against a
    tree edge. It drives the manual's narrative order only."""
    return [entry['part'] for entry in flatten_split_plan(plan, sequence)
            if entry['kind'] == 'remove']


def iter_split_blocks(plan):
    """Yield every split block of the plan, outermost first."""
    if plan is None or plan.get('kind') != 'split':
        return
    yield plan
    yield from iter_split_blocks(plan['S'])
    yield from iter_split_blocks(plan['R'])


def describe_split_plan(plan, indent=0):
    """Multi-line human-readable rendering of the plan."""
    if plan is None:
        return '(no subassembly plan)'
    pad = '  ' * indent
    label = '.'.join(plan['path']) or 'root'
    if plan['kind'] != 'split':
        return f'{pad}{label}: leaf {plan["sequence"]}'
    lines = [
        f'{pad}{label}: split score={plan["split"]["score"]:.4f}',
        f'{pad}  prefix: {plan["prefix"]}',
        f'{pad}  S ({len(plan["split"]["S"])}): {plan["split"]["S"]}',
        describe_split_plan(plan['S'], indent + 2),
        f'{pad}  R ({len(plan["split"]["R"])}): {plan["split"]["R"]}',
        describe_split_plan(plan['R'], indent + 2),
    ]
    return '\n'.join(lines)


def split_direction(asset_folder, assembly_dir, parts_S, parts_R,
                    max_time=30, save_sdf=False):
    """World-axis direction along which unified R separates from unified S, or
    None when no axis works.

    Same combine-and-scan approach as verify_separation; the direction is what
    the renderer and the manual's join page need in order to draw the mating
    motion."""
    import os
    import shutil
    import tempfile

    from plan_sequence.physics_planner import (FORCE_MAG, MultiPartPathPlanner,
                                               _VERIFY_DIRECTIONS)
    from plan_sequence.stable_pose import get_combined_mesh

    if not parts_S or not parts_R:
        return None

    tmp_dir = tempfile.mkdtemp(prefix='split_dir_')
    try:
        get_combined_mesh(assembly_dir, list(parts_S)).export(
            os.path.join(tmp_dir, 'S.obj'))
        get_combined_mesh(assembly_dir, list(parts_R)).export(
            os.path.join(tmp_dir, 'R.obj'))
        planner = MultiPartPathPlanner(
            asset_folder=asset_folder, assembly_dir=tmp_dir,
            parts_fix=['S'], part_move='R', pose=None,
            force_mag=FORCE_MAG, save_sdf=save_sdf,
        )
        planner.max_time = max_time
        for direction in _VERIFY_DIRECTIONS:
            if planner.check_success(direction):
                return [float(x) for x in np.asarray(direction, dtype=float)]
        return None
    except Exception as e:
        print(f'[split_plan] split_direction failed: {e}')
        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
