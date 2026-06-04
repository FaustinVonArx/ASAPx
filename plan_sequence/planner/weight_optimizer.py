"""Online preference learning for the linear cost function's weights.

A `WeightOptimizer` learns the weight vector `w` of a linear cost
`cost(node) = w · phi(node)` from pairwise human preferences, using a
Passive-Aggressive (PA-I) perceptron update with active querying.

This module is intentionally DECOUPLED from the beam search / disassembly tree:
it operates purely on feature vectors (`numpy` arrays) and knows nothing about
how those features are produced. That keeps it reusable and unit-testable
without pulling in any ASAPx / simulation dependencies (only numpy + stdlib).

Cost convention
---------------
The planner MINIMISES cost (lower = better), so the preferred candidate is the
one with the *smallest* `w · phi`. The classic PA preference update is written
for a max-score utility; here it is sign-mirrored so that after an update the
oracle's pick `n_star` becomes cheaper than the model's pick `n_hat`. Set
`lower_is_better=False` to recover the standard max-score behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_TINY = 1e-12


class WeightOptimizer:
    def __init__(self, n_features, feature_names=None, epsilon=0.1,
                 save_path=None, log_path=None, lower_is_better=True):
        self.n_features = int(n_features)
        self.feature_names = (list(feature_names) if feature_names is not None
                              else [f"f{i}" for i in range(self.n_features)])
        assert len(self.feature_names) == self.n_features
        self.epsilon = float(epsilon)
        self.save_path = Path(save_path) if save_path is not None else None
        self.log_path = Path(log_path) if log_path is not None else None
        self.lower_is_better = bool(lower_is_better)
        self.n_updates = 0
        self._step = 0

        # Uniform init, L2-normalized; overwritten by load() if a file exists.
        self.weights = self._normalize(np.full(self.n_features, 1.0 / self.n_features))
        if self.save_path is not None and self.save_path.exists():
            self.load()

    # ------------------------------------------------------------------
    # helpers

    @staticmethod
    def _normalize(w):
        w = np.asarray(w, dtype=float).reshape(-1)
        norm = float(np.linalg.norm(w))
        if norm < _TINY:
            # Degenerate — fall back to uniform so ranking stays well-defined.
            w = np.full(w.shape, 1.0 / max(len(w), 1))
            norm = float(np.linalg.norm(w))
        return w / norm

    def _as_matrix(self, candidate_features):
        """Coerce a list of phi vectors / 2-D array into an (m, n_features) array."""
        Phi = np.asarray(candidate_features, dtype=float)
        if Phi.ndim == 1:
            Phi = Phi.reshape(1, -1)
        return Phi

    def costs(self, candidate_features):
        """Cost (w · phi) for each candidate row."""
        return self._as_matrix(candidate_features) @ self.weights

    # ------------------------------------------------------------------
    # public API

    def predict(self, candidate_features):
        """Index of the model's pick: cheapest (argmin cost) when
        lower_is_better, else highest-scoring (argmax)."""
        c = self.costs(candidate_features)
        return int(np.argmin(c) if self.lower_is_better else np.argmax(c))

    def should_query(self, candidate_features):
        """Active-learning gate: query only when the model is uncertain, i.e.
        the gap between the best and second-best candidate is below `epsilon`.
        Fewer than 2 candidates → never query (no choice to make)."""
        c = self.costs(candidate_features)
        if c.size < 2:
            return False
        c_sorted = np.sort(c)  # ascending
        if self.lower_is_better:
            margin = float(c_sorted[1] - c_sorted[0])      # cheapest two
        else:
            margin = float(c_sorted[-1] - c_sorted[-2])    # highest two
        return margin < self.epsilon

    def update(self, phi_hat, phi_star):
        """Passive-Aggressive (PA-I) update so the oracle's pick `phi_star`
        ends up preferred over the model's pick `phi_hat`. Returns the new
        weight vector. No-op (eta=0) when the two feature vectors coincide
        (model already agreed with the oracle) — this also guards the
        division by ‖diff‖²."""
        phi_hat = np.asarray(phi_hat, dtype=float).reshape(-1)
        phi_star = np.asarray(phi_star, dtype=float).reshape(-1)
        w_before = self.weights.copy()

        diff = phi_star - phi_hat
        denom = float(diff @ diff)
        if denom < _TINY:
            # n_hat == n_star: nothing to learn.
            self._log_update(phi_hat, phi_star, w_before, self.weights,
                             loss=0.0, eta=0.0, margin=0.0)
            self._step += 1
            return self.weights

        if self.lower_is_better:
            # margin = how much cheaper star is than hat; want >= 1.
            margin = float(self.weights @ phi_hat - self.weights @ phi_star)
            loss = max(0.0, 1.0 - margin)
            eta = loss / denom
            # w <- w - eta * (phi_star - phi_hat) raises star's relative cheapness.
            new_w = self.weights - eta * diff
        else:
            # Max-score convention: margin = score(star) - score(hat).
            margin = float(self.weights @ phi_star - self.weights @ phi_hat)
            loss = max(0.0, 1.0 - margin)
            eta = loss / denom
            new_w = self.weights + eta * diff

        if loss > 0.0:
            self.weights = self._normalize(new_w)
            self.n_updates += 1
            if self.save_path is not None:
                self.save()

        self._log_update(phi_hat, phi_star, w_before, self.weights,
                         loss=loss, eta=eta, margin=margin)
        self._step += 1
        return self.weights

    # ------------------------------------------------------------------
    # persistence

    def as_dict(self):
        """Weights keyed by feature name (handy for planners that expect a dict)."""
        return {name: float(v) for name, v in zip(self.feature_names, self.weights)}

    def save(self):
        if self.save_path is None:
            return
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "weights": [float(v) for v in self.weights],
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            "epsilon": self.epsilon,
            "n_updates": self.n_updates,
        }
        self.save_path.write_text(json.dumps(payload, indent=2))

    def load(self):
        if self.save_path is None or not self.save_path.exists():
            return
        data = json.loads(self.save_path.read_text())
        w = data.get("weights")
        if w is not None and len(w) == self.n_features:
            self.weights = self._normalize(w)
        if data.get("feature_names"):
            self.feature_names = list(data["feature_names"])
        self.n_updates = int(data.get("n_updates", self.n_updates))

    # ------------------------------------------------------------------
    # audit log

    def _log_update(self, phi_hat, phi_star, w_before, w_after, loss, eta, margin):
        if self.log_path is None:
            return
        record = {
            "step": self._step,
            "phi_hat": [float(v) for v in phi_hat],
            "phi_star": [float(v) for v in phi_star],
            "w_before": [float(v) for v in w_before],
            "w_after": [float(v) for v in w_after],
            "loss": float(loss),
            "eta": float(eta),
            "margin": float(margin),
        }
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            pass
