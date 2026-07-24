"""Demographic-parity post-processor from Xian, Yin, and Zhao (ICML 2023).

Vendored from https://github.com/uiuctml/fair-classification at tag
``icml.23`` (commit ff83c13c3c17de95ac7a29c0889727665014a08a).
The implementation is Algorithm 2 of "Fair and Optimal Classification via
Post-Processing".  Only formatting and solver diagnostics were added.
"""

from collections import defaultdict
from itertools import chain

import cvxpy as cp
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class PostProcessorDP(BaseEstimator):
    """Post-process class-probability scores for demographic parity."""

    def fit(
        self,
        scores,
        groups,
        alpha=0.0,
        group_weight=None,
        sample_weight=None,
        q_by_group=None,
        tol=1e-8,
        mip_solver="AUTO",
        qp_solver="AUTO",
    ):
        """Fit the empirical Fair-transport map.

        ``alpha`` is the requested maximum pairwise demographic-parity gap.
        Group labels must be zero-indexed integers.
        """
        scores, groups = check_X_y(scores, groups)
        groups = np.asarray(groups, dtype=int)
        if np.any(groups < 0):
            raise ValueError("groups must be zero-indexed non-negative integers.")
        if sample_weight is not None:
            _, r = check_X_y(scores, sample_weight)
        else:
            r = None

        self.n_classes_ = scores.shape[-1]
        self.n_groups_ = int(1 + np.max(groups))
        if not np.array_equal(np.unique(groups), np.arange(self.n_groups_)):
            raise ValueError("Every group code from 0 to n_groups-1 must be present.")
        self.alpha_ = float(alpha)
        if not np.isfinite(self.alpha_) or self.alpha_ < 0:
            raise ValueError("alpha must be finite and non-negative.")

        if group_weight is None:
            w = np.bincount(groups, minlength=self.n_groups_) / len(groups)
        else:
            w = np.asarray(group_weight, dtype=float)
        self.w_ = w

        scores_by_group = [scores[groups == a] for a in range(self.n_groups_)]
        r_by_group = []
        for a in range(self.n_groups_):
            if r is not None:
                this_r = np.asarray(r[groups == a], dtype=float).copy()
                this_r *= len(this_r) / this_r.sum()
                r_by_group.append(this_r)
            else:
                r_by_group.append(np.ones((groups == a).sum()))
        total_r_max = max(len(group_r) for group_r in r_by_group)

        installed = set(cp.installed_solvers())
        if mip_solver == "AUTO":
            mip_solver = next(
                (name for name in ("CBC", "CLARABEL", "SCIPY", "SCS") if name in installed),
                None,
            )
        if qp_solver == "AUTO":
            qp_solver = next(
                (name for name in ("OSQP", "CLARABEL", "SCS") if name in installed),
                None,
            )
        if mip_solver not in installed:
            raise RuntimeError(
                f"Fair-transport requires CVXPY solver {mip_solver}; "
                f"installed solvers are {sorted(installed)}."
            )
        if qp_solver not in installed:
            raise RuntimeError(
                f"Fair-transport requires CVXPY solver {qp_solver}; "
                f"installed solvers are {sorted(installed)}."
            )

        problem = self.linprog_dp_(
            scores_by_group,
            alpha=self.alpha_,
            w=w,
            r_by_group=r_by_group,
            q_by_group=q_by_group,
        )
        lp_options = {"integerTolerance": tol} if mip_solver == "CBC" else {}
        problem.solve(solver=mip_solver, **lp_options)
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(f"Fair-transport LP failed with status {problem.status}.")

        self.score_ = problem.value / total_r_max
        self.q_by_group_ = problem.var_dict["q"].value
        self.gamma_by_group_ = [
            problem.var_dict[f"gamma_{a}"].value for a in range(self.n_groups_)
        ]

        psi_by_group = []
        map_status = []
        for a in range(self.n_groups_):
            try:
                map_problem = self.quadprog_find_point_(
                    scores_by_group[a], self.gamma_by_group_[a], tol=tol
                )
                map_problem.solve(solver=qp_solver)
                z = map_problem.var_dict["z"].value
                if (
                    map_problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
                    or z is None
                ):
                    raise cp.error.SolverError
                psi_by_group.append(
                    [0.0]
                    + [2 * (z[0] - z[j]) for j in range(1, self.n_classes_)]
                )
                map_status.append(map_problem.status)
            except cp.error.SolverError:
                map_problem = self.linprog_score_transform_(
                    scores_by_group[a], self.gamma_by_group_[a], tol=tol
                )
                map_problem.solve(solver=mip_solver, **lp_options)
                if map_problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                    raise RuntimeError(
                        "Fair-transport fallback map LP failed with status "
                        f"{map_problem.status}."
                    )
                psi_by_group.append(2 * map_problem.var_dict["bias"].value)
                map_status.append(map_problem.status)

        self.psi_by_group_ = np.stack(psi_by_group)
        self.lp_status_ = problem.status
        self.map_status_by_group_ = tuple(map_status)
        self.mip_solver_ = mip_solver
        self.qp_solver_ = qp_solver
        return self

    def linprog_dp_(
        self, scores_by_group, alpha, w=None, r_by_group=None, q_by_group=None
    ):
        """Implement line 3 of Algorithm 2 in the ICML 2023 paper."""
        alpha = cp.Parameter(value=alpha, name="alpha")
        gamma_by_group = [
            cp.Variable(scores_by_group[a].shape, name=f"gamma_{a}")
            for a in range(self.n_groups_)
        ]
        barycenter = cp.Variable(self.n_classes_, name="barycenter")
        q = cp.Variable((self.n_groups_, self.n_classes_), name="q")
        slack = cp.Variable((self.n_groups_, self.n_classes_), name="slack")

        total_r = np.array([r.sum() for r in r_by_group])
        cost_by_group = [
            (1 - scores_by_group[a])
            * w[a]
            * total_r.max()
            / total_r[a]
            / w.sum()
            for a in range(self.n_groups_)
        ]
        cost = sum(
            cp.sum(cp.multiply(gamma_by_group[a], cost_by_group[a]))
            for a in range(self.n_groups_)
        )

        constraints = []
        for a in range(self.n_groups_):
            constraints.append(cp.sum(gamma_by_group[a], axis=1) == r_by_group[a])
            constraints.append(
                cp.sum(gamma_by_group[a], axis=0) == q[a] * total_r[a]
            )

        if q_by_group is None:
            for a in range(self.n_groups_):
                constraints.append(-slack[a] <= q[a] - barycenter)
                constraints.append(q[a] - barycenter <= slack[a])
        else:
            q_by_group = cp.Parameter(
                (self.n_groups_, self.n_classes_),
                value=q_by_group,
                name="q_by_group",
            )
            for a in range(self.n_groups_):
                constraints.append(-slack[a] <= q[a] - q_by_group[a])
                constraints.append(q[a] - q_by_group[a] <= slack[a])

        constraints.append(slack <= alpha / 2)
        constraints.extend(gamma >= 0 for gamma in gamma_by_group)
        constraints.extend([q >= 0, barycenter >= 0, slack >= 0])
        return cp.Problem(cp.Minimize(cost), constraints)

    def quadprog_find_point_(self, scores, gamma, tol=1e-8):
        """Extract the score-shift map (lines 6-9 of Algorithm 2)."""
        z = cp.Variable(self.n_classes_, name="z")
        boundaries = np.zeros((self.n_classes_, self.n_classes_))
        for i in range(self.n_classes_):
            for j in chain(range(i), range(i + 1, self.n_classes_)):
                idx = gamma[:, i] > tol
                boundaries[i, j] = np.max(
                    scores[idx, j] - scores[idx, i] + 1, initial=0
                )
        boundaries -= np.clip(boundaries + boundaries.T - 2, 0, None) / 2
        gaps = np.clip(2 - boundaries.T - boundaries, 1e-2, None)

        cost = 0
        constraints = []
        for i in range(self.n_classes_):
            for j in chain(range(i), range(i + 1, self.n_classes_)):
                constraints.append(z[j] - z[i] >= boundaries[i, j] - 1)
                cost += (
                    cp.square(z[j] - z[i] - (boundaries[i, j] - 1))
                    / gaps[i, j] ** 2
                )
        return cp.Problem(cp.Minimize(cost), constraints)

    def linprog_score_transform_(self, scores, gamma, tol=1e-8):
        """Fallback extraction of the score-shift map."""
        diffs = defaultdict(dict)
        for score, coupling in zip(scores, gamma):
            candidates = set(np.where(coupling > tol)[0])
            for i in candidates:
                for j in chain(range(i), range(i + 1, self.n_classes_)):
                    difference = score[j] - score[i]
                    diffs[i][j] = max(difference, diffs[i].get(j, difference))

        bias = cp.Variable(self.n_classes_, name="bias")
        slack = cp.Variable((self.n_classes_, self.n_classes_), name="slack")
        constraints = [slack >= 0]
        for i in diffs:
            for j in diffs[i]:
                constraints.append(bias[i] + slack[i][j] >= bias[j] + diffs[i][j])
        return cp.Problem(cp.Minimize(cp.sum(slack)), constraints)

    def predict(self, scores, groups):
        """Return zero-indexed fair class assignments."""
        scores = check_array(scores)
        groups = check_array(groups, ensure_2d=False)
        scores, groups = check_X_y(scores, groups)
        groups = np.asarray(groups, dtype=int)
        check_is_fitted(self, "psi_by_group_")
        if np.any(groups < 0) or np.any(groups >= self.n_groups_):
            raise ValueError("Prediction contains an unseen group code.")
        return np.argmin(-2 * scores - self.psi_by_group_[groups], axis=1)
