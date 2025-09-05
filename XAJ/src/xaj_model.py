
"""
xaj_model.py — A practical, self-contained Python implementation of the Xinanjiang (XAJ) model.

Features
--------
- Saturation-excess runoff generation using the XAJ capacity curve (parameter B).
- Three-layer tension water accounting for evapotranspiration (upper, middle, lower).
- Simple runoff separation into surface, interflow, and baseflow.
- Two routing options: (a) 3 parallel linear reservoirs (default), (b) Muskingum (optional).
- Minimal, dependency-free optimizer (random search) for parameter calibration against NSE/RMSE/KGE.

Disclaimer
----------
The XAJ family has many variants. This implementation aims to be practical and readable rather than
a perfect reproduction of any single canonical variant. It follows widely used motifs:
- Capacity curve runoff generation (saturation excess).
- Layered evapotranspiration demand satisfaction.
- Component separation + simple routing.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, List
import numpy as np

# ---------------------
# Helper metrics
# ---------------------
def nse(sim: np.ndarray, obs: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    denom = np.sum((obs - np.mean(obs))**2)
    if denom <= 0:
        return -np.inf
    return 1.0 - np.sum((obs - sim)**2) / denom

def rmse(sim: np.ndarray, obs: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    return float(np.sqrt(np.mean((obs - sim)**2)))

def kge(sim: np.ndarray, obs: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    r = np.corrcoef(sim, obs)[0,1] if np.std(sim)>0 and np.std(obs)>0 else -1.0
    alpha = np.std(sim)/np.std(obs) if np.std(obs)>0 else 0.0
    beta  = np.mean(sim)/np.mean(obs) if np.mean(obs)>0 else 0.0
    return 1.0 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)

# ---------------------
# Parameters
# ---------------------
@dataclass
class XAJParameters:
    # Tension water capacity (total) & shape
    WM: float = 150.0      # mm, areal mean max tension water capacity
    B: float = 0.3         # shape parameter for capacity curve (typically 0-1)

    # Three-layer capacities (sum need not equal WM; these are "pools" for E)
    CU: float = 30.0       # upper layer capacity (mm)
    CM: float = 60.0       # middle layer capacity (mm)
    CL: float = 60.0       # lower layer capacity (mm)

    # Impervious fraction
    IMP: float = 0.02      # 0-1

    # Runoff separation coefficients (fractions of generated runoff going to each store)
    CS: float = 0.25       # surface
    CI: float = 0.35       # interflow
    # remainder goes to baseflow = 1 - CS - CI

    # Linear reservoir routing coefficients (per-timestep recession factors, 0-1)
    KS: float = 0.6        # surface reservoir outflow coefficient
    KI: float = 0.4        # interflow
    KG: float = 0.15       # baseflow

    # Optional Muskingum routing (used if use_muskingum=True)
    K_musk: float = 2.0    # storage time constant
    X_musk: float = 0.2    # weighting (0-0.5 typical)

    # Evapotranspiration reduction for middle/lower layers (0-1; 1 = full demand allowed)
    EM_red_m: float = 0.7
    EM_red_l: float = 0.4

    # Bounds for calibration (used by random_search)
    bounds: Optional[Dict[str, Tuple[float,float]]] = None

    def to_dict(self) -> Dict:
        return asdict(self)

# ---------------------
# Model state
# ---------------------
@dataclass
class XAJState:
    # Tension water pools for E
    WU: float = 10.0
    WMd: float = 30.0
    WL: float = 40.0

    # Areal mean tension water (for capacity curve)
    W: float = 80.0

    # Linear reservoirs storage for routing
    Ss: float = 0.0
    Si: float = 0.0
    Sg: float = 0.0

    # Muskingum routing memory
    Q_prev: float = 0.0
    I_prev: float = 0.0

# ---------------------
# Core model
# ---------------------
class XAJModel:
    def __init__(self, params: XAJParameters, state: Optional[XAJState]=None, use_muskingum: bool=False, dt_days: float=1.0):
        self.p = params
        self.s = state if state is not None else XAJState()
        self.use_muskingum = use_muskingum
        self.dt_days = dt_days

    # ----- Capacity curve helper -----
    def saturation_fraction(self, W: float) -> float:
        """
        Fraction of saturated area as a function of areal-mean tension water W.
        XAJ capacity curve (Zhao, 1992-style): F = 1 - (1 - W/WM)^(B+1)
        """
        W = np.clip(W, 0.0, self.p.WM)
        if self.p.WM <= 0:
            return 0.0
        return 1.0 - (1.0 - W/self.p.WM)**(self.p.B + 1.0)

    def runoff_generation(self, P: float, W: float) -> Tuple[float,float]:
        """
        Compute saturation-excess runoff (mm) and updated areal-mean tension water W (mm)
        using capacity curve before and after rainfall P (mm). Adds impervious runoff.
        """
        P = max(0.0, P)
        # Impervious direct runoff
        R_imp = self.p.IMP * P

        # Saturation-excess via capacity curve incremental area
        F1 = self.saturation_fraction(W)
        W2 = np.clip(W + P, 0.0, self.p.WM)
        F2 = self.saturation_fraction(W2)

        R_sat = (F2 - F1) * P  # incremental contributing area * rainfall
        # Update mean tension water (what remained)
        infiltrated = P - R_sat
        W_new = np.clip(W + infiltrated, 0.0, self.p.WM)

        return R_imp + R_sat, W_new

    # ----- Evapotranspiration from 3 layers -----
    def evapotranspiration(self, E0: float) -> Tuple[float,float,float]:
        """
        Distribute potential ET (E0) across 3 layers with simple demand-first logic:
        - Upper layer uses as much as available (WU -> 0 to CU).
        - Middle uses reduced demand: EM_red_m * remaining demand.
        - Lower uses reduced demand: EM_red_l * remaining demand.
        Returns (EU, EM, EL). Updates state storages accordingly.
        """
        E0 = max(0.0, E0)
        EU = min(self.s.WU, E0)
        self.s.WU -= EU
        rem = E0 - EU

        demand_m = self.p.EM_red_m * rem
        EM = min(self.s.WMd, demand_m)
        self.s.WMd -= EM
        rem2 = rem - EM

        demand_l = self.p.EM_red_l * rem2
        EL = min(self.s.WL, demand_l)
        self.s.WL -= EL

        return EU, EM, EL

    # ----- Separation & Routing -----
    def separate_runoff(self, R: float) -> Tuple[float,float,float]:
        CS = np.clip(self.p.CS, 0.0, 1.0)
        CI = np.clip(self.p.CI, 0.0, 1.0 - CS)
        CG = max(0.0, 1.0 - CS - CI)
        Rs = CS * R
        Ri = CI * R
        Rg = CG * R
        return Rs, Ri, Rg

    def route_linear_reservoirs(self, Rs: float, Ri: float, Rg: float) -> float:
        # Update storages
        self.s.Ss += Rs
        self.s.Si += Ri
        self.s.Sg += Rg
        # Outflows
        Qs = self.p.KS * self.s.Ss
        Qi = self.p.KI * self.s.Si
        Qg = self.p.KG * self.s.Sg
        # Update remaining storage
        self.s.Ss -= Qs
        self.s.Si -= Qi
        self.s.Sg -= Qg
        return Qs + Qi + Qg

    def route_muskingum(self, I: float) -> float:
        """ Single-reach Muskingum for total inflow I. """
        K = max(1e-6, self.p.K_musk)
        X = np.clip(self.p.X_musk, 0.0, 0.5)
        C0 = (-K*X + 0.5*self.dt_days)/(K - K*X + 0.5*self.dt_days)
        C1 = ( K*X + 0.5*self.dt_days)/(K - K*X + 0.5*self.dt_days)
        C2 = ( K - K*X - 0.5*self.dt_days)/(K - K*X + 0.5*self.dt_days)

        Q = C0*self.s.I_prev + C1*I + C2*self.s.Q_prev
        self.s.I_prev = I
        self.s.Q_prev = Q
        return Q

    # ----- One step -----
    def step(self, P: float, E0: float) -> Dict[str, float]:
        # 1) Evapotranspiration from pools (affects WU, WMd, WL)
        EU, EM, EL = self.evapotranspiration(E0)
        E = EU + EM + EL

        # 2) Add precipitation to tension water mean store with capacity curve runoff
        R_gen, W_new = self.runoff_generation(P, self.s.W)
        self.s.W = W_new

        # 3) Separate runoff components
        Rs, Ri, Rg = self.separate_runoff(R_gen)

        # 4) Routing
        if self.use_muskingum:
            Q = self.route_muskingum(Rs + Ri + Rg)
        else:
            Q = self.route_linear_reservoirs(Rs, Ri, Rg)

        return {
            "Q": Q, "E": E, "R": R_gen,
            "EU": EU, "EM": EM, "EL": EL,
            "Rs": Rs, "Ri": Ri, "Rg": Rg,
            "WU": self.s.WU, "WMd": self.s.WMd, "WL": self.s.WL, "W": self.s.W
        }

    # ----- Run series -----
    def run(self, P: np.ndarray, E0: np.ndarray) -> Dict[str, np.ndarray]:
        n = len(P)
        out_keys = ["Q","E","R","EU","EM","EL","Rs","Ri","Rg","WU","WMd","WL","W"]
        out = {k: np.zeros(n, dtype=float) for k in out_keys}
        for t in range(n):
            step = self.step(float(P[t]), float(E0[t]))
            for k in out_keys:
                out[k][t] = step[k]
        return out

# ---------------------
# Minimal random-search calibration
# ---------------------
def pso_calibration(
    P: np.ndarray,
    E0: np.ndarray,
    Qobs: np.ndarray,
    n_particles: int = 20,
    n_iter: int = 100,
    seed: int = 42,
    use_muskingum: bool = False,
    score: str = "NSE",
    param_bounds: Optional[Dict[str, Tuple[float,float]]] = None,
    init_params: Optional[XAJParameters] = None,
    w: float = 0.7,     # 惯性权重
    c1: float = 1.5,    # 个体学习因子
    c2: float = 1.5     # 群体学习因子
) -> Tuple[XAJParameters, float, np.ndarray]:
    """
    Calibrate parameters using Particle Swarm Optimization (PSO).
    Returns (best_params, best_score, best_simQ).
    """
    rng = np.random.default_rng(seed)
    params = init_params if init_params is not None else XAJParameters()

    # 默认参数范围
    if param_bounds is None:
        param_bounds = {
            "WM": (80, 500),
            "B": (0.01, 1.2),
            "CU": (10, 120),
            "CM": (20, 120),
            "CL": (20, 120),
            "IMP": (0.0, 0.15),
            "CS": (0.01, 0.6),
            "CI": (0.01, 0.6),
            "KS": (0.01, 0.9),
            "KI": (0.01, 0.7),
            "KG": (0.001, 0.4),
            "K_musk": (0.01, 10.0),
            "X_musk": (0.0, 0.5),
            "EM_red_m": (0.4, 1.0),
            "EM_red_l": (0.2, 0.8),
        }

    keys = list(param_bounds.keys())
    dim = len(keys)

    def score_fn(sim, obs):
        if score.upper() == "NSE":
            return nse(sim, obs)
        elif score.upper() == "KGE":
            return kge(sim, obs)
        elif score.upper() == "RMSE":
            return -rmse(sim, obs)  # minimize
        else:
            return nse(sim, obs)

    # 初始化粒子位置与速度
    lb = np.array([param_bounds[k][0] for k in keys])
    ub = np.array([param_bounds[k][1] for k in keys])
    X = rng.uniform(lb, ub, size=(n_particles, dim))
    V = rng.uniform(-abs(ub-lb), abs(ub-lb), size=(n_particles, dim))

    # 评估初始适应度
    def eval_particle(x):
        d = {k: float(v) for k,v in zip(keys, x)}
        if d["CS"] + d["CI"] > 0.95:  # 保证约束
            s = d["CS"] + d["CI"]
            d["CS"] *= 0.95/s
            d["CI"] *= 0.95/s
        model = XAJModel(XAJParameters(**d), XAJState(), use_muskingum=use_muskingum)
        out = model.run(P, E0)
        return score_fn(out["Q"], Qobs), out["Q"]

    pbest = X.copy()
    pbest_val = np.empty(n_particles)
    pbest_sim = [None]*n_particles
    for i in range(n_particles):
        val, sim = eval_particle(X[i])
        pbest_val[i] = val
        pbest_sim[i] = sim

    gbest_idx = np.argmax(pbest_val)
    gbest = pbest[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]
    gbest_sim = pbest_sim[gbest_idx]

    # 主循环
    for it in range(n_iter):
        for i in range(n_particles):
            r1, r2 = rng.random(dim), rng.random(dim)
            V[i] = w*V[i] + c1*r1*(pbest[i]-X[i]) + c2*r2*(gbest-X[i])
            X[i] = np.clip(X[i] + V[i], lb, ub)

            val, sim = eval_particle(X[i])
            if val > pbest_val[i]:
                pbest[i] = X[i].copy()
                pbest_val[i] = val
                pbest_sim[i] = sim
                if val > gbest_val:
                    gbest = X[i].copy()
                    gbest_val = val
                    gbest_sim = sim

        print(f"Iter {it+1}/{n_iter} - Best {score.upper()}: {gbest_val:.4f}")

    # 转换回参数对象
    best_dict = {k: float(v) for k,v in zip(keys, gbest)}
    best_params = XAJParameters(**best_dict)

    return best_params, float(gbest_val), np.asarray(gbest_sim)
