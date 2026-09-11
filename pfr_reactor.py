import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
from scipy.integrate import solve_ivp

from power_law import PowerLawReaction

R_Gas = 8.314462618 #J/(mol K)

class PFRIsothermal:
    """
    Isothermal Plug Flow Reactor (PFR) with variable volumetric flow rate
    (variable volume/variable density gas-phase system).
    """
    def __init__(self,reactions: List[PowerLawReaction],
                 F0 :Dict[str,float],
                 T:float,
                 P:float):
        if not reactions:
            raise ValueError("At least one reaction must be provided.")
        self.reactions = reactions
        self.T = T
        self.P = P
        #Collect every species that appears in any of the reactions
        species_set = set()
        for rxn in reactions:
            species_set.update(rxn.species)
        self.species = sorted(species_set)
        #Make sure F0 has an entry for every species in the reactions
        self.F0 = {sp:F0.get(sp,0.0) for sp in self.species}
        self.results = None  # populated after solve() or solve_for_conversion()

    # Alternate Flow Initialization
    @classmethod
    def from_composition(cls, reactions: List[PowerLawReaction],
                         F_total: float,
                         z0: Dict[str,float],
                         T:float,
                         P:float)-> "PFRIsothermal":
        cls._validate_mole_fractions(z0)
        if F_total <= 0:
            raise ValueError("Total molar flow rate must be positive.")
        F0 = {sp: F_total*frac for sp, frac in z0.items()}
        return cls(reactions=reactions, F0=F0, T=T, P=P)

    @staticmethod
    def _validate_mole_fractions(z0: Dict[str,float])->None:
        if any(frac<0 for frac in z0.values()):
            raise ValueError("Mole fractions must be non-negative.")
        total = sum(z0.values())
        if abs(total-1.0) > 1e-6:
            raise ValueError(f"Mole fractions must sum to 1.0. Current sum: {total}")

    def _total_molar_flow(self,F:Dict[str,float])->float:
        return sum(F.values())

    def _volumetric_flow(self,F:Dict[str,float])->float:
        F_total = self._total_molar_flow(F)
        return F_total * R_Gas * self.T / self.P

    def _concentrations(self,F:Dict[str,float])->Dict[str,float]:
        Q = self._volumetric_flow(F)
        if Q<=0:
            raise ValueError(
                "Volumetric flow rate become zero or negative"
                "(check F0 and reaction stoichiometry)"
            )
        return {sp: F.get(sp,0.0)/Q for sp in self.species}

    # Mole Balances
    def _derivatives(self,F:Dict[str,float])->Dict[str,float]:
        C = self._concentrations(F)
        dF = {sp:0.0 for sp in self.species}
        for reaction in self.reactions:
            species_rates = reaction.species_rates(T=self.T,concentrations=C)
            for species, rate in species_rates.items():
                dF[species] += rate
        return dF

    #ODE Solver
    def _ode_rhs(self,V: float, F_vector:np.ndarray)->np.ndarray:
        F = {sp: F_vector[i] for i, sp in enumerate(self.species)}
        dF = self._derivatives(F)
        return np.array([dF[sp] for sp in self.species])

    def _resolve_basis_species(self, basis_species: str) -> str:
        if basis_species is not None:
            if basis_species not in self.species:
                raise ValueError(
                    f"'{basis_species}' is not among the species "
                    f"in this reactor: {self.species}"
                )
            return basis_species

        for reaction in self.reactions:
            for species, nu in reaction.stoichiometry.items():
                if nu < 0:
                    return species

        raise ValueError(
            "Could not auto-detect a reactant (all stoichiometric "
            "coefficients are >= 0). Please specify basis_species explicitly."
        )

    #Rating Mode
    def solve(self, V_max: float, basis_species: str = None, n_points : int =200)->Dict:
        if V_max <=0:
            raise ValueError("Maximum reactor volume must be positive.")
        basis_species = self._resolve_basis_species(basis_species)
        F0_vector = np.array([self.F0[sp] for sp in self.species])
        V_eval = np.linspace(0, V_max, n_points)
        solution = solve_ivp(
            fun = self._ode_rhs,
            t_span = (0, V_max),
            y0 = F0_vector,
            t_eval = V_eval,
            method = 'RK45',
        )
        if not solution.success:
            raise RuntimeError("ODE solver failed: " + solution.message)
        self.results = self._assemble_results(V_eval, solution.y, basis_species)
        return self.results

    #Sizing Mode
    def solve_for_conversion(self, X_target:float, basis_species:str=None, V_search_max:float=100,n_points:int=200)->Dict:
        if not (0.0 < X_target < 1.0):
            raise ValueError("Target conversion must be between 0 and 1.")
        basis_species = self._resolve_basis_species(basis_species)
        F0_basis = self.F0[basis_species]
        basis_index = self.species.index(basis_species)

        def conversion_event(V, F_vector):
            X_current = (F0_basis - F_vector[basis_index]) / F0_basis
            return X_current - X_target
        conversion_event.terminal = True
        conversion_event.direction = 1

        F0_vector = np.array([self.F0[sp] for sp in self.species])
        solution = solve_ivp(
            fun = self._ode_rhs,
            t_span = (0, V_search_max),
            y0 = F0_vector,
            events = conversion_event,
            method = 'RK45',
            dense_output = True,
        )
        if not solution.success:
            raise RuntimeError("ODE solver failed: " + solution.message)
        if len(solution.t_events[0]) == 0:
            raise RuntimeError(
                    f"Target conversion X={X_target} was not reached within "
                    f"V_search_max={V_search_max}. Try increasing V_search_max."
                )
        V_required = solution.t_events[0][0]
        V_eval = np.linspace(0, V_required, n_points)
        F_matrix = solution.sol(V_eval)
        self.results = self._assemble_results(V_eval, F_matrix, basis_species)
        self.results['V_required'] = V_required
        return self.results

    #Post-Processing
    def _assemble_results(self, V_eval: np.ndarray,
                           F_matrix: np.ndarray,
                           basis_species: str) -> Dict:
        """
        Turn a raw F_i(V) matrix (from either solve() or
        solve_for_conversion()) into the same results dictionary
        format: F, C, Q, X profiles. Kept as one shared method so the
        physics (Q from ideal gas law, C from F/Q) is never duplicated
        between the two solving modes.
        """
        F_profiles = {
            sp: F_matrix[i, :] for i, sp in enumerate(self.species)
        }

        Q_profile = np.array([
            self._volumetric_flow(
                {sp: F_profiles[sp][k] for sp in self.species}
            )
            for k in range(len(V_eval))
        ])

        C_profiles = {sp: np.zeros(len(V_eval)) for sp in self.species}
        for k in range(len(V_eval)):
            F_k = {sp: F_profiles[sp][k] for sp in self.species}
            C_k = self._concentrations(F_k)
            for sp in self.species:
                C_profiles[sp][k] = C_k[sp]

        F0_basis = self.F0[basis_species]
        X_profile = (F0_basis - F_profiles[basis_species]) / F0_basis

        return {
            "V": V_eval,
            "F": F_profiles,
            "C": C_profiles,
            "Q": Q_profile,
            "X": X_profile,
            "basis_species": basis_species,
        }

    # ============================================================
    # PLOTTING
    # ============================================================

    def plot(self, show_species: bool = False, dark_mode: bool = True):
        if self.results is None:
            raise RuntimeError(
                "No results to plot yet. Call solve() or "
                "solve_for_conversion() first."
            )

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        V = self.results["V"]
        X = self.results["X"]
        basis = self.results["basis_species"]

        template = "plotly_dark" if dark_mode else "plotly_white"
        paper_bg = "#12161C" if dark_mode else "white"
        plot_bg = "#12161C" if dark_mode else "white"
        line_color = "#00B4D8"

        if not show_species:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=V, y=X,
                mode="lines",
                name=f"X ({basis})",
                line=dict(width=3, color=line_color),
            ))
            fig.update_layout(
                title="Conversion vs Reactor Volume",
                xaxis_title="Volume V (m3)",
                yaxis_title=f"Conversion X ({basis})",
                yaxis_range=[0, 1],
                template=template,
                paper_bgcolor=paper_bg,
                plot_bgcolor=plot_bg,
                font=dict(family="Roboto Mono, Consolas, monospace", size=12),
            )
            return fig

        # Two-panel version: conversion on top, all species F_i below
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=(
                "Conversion vs Reactor Volume",
                "Molar Flow Rate Profiles",
            ),
            vertical_spacing=0.12,
        )

        fig.add_trace(
            go.Scatter(x=V, y=X, mode="lines", name=f"X ({basis})",
                       line=dict(width=3, color=line_color)),
            row=1, col=1,
        )

        for sp in self.species:
            fig.add_trace(
                go.Scatter(x=V, y=self.results["F"][sp], mode="lines",
                           name=f"F_{sp}"),
                row=2, col=1,
            )

        fig.update_yaxes(title_text=f"X ({basis})", range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text="F_i (mol/s)", row=2, col=1)
        fig.update_xaxes(title_text="Volume V (m3)", row=2, col=1)
        fig.update_layout(
            template=template,
            paper_bgcolor=paper_bg,
            plot_bgcolor=plot_bg,
            height=700,
            font=dict(family="Roboto Mono, Consolas, monospace", size=12),
        )

        return fig
