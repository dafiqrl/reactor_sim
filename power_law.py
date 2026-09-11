"""
kinetics.py

Power-law kinetics module for the modular reactor simulator project.

Design decisions
-----------------
1. STOICHIOMETRY vs RATE-LAW ORDER are kept as two SEPARATE dicts.
   - `stoichiometry[species]` = nu_i (signed: negative = reactant,
     positive = product). Used only to convert the reaction rate r into
     per-species generation/consumption rates: r_i = nu_i * r.
   - `order[species]` = n_i, the exponent that species actually carries
     in the rate law r = k * Prod(C_i ** n_i).
   For an ELEMENTARY reaction, n_i = |nu_i| for reactants and 0 for
   products, and this is filled in automatically if you don't specify
   `order`. For a NON-ELEMENTARY reaction (very common in Fogler /
   Levenspiel problems, e.g. rate depends on a product concentration, or
   on a fractional/pseudo order), just pass `order` explicitly and it can
   differ freely from `stoichiometry`.

2. TWO COMPONENT MODES:
   - ComponentMode.CUSTOM  -> species are plain symbolic labels
     ("A", "B", "C", "D", ...) for textbook problems, no thermo lookup.
   - ComponentMode.THERMO  -> species names are validated against a
     `ThermoModel` instance (see thermo_model.py / the `thermo` library),
     for real chemical systems where you'll later want MW, Cp, etc.
     Kinetics math is identical either way; THERMO mode just adds a
     validation + lookup convenience layer on top.

3. RATE CONSTANT: either
   - ConstantRateConstant(k_value)                -> k independent of T
   - ArrheniusRateConstant(Ea, A=...)              -> k(T) = A*exp(-Ea/RT)
   - ArrheniusRateConstant(Ea, T_ref=..., k_ref=...) -> A is back-solved
     from a reference point k_ref at T_ref (handy when a textbook gives
     you "k = 0.05 1/min at 300 K" instead of A directly).

4. UNITS are NOT auto-converted (no pint dependency by default) -- you
   are responsible for using one consistent set (SI is the default:
   concentration mol/m3, time s, Ea J/mol, R J/mol/K). `conc_unit` and
   `time_unit` are just cosmetic labels so `PowerLawReaction.k_units`
   can tell you what units k must be in for the reaction as configured,
   since that depends on the overall order and is easy to get wrong by
   hand.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Optional, Dict, List

try:
    from thermo_model import ThermoModel
except ImportError:
    ThermoModel = None


R_GAS = 8.314462618  # J/(mol K)


class ComponentMode(Enum):
    CUSTOM = "custom"
    THERMO = "thermo"


# ============================================================
# RATE CONSTANTS
# ============================================================

class RateConstant:
    """Base interface. Use ConstantRateConstant or ArrheniusRateConstant."""

    def value(self, T: float) -> float:
        raise NotImplementedError


@dataclass
class ConstantRateConstant(RateConstant):
    """
    Temperature-independent rate constant.

    k_value : the rate constant, in whatever units the reaction's overall
              order demands (see PowerLawReaction.k_units).
    """
    k_value: float

    def value(self, T: float = None) -> float:
        return self.k_value


@dataclass
class ArrheniusRateConstant(RateConstant):
    """
    k(T) = A * exp(-Ea / (R*T))

    Build it one of two ways:
      1) A directly known           -> ArrheniusRateConstant(Ea=..., A=...)
      2) A reference point known    -> ArrheniusRateConstant(
                                            Ea=..., T_ref=..., k_ref=...)
         A is back-calculated from  A = k_ref * exp(Ea / (R * T_ref))

    Ea : activation energy, J/mol
    R  : gas constant, J/(mol K) (override only if you insist on
         non-SI Ea/units elsewhere -- not recommended, keep SI)
    """
    Ea: float
    A: Optional[float] = None
    T_ref: Optional[float] = None
    k_ref: Optional[float] = None
    R: float = R_GAS

    def __post_init__(self):
        if self.A is None:
            if self.T_ref is None or self.k_ref is None:
                raise ValueError(
                    "Provide either A directly, or both T_ref and k_ref "
                    "so A can be back-calculated from "
                    "k_ref = A * exp(-Ea / (R * T_ref))."
                )
            self.A = self.k_ref * math.exp(self.Ea / (self.R * self.T_ref))

    def value(self, T: float) -> float:
        if T <= 0:
            raise ValueError("Temperature must be greater than 0 K.")
        return self.A * math.exp(-self.Ea / (self.R * T))


# ============================================================
# POWER LAW REACTION
# ============================================================

@dataclass
class PowerLawReaction:
    """
    A single power-law reaction:

        r = k(T) * Prod_i ( C_i ** n_i )

    Parameters
    ----------
    stoichiometry : dict {species: nu_i}
        Signed stoichiometric coefficients. Negative = reactant,
        positive = product. e.g. {"A": -1, "B": -2, "C": 1} for
        A + 2B -> C.

    rate_constant : ConstantRateConstant | ArrheniusRateConstant

    order : dict {species: n_i}, optional
        Rate-law exponents. If omitted (or partially omitted), missing
        species default to the ELEMENTARY assumption: n_i = |nu_i| for
        reactants, 0 for products. Pass this explicitly whenever the
        reaction is NON-elementary (order != stoichiometric coefficient).

    mode : ComponentMode.CUSTOM (default) or ComponentMode.THERMO

    reversible : bool
        If True, r = r_forward - r_reverse, where r_reverse uses
        `reverse_rate_constant` and mirrors the elementary exponents onto
        the product side (|nu_i| for each product). This covers simple
        elementary-reversible cases; for a non-elementary reverse rate
        law, model it as two separate PowerLawReaction objects instead
        and sum species_rates() yourself.

    conc_unit, time_unit : str
        Cosmetic labels only (nothing is converted). Default mol/m3, s.
    """
    stoichiometry: Dict[str, float]
    rate_constant: RateConstant
    order: Dict[str, float] = field(default_factory=dict)
    mode: ComponentMode = ComponentMode.CUSTOM
    reversible: bool = False
    reverse_rate_constant: Optional[RateConstant] = None
    conc_unit: str = "mol/m3"
    time_unit: str = "s"
    name: str = "R1"

    def __post_init__(self):
        # Fill in missing rate-law exponents with the elementary default.
        for species, nu in self.stoichiometry.items():
            if species not in self.order:
                self.order[species] = abs(nu) if nu < 0 else 0.0

        if self.reversible and self.reverse_rate_constant is None:
            raise ValueError(
                "reversible=True requires a reverse_rate_constant."
            )

    # -------------------- validation helpers --------------------

    def validate_against_thermo(self, thermo_model) -> None:
        """
        Only meaningful when mode == ComponentMode.THERMO. Checks every
        species name in the reaction actually resolves in the given
        ThermoModel instance (i.e. was accepted by the `thermo` library).
        """
        if ThermoModel is None:
            raise ImportError(
                "thermo_model.py / the `thermo` package is not available "
                "in this environment; THERMO mode cannot be validated."
            )
        if not isinstance(thermo_model, ThermoModel):
            raise TypeError("thermo_model must be a ThermoModel instance.")

        missing = [
            s for s in self.species if s not in thermo_model.components_inputs
        ]
        if missing:
            raise ValueError(
                f"Species not found in the supplied ThermoModel: {missing}. "
                f"Available components: {thermo_model.components_inputs}"
            )

    # -------------------- properties --------------------

    @property
    def species(self) -> List[str]:
        return list(self.stoichiometry.keys())

    @property
    def reaction_order(self) -> float:
        """Overall (global) forward reaction order = sum of the n_i."""
        return sum(self.order.values())

    @property
    def k_units(self) -> str:
        """
        Units k must carry for r = k * Prod(C_i**n_i) to come out in
        [conc_unit]/[time_unit]:

            [k] = [conc_unit]^(1 - sum(n_i)) / [time_unit]
        """
        exponent = 1 - self.reaction_order
        if abs(exponent) < 1e-9:
            return f"1/{self.time_unit}"
        return f"({self.conc_unit})^{exponent:.4g} / {self.time_unit}"

    # -------------------- rate calculations --------------------

    def _power_law(self, k: float, exponents: Dict[str, float],
                    concentrations: Dict[str, float]) -> float:
        rate = k
        for species, n in exponents.items():
            if n == 0:
                continue
            C = concentrations.get(species, 0.0)
            if C < 0:
                raise ValueError(
                    f"Concentration of '{species}' cannot be negative."
                )
            rate *= C ** n
        return rate

    def rate(self, T: float, concentrations: Dict[str, float]) -> float:
        """
        Net reaction rate r, in [conc_unit]/[time_unit].
        """
        k_fwd = self.rate_constant.value(T)
        r = self._power_law(k_fwd, self.order, concentrations)

        if self.reversible:
            k_rev = self.reverse_rate_constant.value(T)
            reverse_exponents = {
                sp: abs(nu) for sp, nu in self.stoichiometry.items() if nu > 0
            }
            r -= self._power_law(k_rev, reverse_exponents, concentrations)

        return r

    def species_rates(self, T: float,
                       concentrations: Dict[str, float]) -> Dict[str, float]:
        """
        Per-species rate r_i = nu_i * r, in [conc_unit]/[time_unit].
        Negative -> net consumption, positive -> net formation.
        """
        r = self.rate(T, concentrations)
        return {species: nu * r for species, nu in self.stoichiometry.items()}


# ============================================================
# CONVENIENCE FACTORIES
# ============================================================

def elementary_reaction(stoichiometry: Dict[str, float],
                         rate_constant: RateConstant,
                         **kwargs) -> PowerLawReaction:
    """Shorthand for a PowerLawReaction with no explicit `order` given
    (i.e. order defaults to the elementary assumption)."""
    return PowerLawReaction(stoichiometry=stoichiometry,
                             rate_constant=rate_constant, **kwargs)


def from_thermo(thermo_model, stoichiometry: Dict[str, float],
                 rate_constant: RateConstant, **kwargs) -> PowerLawReaction:
    """Build a THERMO-mode reaction and immediately validate species names
    against the given ThermoModel instance."""
    rxn = PowerLawReaction(stoichiometry=stoichiometry,
                            rate_constant=rate_constant,
                            mode=ComponentMode.THERMO, **kwargs)
    rxn.validate_against_thermo(thermo_model)
    return rxn


# # ============================================================
# # DEMO / SANITY CHECKS
# # ============================================================

# if __name__ == "__main__":

#     print("=" * 70)
#     print("DEMO 1: Elementary, irreversible, constant k")
#     print("A -> B , r = k * C_A , k = 0.05 1/min (isothermal, T ignored)")
#     print("=" * 70)

#     rxn1 = elementary_reaction(
#         stoichiometry={"A": -1, "B": 1},
#         rate_constant=ConstantRateConstant(k_value=0.05),
#         time_unit="min",
#     )
#     print("k units required:", rxn1.k_units)
#     print("overall order   :", rxn1.reaction_order)
#     conc = {"A": 2.0, "B": 0.0}
#     print("r =", rxn1.rate(T=None, concentrations=conc))
#     print("species rates   :", rxn1.species_rates(T=None, concentrations=conc))

#     print("\n" + "=" * 70)
#     print("DEMO 2: Non-elementary, Arrhenius k")
#     print("2A + B -> C , rate law order in A is 1.5 (not 2!), order in B is 1")
#     print("=" * 70)

#     rxn2 = PowerLawReaction(
#         stoichiometry={"A": -2, "B": -1, "C": 1},
#         order={"A": 1.5, "B": 1.0, "C": 0.0},
#         rate_constant=ArrheniusRateConstant(Ea=75000.0, A=4.2e7),
#         conc_unit="mol/m3", time_unit="s",
#     )
#     print("k units required:", rxn2.k_units)
#     print("overall order   :", rxn2.reaction_order)
#     conc2 = {"A": 1.5, "B": 0.8, "C": 0.0}
#     T = 350.0
#     print(f"k(T={T} K)      :", rxn2.rate_constant.value(T))
#     print("r =", rxn2.rate(T=T, concentrations=conc2))
#     print("species rates   :", rxn2.species_rates(T=T, concentrations=conc2))

#     print("\n" + "=" * 70)
#     print("DEMO 3: Reversible elementary, k from a reference point")
#     print("A <=> B , forward k_ref=0.1 1/s at 300K, reverse k_ref=0.02 1/s at 300K")
#     print("=" * 70)

#     rxn3 = PowerLawReaction(
#         stoichiometry={"A": -1, "B": 1},
#         rate_constant=ArrheniusRateConstant(Ea=50000.0, T_ref=300.0, k_ref=0.1),
#         reversible=True,
#         reverse_rate_constant=ArrheniusRateConstant(
#             Ea=60000.0, T_ref=300.0, k_ref=0.02
#         ),
#     )
#     conc3 = {"A": 1.0, "B": 0.2}
#     print("r(300K) =", rxn3.rate(T=300.0, concentrations=conc3))
#     print("r(350K) =", rxn3.rate(T=350.0, concentrations=conc3))