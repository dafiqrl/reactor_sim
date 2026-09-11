from thermo import Chemical, ChemicalConstantsPackage, IdealGas


class ThermoModel:
    """
    Thermodynamic model for component resolution
    and ideal-gas mixture calculations.
    """

    R = 8.314462618
    R_KMOL = 8314.462618
    P_REF = 101325.0
    T_REF = 298.15

    def __init__(self, components):

        if not components:
            raise ValueError("Components have to be filled.")

        self.components = []
        self.components_inputs = []

        for component in components:

            component_input = str(component).strip()

            if not component_input:
                raise ValueError(
                    "Components cannot be empty."
                )

            try:
                chemical = Chemical(component_input)

            except Exception as e:
                raise ValueError(
                    f"Component '{component_input}' "
                    "was not found in the thermo database."
                ) from e

            self.components.append(chemical)
            self.components_inputs.append(component_input)

        # Build thermo property package
        CASs = [
            component.CAS
            for component in self.components
        ]

        self.constants, self.correlations = (
            ChemicalConstantsPackage.from_IDs(CASs)
        )

    # ============================================================
    # VALIDATION
    # ============================================================

    def _validate_state(self, T, P):

        if T <= 0:
            raise ValueError(
                "Temperature must be greater than 0 K."
            )

        if P <= 0:
            raise ValueError(
                "Pressure must be greater than 0 Pa."
            )

    def _validate_composition(self, composition):

        if len(composition) != len(self.components):
            raise ValueError(
                "Number of composition values must match "
                "number of components."
            )

        if any(x < 0 for x in composition):
            raise ValueError(
                "Mole fractions cannot be negative."
            )

        total = sum(composition)

        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Mole fractions must sum to 1.0. "
                f"Current sum = {total}"
            )

    # ============================================================
    # COMPONENT INFORMATION
    # ============================================================

    def component_info(self):

        info = []

        for component in self.components:

            info.append({
                "name": component.name,
                "formula": component.formula,
                "CAS": component.CAS,
                "MW": component.MW,
                "Tc": component.Tc,
                "Pc": component.Pc,
                "Tb": component.Tb,
                "Tm": component.Tm,
                "Hf": component.Hf,
                "Hfg": component.Hfg,
                "Hfm": component.Hfm,
                "Hfgm": component.Hfgm,
                "S0m": component.S0m,
                "Hc": component.Hc,
                "Hc_lower": component.Hc_lower,
            })

        return info

    # ============================================================
    # IDEAL GAS PHASE
    # ============================================================

    def _create_ideal_gas(self, T, P, zs):

        gas = IdealGas(
            T=T,
            P=P,
            zs=zs,
            HeatCapacityGases=(
                self.correlations.HeatCapacityGases
            ),
        )

        return gas

    # ============================================================
    # PURE COMPONENT PROPERTIES
    # ============================================================

    def component_properties(self, T, P):

        self._validate_state(T, P)

        properties = []

        for i, component in enumerate(self.components):

            zs = [0.0] * len(self.components)
            zs[i] = 1.0

            gas = self._create_ideal_gas(
                T=T,
                P=P,
                zs=zs,
            )

            properties.append({
                "name": component.name,
                "formula": component.formula,
                "CAS": component.CAS,
                "MW": component.MW,

                "Cp": gas.Cp(),
                "Cv": gas.Cv(),

                "H": gas.H(),
                "U": gas.U(),

                "S": gas.S(),
                "G": gas.G(),

                "V": gas.V(),
                "density": gas.rho(),

                "Z": gas.Z(),

                "Hfg": component.Hfgm,
                "S0": component.S0m,
            })

        return properties

    # ============================================================
    # MIXTURE CALCULATION
    # ============================================================

    def calculate(self, T, P, composition):

        self._validate_state(T, P)
        self._validate_composition(composition)

        gas = self._create_ideal_gas(
            T=T,
            P=P,
            zs=composition,
        )

        # Mixture molecular weight
        MW = sum(
            zi * MWi
            for zi, MWi in zip(
                composition,
                self.constants.MWs
            )
        )

        result = {
            "T": T,
            "P": P,

            "MW": MW,

            "Cp": gas.Cp(),
            "Cv": gas.Cv(),

            "H": gas.H(),
            "U": gas.U(),

            "S": gas.S(),
            "G": gas.G(),

            "molar_volume": gas.V(),
            "density": gas.rho(),

            "Z": gas.Z(),
        }

        return result


# ================================================================
# INTERACTIVE TESTING
# ================================================================

def run_test():

    print("=" * 70)
    print("THERMO ENGINE - IDEAL GAS")
    print("=" * 70)

    # ------------------------------------------------------------
    # COMPONENT INPUT
    # ------------------------------------------------------------

    print("\nEnter components separated by comma.")
    print("Example: methane, ethane, propane")

    component_input = input("\nComponents: ")

    components = [
        component.strip()
        for component in component_input.split(",")
        if component.strip()
    ]

    if not components:
        print("\nError: No components were entered.")
        return

    # ------------------------------------------------------------
    # CREATE MODEL
    # ------------------------------------------------------------

    try:
        model = ThermoModel(components)

    except ValueError as e:
        print(f"\nError: {e}")
        return

    # ------------------------------------------------------------
    # DISPLAY COMPONENTS
    # ------------------------------------------------------------

    print("\nCOMPONENTS")
    print("-" * 70)

    info = model.component_info()

    for i, component in enumerate(info, start=1):

        print(
            f"{i}. "
            f"{component['name']} "
            f"({component['formula']}) "
            f"| CAS: {component['CAS']} "
            f"| MW: {component['MW']:.5f} g/mol"
        )

    # ------------------------------------------------------------
    # TEMPERATURE
    # ------------------------------------------------------------

    try:
        T = float(
            input("\nTemperature [K]: ")
        )

    except ValueError:
        print("\nError: Temperature must be a number.")
        return

    # ------------------------------------------------------------
    # PRESSURE
    # ------------------------------------------------------------

    try:
        P_kPa = float(
            input("\nPressure [kPa]: ")
        )

    except ValueError:
        print("\nError: Pressure must be a number.")
        return

    P = P_kPa * 1000.0

    # ------------------------------------------------------------
    # COMPOSITION
    # ------------------------------------------------------------

    print("\nEnter mole fractions.")
    print("Example: 0.70, 0.20, 0.10")

    composition_input = input(
        "\nMole fractions: "
    )

    try:
        composition = [
            float(x.strip())
            for x in composition_input.split(",")
        ]

    except ValueError:
        print(
            "\nError: Mole fractions must be numbers."
        )
        return

    # ------------------------------------------------------------
    # CALCULATION
    # ------------------------------------------------------------

    try:

        result = model.calculate(
            T=T,
            P=P,
            composition=composition,
        )

    except ValueError as e:

        print(f"\nError: {e}")
        return

    # ------------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MIXTURE RESULTS")
    print("=" * 70)

    print(
        f"{'Temperature':25s}: "
        f"{result['T']:.3f} K"
    )

    print(
        f"{'Pressure':25s}: "
        f"{result['P'] / 1000:.3f} kPa"
    )

    print(
        f"{'Molecular weight':25s}: "
        f"{result['MW']:.6f} g/mol"
    )

    print(
        f"{'Cp':25s}: "
        f"{result['Cp']:.6f} J/mol/K"
    )

    print(
        f"{'Cv':25s}: "
        f"{result['Cv']:.6f} J/mol/K"
    )

    print(
        f"{'Enthalpy (H)':25s}: "
        f"{result['H']:.6f} J/mol"
    )

    print(
        f"{'Internal energy (U)':25s}: "
        f"{result['U']:.6f} J/mol"
    )

    print(
        f"{'Entropy (S)':25s}: "
        f"{result['S']:.6f} J/mol/K"
    )

    print(
        f"{'Gibbs energy (G)':25s}: "
        f"{result['G']:.6f} J/mol"
    )

    print(
        f"{'Molar volume':25s}: "
        f"{result['molar_volume']:.6f} m3/mol"
    )

    print(
        f"{'Density':25s}: "
        f"{result['density']:.6f} kg/m3"
    )

    print(
        f"{'Compressibility factor Z':25s}: "
        f"{result['Z']:.6f}"
    )

    print("=" * 70)


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":
    run_test()