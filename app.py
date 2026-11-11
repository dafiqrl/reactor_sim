"""
app.py

Streamlit front-end for the isothermal PFR workbench.
Wraps: thermo_model.ThermoModel, power_law.PowerLawReaction /
ConstantRateConstant / ArrheniusRateConstant, and pfr_reactor.PFRIsothermal.

UI intent: dense, tabular, flat -- a process-simulator "specification
sheet" feel (Aspen Plus-ish) rather than a conversational/chat-app look.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from thermo_model import ThermoModel
from power_law import (
    PowerLawReaction,
    ConstantRateConstant,
    ArrheniusRateConstant,
    ComponentMode,
)
from pfr_reactor import PFRIsothermal


# ============================================================
# PAGE CONFIG + FLAT / DARK "SPEC SHEET" STYLING
# ============================================================

st.set_page_config(
    page_title="PFR-SIM | Isothermal Reactor Workbench",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
html, body, [class*="css"] {
    font-family: 'Roboto Mono', 'Consolas', monospace;
}

/* flat, sharp-edged controls -- no rounded "app" look */
.stButton > button, .stDownloadButton > button {
    border-radius: 2px;
    border: 1px solid #00B4D8;
    background-color: #1B212C;
    color: #D6DEE8;
    font-family: 'Roboto Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    padding: 0.4rem 1rem;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background-color: #00B4D8;
    color: #12161C;
    border: 1px solid #00B4D8;
}

/* section headers: small-caps, uppercase, tracked -- spec-sheet style */
.section-header {
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7FA8C9;
    border-bottom: 1px solid #2A3140;
    padding-bottom: 0.3rem;
    margin-bottom: 0.6rem;
    margin-top: 0.4rem;
}

/* status bar */
.status-bar {
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    padding: 0.3rem 0.6rem;
    border: 1px solid #2A3140;
    background-color: #1B212C;
    color: #7FA8C9;
    margin-bottom: 0.6rem;
}
.status-solved { color: #4ADE80; border-color: #4ADE80; }
.status-unsolved { color: #F2B85C; border-color: #F2B85C; }

/* tighten default streamlit spacing */
div[data-testid="stVerticalBlock"] > div { gap: 0.4rem; }
section[data-testid="stSidebar"] { border-right: 1px solid #2A3140; }

/* tables */
[data-testid="stDataFrame"] { border: 1px solid #2A3140; }

hr { border-color: #2A3140; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def section_header(text: str):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "species_list" not in st.session_state:
    st.session_state.species_list = ["A", "B"]
if "thermo_model" not in st.session_state:
    st.session_state.thermo_model = None
if "reactions_built" not in st.session_state:
    st.session_state.reactions_built = []
if "reactor" not in st.session_state:
    st.session_state.reactor = None
if "results" not in st.session_state:
    st.session_state.results = None


def parse_kv(text: str) -> dict:
    """Parse 'A:-1, B:-2, C:1' style strings into {"A": -1.0, "B": -2.0, "C": 1.0}."""
    result = {}
    if not text or not text.strip():
        return result
    for pair in text.split(","):
        if ":" not in pair:
            raise ValueError(f"Invalid entry '{pair.strip()}'. Use KEY:VALUE format.")
        key, value = pair.split(":")
        key = key.strip()
        if not key:
            raise ValueError("Empty species label found.")
        result[key] = float(value.strip())
    return result


# ============================================================
# TITLE BAR + STATUS
# ============================================================

st.markdown(
    "<h3 style='margin-bottom:0;'>PFR-SIM</h3>"
    "<div style='color:#7FA8C9; font-size:0.85rem; margin-top:-0.4rem;'>"
    "Isothermal Plug-Flow Reactor &mdash; Variable Volume Workbench</div>",
    unsafe_allow_html=True,
)

status_class = "status-solved" if st.session_state.results is not None else "status-unsolved"
status_text = "SOLVED" if st.session_state.results is not None else "NOT SOLVED"
st.markdown(
    f'<div class="status-bar {status_class}">STATUS: {status_text}'
    f' &nbsp;|&nbsp; SPECIES: {len(st.session_state.species_list)}'
    f' &nbsp;|&nbsp; REACTIONS: {len(st.session_state.reactions_built)}</div>',
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR -- GLOBAL SPECIFICATIONS
# ============================================================

with st.sidebar:
    section_header("Global Specifications")

    component_mode = st.selectbox(
        "Component Mode",
        ["Custom (symbolic)", "Thermo (real components)"],
    )

    T = st.number_input("Temperature T [K]", value=350.00, min_value=0.01, step=1.0, format="%.2f")
    P_kPa = st.number_input("Pressure P [kPa]", value=101.325, min_value=0.001, step=1.0, format="%.3f")
    P = P_kPa * 1000.0

    st.caption("T, P apply to the reactor (isothermal) and to Arrhenius k(T) evaluation.")


# ============================================================
# TABS -- SPEC SHEETS
# ============================================================

tab_components, tab_reactions, tab_reactor, tab_results = st.tabs(
    ["1. COMPONENTS", "2. REACTIONS", "3. REACTOR", "4. RESULTS"]
)

# ------------------------------------------------------------
# TAB 1: COMPONENTS
# ------------------------------------------------------------
with tab_components:
    section_header("Component Definition")

    if component_mode == "Custom (symbolic)":
        species_text = st.text_input(
            "Species labels (comma-separated)",
            value=",".join(st.session_state.species_list),
            help="Symbolic labels for textbook-style problems, e.g. A,B,C,D",
        )
        if st.button("APPLY SPECIES LIST"):
            try:
                species = [s.strip() for s in species_text.split(",") if s.strip()]
                if not species:
                    raise ValueError("Species list cannot be empty.")
                st.session_state.species_list = species
                st.session_state.thermo_model = None
                st.success(f"Species list applied: {species}")
            except ValueError as e:
                st.error(str(e))

        st.dataframe(
            pd.DataFrame({"Species": st.session_state.species_list}),
            use_container_width=True,
            hide_index=True,
        )

    else:
        chem_text = st.text_input(
            "Chemical names / formulas (comma-separated)",
            value="methane, ethane",
            help="Resolved via the `thermo` library, e.g. methane, ethylene, carbon dioxide",
        )
        if st.button("RESOLVE COMPONENTS"):
            try:
                names = [c.strip() for c in chem_text.split(",") if c.strip()]
                model = ThermoModel(names)
                st.session_state.thermo_model = model
                st.session_state.species_list = model.components_inputs
                st.success("Components resolved.")
            except ValueError as e:
                st.error(str(e))

        if st.session_state.thermo_model is not None:
            info = st.session_state.thermo_model.component_info()
            df = pd.DataFrame(info)[["name", "formula", "CAS", "MW", "Tc", "Pc", "Tb", "Tm"]]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No components resolved yet.")

# ------------------------------------------------------------
# TAB 2: REACTIONS
# ------------------------------------------------------------
with tab_reactions:
    section_header("Reaction Network")

    n_reactions = st.number_input("Number of reactions", min_value=1, max_value=10, value=1, step=1)

    reaction_configs = []
    for i in range(int(n_reactions)):
        with st.expander(f"REACTION R{i + 1}", expanded=(i == 0)):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Name", value=f"R{i + 1}", key=f"name_{i}")
                stoich_text = st.text_input(
                    "Stoichiometry (SPECIES:nu)",
                    value="A:-1,B:1",
                    key=f"stoich_{i}",
                    help="Negative = reactant, positive = product. e.g. A:-1,B:-2,C:1",
                )
                order_text = st.text_input(
                    "Rate law order override (optional)",
                    value="",
                    key=f"order_{i}",
                    help="Leave blank for elementary assumption. e.g. A:1.5,B:1.0",
                )
            with col2:
                k_type = st.radio(
                    "Rate constant type", ["Constant", "Arrhenius"],
                    key=f"ktype_{i}", horizontal=True,
                )
                if k_type == "Constant":
                    k_value = st.number_input("k value", value=0.1, format="%.6g", key=f"kval_{i}")
                    rate_constant_spec = {"type": "constant", "k_value": k_value}
                else:
                    ea = st.number_input("Ea [J/mol]", value=50000.0, key=f"ea_{i}")
                    arr_mode = st.radio(
                        "Specify via", ["A directly", "Reference point (T_ref, k_ref)"],
                        key=f"arrmode_{i}",
                    )
                    if arr_mode == "A directly":
                        A_val = st.number_input("A (pre-exponential)", value=1.0e7, format="%.6g", key=f"aval_{i}")
                        rate_constant_spec = {"type": "arrhenius", "Ea": ea, "A": A_val}
                    else:
                        t_ref = st.number_input("T_ref [K]", value=300.0, key=f"tref_{i}")
                        k_ref = st.number_input("k_ref", value=0.1, format="%.6g", key=f"kref_{i}")
                        rate_constant_spec = {"type": "arrhenius", "Ea": ea, "T_ref": t_ref, "k_ref": k_ref}

                reversible = st.checkbox("Reversible", key=f"rev_{i}")
                reverse_spec = None
                if reversible:
                    st.caption("Reverse rate constant:")
                    rk_type = st.radio(
                        "Reverse type", ["Constant", "Arrhenius"],
                        key=f"rktype_{i}", horizontal=True,
                    )
                    if rk_type == "Constant":
                        rk_val = st.number_input("Reverse k value", value=0.02, format="%.6g", key=f"rkval_{i}")
                        reverse_spec = {"type": "constant", "k_value": rk_val}
                    else:
                        r_ea = st.number_input("Reverse Ea [J/mol]", value=60000.0, key=f"rea_{i}")
                        r_tref = st.number_input("Reverse T_ref [K]", value=300.0, key=f"rtref_{i}")
                        r_kref = st.number_input("Reverse k_ref", value=0.02, format="%.6g", key=f"rkref_{i}")
                        reverse_spec = {"type": "arrhenius", "Ea": r_ea, "T_ref": r_tref, "k_ref": r_kref}

            reaction_configs.append({
                "name": name,
                "stoich_text": stoich_text,
                "order_text": order_text,
                "rate_constant_spec": rate_constant_spec,
                "reversible": reversible,
                "reverse_spec": reverse_spec,
            })

    def build_rate_constant(spec):
        if spec["type"] == "constant":
            return ConstantRateConstant(k_value=spec["k_value"])
        if "A" in spec:
            return ArrheniusRateConstant(Ea=spec["Ea"], A=spec["A"])
        return ArrheniusRateConstant(Ea=spec["Ea"], T_ref=spec["T_ref"], k_ref=spec["k_ref"])

    if st.button("BUILD REACTIONS"):
        try:
            built = []
            for cfg in reaction_configs:
                stoich = parse_kv(cfg["stoich_text"])
                if not stoich:
                    raise ValueError(f"Reaction '{cfg['name']}': stoichiometry cannot be empty.")
                order = parse_kv(cfg["order_text"]) if cfg["order_text"].strip() else {}

                rate_const = build_rate_constant(cfg["rate_constant_spec"])
                reverse_rate_const = (
                    build_rate_constant(cfg["reverse_spec"]) if cfg["reversible"] else None
                )

                rxn = PowerLawReaction(
                    stoichiometry=stoich,
                    rate_constant=rate_const,
                    order=order,
                    reversible=cfg["reversible"],
                    reverse_rate_constant=reverse_rate_const,
                    name=cfg["name"],
                )
                built.append(rxn)

            st.session_state.reactions_built = built
            st.session_state.reactor = None
            st.session_state.results = None
            st.success(f"{len(built)} reaction(s) built successfully.")
        except (ValueError, TypeError) as e:
            st.error(str(e))

    if st.session_state.reactions_built:
        rows = []
        for rxn in st.session_state.reactions_built:
            rows.append({
                "Name": rxn.name,
                "Stoichiometry": rxn.stoichiometry,
                "Order": rxn.order,
                "Overall order": rxn.reaction_order,
                "k units": rxn.k_units,
                "Reversible": rxn.reversible,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# TAB 3: REACTOR
# ------------------------------------------------------------
with tab_reactor:
    section_header("Feed Specification")

    if not st.session_state.reactions_built:
        st.warning("Build at least one reaction in the REACTIONS tab first.")
    else:
        feed_mode = st.radio(
            "Feed input mode",
            ["Per-species molar flow", "Total flow + mole fractions"],
            horizontal=True,
        )

        if feed_mode == "Per-species molar flow":
            f0_text = st.text_input(
                "F0 (mol/s) per species (SPECIES:value)",
                value="A:10,B:0",
            )
        else:
            f_total = st.number_input("Total inlet molar flow F_total [mol/s]", value=10.0, min_value=0.0001)
            z0_text = st.text_input(
                "Inlet mole fractions (SPECIES:fraction, must sum to 1.0)",
                value="A:1.0,B:0.0",
            )

        section_header("Solve Mode")

        solve_mode = st.radio("Mode", ["Rating (given V, find X)", "Sizing (given X, find V)"], horizontal=True)

        all_species = sorted({sp for rxn in st.session_state.reactions_built for sp in rxn.species})
        basis_choice = st.selectbox("Basis species (conversion reference)", ["Auto-detect"] + all_species)
        basis_species = None if basis_choice == "Auto-detect" else basis_choice

        n_points = st.slider("Profile resolution (points)", min_value=50, max_value=1000, value=200, step=50)

        if solve_mode.startswith("Rating"):
            V_max = st.number_input("Reactor volume V_max [m3]", value=2.0, min_value=0.0001)
        else:
            X_target = st.slider("Target conversion X_target", min_value=0.01, max_value=0.99, value=0.85)
            V_search_max = st.number_input("Search upper bound V_search_max [m3]", value=100.0, min_value=0.001)

        if st.button("RUN"):
            try:
                if feed_mode == "Per-species molar flow":
                    F0 = parse_kv(f0_text)
                    reactor = PFRIsothermal(
                        reactions=st.session_state.reactions_built, F0=F0, T=T, P=P
                    )
                else:
                    z0 = parse_kv(z0_text)
                    reactor = PFRIsothermal.from_composition(
                        reactions=st.session_state.reactions_built,
                        F_total=f_total, z0=z0, T=T, P=P,
                    )

                if solve_mode.startswith("Rating"):
                    results = reactor.solve(V_max=V_max, basis_species=basis_species, n_points=n_points)
                else:
                    results = reactor.solve_for_conversion(
                        X_target=X_target, basis_species=basis_species,
                        V_search_max=V_search_max, n_points=n_points,
                    )

                st.session_state.reactor = reactor
                st.session_state.results = results
                st.success("Solved. See RESULTS tab.")
            except (ValueError, RuntimeError) as e:
                st.error(str(e))

# ------------------------------------------------------------
# TAB 4: RESULTS
# ------------------------------------------------------------
with tab_results:
    section_header("Solution Summary")

    if st.session_state.results is None:
        st.info("No results yet. Configure and RUN from the REACTOR tab.")
    else:
        results = st.session_state.results
        reactor = st.session_state.reactor

        col1, col2, col3 = st.columns(3)
        col1.metric("Basis species", results["basis_species"])
        col2.metric("Final conversion X", f"{results['X'][-1]:.4f}")
        if "V_required" in results:
            col3.metric("V required [m3]", f"{results['V_required']:.4f}")
        else:
            col3.metric("Reactor volume [m3]", f"{results['V'][-1]:.4f}")

        show_species = st.checkbox("Show per-species molar flow profiles", value=False)
        fig = reactor.plot(show_species=show_species, dark_mode=True)
        st.plotly_chart(fig, use_container_width=True)

        section_header("Profile Data")

        data = {"V": results["V"], "X": results["X"], "Q": results["Q"]}
        for sp in reactor.species:
            data[f"F_{sp}"] = results["F"][sp]
            data[f"C_{sp}"] = results["C"][sp]

        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("DOWNLOAD RESULTS (CSV)", data=csv, file_name="pfr_results.csv", mime="text/csv")
