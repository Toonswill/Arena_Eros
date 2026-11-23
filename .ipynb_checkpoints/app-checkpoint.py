import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
import tempfile
import os

st.set_page_config(page_title="DC Meshgrid & Energy Router Planner", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
ESTATES_DIR = BASE_DIR / "estates"
ESTATES_DIR.mkdir(exist_ok=True)


def discover_local_estates(folder: Path) -> dict:
    """Scan the estates folder for all .geojson files and return {Display Name: file_path}."""
    estates = {}
    for file_path in sorted(folder.glob("*.geojson")):
        clean_name = (
            file_path.stem.replace("_", " ")
            .replace("`", "'")
            .title()
        )
        estates[clean_name] = str(file_path)
    return estates


def compute_irr(cashflows):
    roots = np.roots(cashflows)
    real_pos = [r.real for r in roots if np.isreal(r) and r.real > 0]
    if not real_pos:
        return np.nan
    best_x = min(real_pos, key=lambda x: abs(x - 1.0))
    return (best_x - 1.0) * 100.0


@st.cache_data
def run_estate_simulation(
    geojson_path,
    estate_name,
    pv_kwp,
    batt_kwh,
    dc_bus_v,
    cable_mm2,
    kplc_tariff_kes,
    retail_discount_pct,
    fixed_cap_kes_mo,
    anchor_kwh_day,
    lease_years,
    pv_usd_kwp,
    batt_usd_kwh,
    router_usd_hh,
):
    raw = gpd.read_file(geojson_path)
    buildings = raw[raw.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    if "building" in buildings.columns:
        buildings = buildings[buildings["building"].notna()].copy()

    houses = buildings.to_crs(buildings.estimate_utm_crs()).reset_index(drop=True)
    n_houses = len(houses)
    houses["house_id"] = [f"H{i+1:02d}" for i in range(n_houses)]

    centroids = houses.geometry.centroid
    coords_m = np.column_stack([centroids.x, centroids.y])
    dist_matrix = squareform(pdist(coords_m))

    houses["nearest_neighbor_m"] = np.round([np.delete(dist_matrix[i], i).min() for i in range(n_houses)], 1)
    houses["roof_area_m2"] = houses.geometry.area.round(1)

    # Build MST + N-1 closed ring
    G = nx.from_numpy_array(dist_matrix)
    mst = nx.minimum_spanning_tree(G, weight="weight")
    ring_graph = mst.copy()

    leaves = [n for n, d in mst.degree() if d == 1]
    used = set()
    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            u, v = leaves[i], leaves[j]
            if u in used or v in used:
                continue
            d_uv = dist_matrix[u, v]
            if 18.0 <= d_uv <= 42.0 and not ring_graph.has_edge(u, v):
                if nx.shortest_path_length(mst, u, v, weight="weight") > 80.0:
                    ring_graph.add_edge(u, v, weight=d_uv, is_redundancy_tie=True)
                    used.update([u, v])

    ring_m = sum(d["weight"] for _, _, d in ring_graph.edges(data=True))

    # 24-hour nodal DC droop simulation
    np.random.seed(42)
    rho_cu = 0.0175
    dt_h = 0.25
    steps = 96
    hours = np.arange(steps) * dt_h

    G_bus = np.zeros((n_houses, n_houses))
    for u, v, data in ring_graph.edges(data=True):
        r_line = (2 * rho_cu * data["weight"]) / cable_mm2
        g_line = 1.0 / r_line
        G_bus[u, u] += g_line
        G_bus[v, v] += g_line
        G_bus[u, v] -= g_line
        G_bus[v, u] -= g_line

    solar_irr = np.maximum(0, np.sin(np.pi * (hours - 6.25) / 12.25)) ** 1.35
    cloud_dip = 1.0 - 0.18 * np.exp(-((hours - 14.5) / 1.2) ** 2)
    pv_gen_kw = np.outer(np.full(n_houses, pv_kwp) * 0.78, solar_irr * cloud_dip)

    load_kw = np.zeros((n_houses, steps))
    for i in range(n_houses):
        base = np.random.uniform(0.25, 0.55)
        morn = np.random.uniform(0.8, 2.2) * np.exp(-((hours - np.random.uniform(6.5, 8.0)) / 0.9) ** 2)
        day = np.random.uniform(0.2, 1.8) * np.exp(-((hours - np.random.uniform(12.0, 14.5)) / 1.8) ** 2)
        eve = np.random.uniform(1.5, 3.6) * np.exp(-((hours - np.random.uniform(19.0, 21.0)) / 1.4) ** 2)
        load_kw[i] = base + morn + day + eve

    soc = np.random.uniform(0.30, 0.75, size=n_houses)
    soc[0] = 0.22
    soc[-1] = 0.85

    g_droop = np.full(n_houses, 0.85)
    soc_hist = np.zeros((n_houses, steps))
    v_bus_hist = np.zeros((n_houses, steps))
    p_share_hist = np.zeros((n_houses, steps))
    grid_backup_hist = np.zeros((n_houses, steps))

    for t in range(steps):
        soc_hist[:, t] = soc
        net_kw = pv_gen_kw[:, t] - load_kw[:, t]
        max_chg = np.minimum(5.0, ((0.95 - soc) * batt_kwh) / (dt_h * 0.95))
        max_dis = np.minimum(5.0, ((soc - 0.15) * batt_kwh * 0.95) / dt_h)

        v_ref = np.clip(dc_bus_v + 14.0 * (soc - 0.55) + 4.0 * np.tanh(net_kw / 3.0), dc_bus_v - 12.0, dc_bus_v + 12.0)
        v_nodes = np.linalg.solve(G_bus + np.diag(g_droop), g_droop * v_ref)
        p_bus_kw = np.clip((v_nodes * g_droop * (v_ref - v_nodes)) / 1000.0, -4.0, 4.0)
        p_batt_req = net_kw - p_bus_kw

        for i in range(n_houses):
            if p_batt_req[i] >= 0:
                p_chg = min(p_batt_req[i], max_chg[i])
                soc[i] += (p_chg * 0.95 * dt_h) / batt_kwh
            else:
                needed = -p_batt_req[i]
                p_dis = min(needed, max_dis[i])
                grid_backup_hist[i, t] = max(0.0, needed - p_dis)
                soc[i] -= (p_dis / 0.95 * dt_h) / batt_kwh

        soc = np.clip(soc, 0.15, 0.95)
        v_bus_hist[:, t] = v_nodes
        p_share_hist[:, t] = p_bus_kw

    # Financials (Lease-to-Own SPV)
    usd_to_kes = 129.0
    pv_cost = pv_kwp * n_houses * pv_usd_kwp
    batt_cost = batt_kwh * n_houses * batt_usd_kwh
    router_cost = n_houses * router_usd_hh
    cable_cost = ring_m * 10.0
    capex_usd = pv_cost + batt_cost + router_cost + cable_cost
    capex_kes = capex_usd * usd_to_kes

    grid_backup_daily = float(np.sum(grid_backup_hist) * dt_h)
    total_load_daily = 12.0 * n_houses
    displaced_daily = max(0.0, total_load_daily - grid_backup_daily)
    self_suff_pct = (displaced_daily / total_load_daily) * 100.0

    annual_res_kwh = displaced_daily * 365.0 * 0.92
    annual_anchor_kwh = anchor_kwh_day * 365.0 * 0.92
    retail_tariff = kplc_tariff_kes * (1 - retail_discount_pct / 100.0)

    cashflows = [-capex_kes]
    cum_cf = -capex_kes
    payback_yr = None
    for yr in range(1, 16):
        esc = (1.04) ** (yr - 1)
        om = capex_kes * 0.015
        if yr <= lease_years:
            rev = (annual_res_kwh * retail_tariff * esc) + (n_houses * fixed_cap_kes_mo * 12 * esc) + (annual_anchor_kwh * 25.0 * esc)
        else:
            rev = (n_houses * 1800.0 * 12 * esc) + (annual_anchor_kwh * 25.0 * esc * 0.5)
        net_cf = rev - om
        cashflows.append(net_cf)
        prev = cum_cf
        cum_cf += net_cf
        if payback_yr is None and cum_cf >= 0:
            payback_yr = round((yr - 1) + abs(prev) / net_cf, 1)

    irr_pct = compute_irr(cashflows)
    p2p_shared_kwh = float(np.sum(np.maximum(0, p_share_hist)) * dt_h)

    return {
        "estate_name": estate_name,
        "houses": houses,
        "coords_m": coords_m,
        "ring_graph": ring_graph,
        "ring_m": round(ring_m, 1),
        "n_houses": n_houses,
        "self_suff_pct": round(self_suff_pct, 1),
        "p2p_shared_kwh": round(p2p_shared_kwh, 1),
        "capex_usd": round(capex_usd),
        "capex_per_hh_usd": round(capex_usd / n_houses),
        "cost_breakdown": {"PV + Inverter": pv_cost, "LiFePO4 BESS": batt_cost, "Energy Routers": router_cost, "DC Ring Cable": cable_cost},
        "payback_yr": payback_yr if payback_yr else 15.0,
        "irr_pct": round(irr_pct, 1),
        "cashflows_m_kes": [c / 1e6 for c in cashflows],
        "hours": hours,
        "soc_hist": soc_hist,
        "v_bus_hist": v_bus_hist,
        "p_share_hist": p_share_hist,
    }


# ================= SIDEBAR CONTROLS =================
st.sidebar.header("1. Select or Upload Estate")
folder_estates = discover_local_estates(ESTATES_DIR)

if folder_estates:
    selected_label = st.sidebar.selectbox(
        f"Estates Folder ({len(folder_estates)} found)",
        list(folder_estates.keys()),
    )
else:
    selected_label = None
    st.sidebar.warning("No .geojson files found in estates/ folder.")

uploaded_file = st.sidebar.file_uploader("Or Upload New Estate (.geojson)", type=["geojson", "json"])

if uploaded_file is not None:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".geojson")
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    active_path = tmp.name
    active_name = uploaded_file.name.replace(".geojson", "").replace(".json", "").replace("_", " ").title()

    if st.sidebar.button("💾 Save to estates/ Folder"):
        save_dest = ESTATES_DIR / uploaded_file.name
        save_dest.write_bytes(uploaded_file.getvalue())
        st.sidebar.success(f"Saved {uploaded_file.name} to estates/!")
        st.rerun()
elif selected_label:
    active_path = folder_estates[selected_label]
    active_name = selected_label
else:
    st.error("Please add a .geojson file to the estates/ folder or upload one in the sidebar.")
    st.stop()

st.sidebar.header("2. Hardware & DC Mesh Sizing")
pv_kwp = st.sidebar.slider("PV per House (kWp)", 2.0, 6.0, 3.0, 0.5)
batt_kwh = st.sidebar.slider("Battery per House (kWh)", 5.0, 15.0, 10.0, 0.5)
dc_bus_v = st.sidebar.selectbox("Shared DC Bus Voltage (V)", [380.0, 48.0], index=0)
cable_mm2 = st.sidebar.selectbox("DC Ring Copper Size (mm²)", [2.5, 4.0, 6.0, 10.0, 50.0, 95.0], index=0)

st.sidebar.header("3. Lease-to-Own SPV Terms")
kplc_tariff = st.sidebar.slider("KPLC Tariff (KES/kWh)", 25.0, 40.0, 32.0, 1.0)
discount_pct = st.sidebar.slider("Resident Tariff Discount (%)", 5, 30, 15, 1)
fixed_fee = st.sidebar.number_input("Monthly Capacity Fee (KES)", 0, 5000, 2500, 250)
anchor_kwh = st.sidebar.slider("Shared Estate Borehole/Light Load (kWh/day)", 0, 80, 35, 5)
lease_yrs = st.sidebar.slider("Lease-to-Own Transfer Term (Years)", 6, 15, 10, 1)

st.sidebar.header("4. Unit CapEx ($ USD)")
pv_usd = st.sidebar.number_input("PV + Inverter ($/kWp)", 500, 1200, 750, 25)
batt_usd = st.sidebar.number_input("LiFePO4 BESS ($/kWh)", 150, 400, 220, 10)
router_usd = st.sidebar.number_input("Energy Router ($/house)", 250, 800, 450, 25)

res = run_estate_simulation(
    active_path, active_name, pv_kwp, batt_kwh, dc_bus_v, cable_mm2,
    kplc_tariff, discount_pct, fixed_fee, anchor_kwh, lease_yrs,
    pv_usd, batt_usd, router_usd
)

# ================= MAIN DASHBOARD =================
st.title(f"⚡ {res['estate_name']} — DC Meshgrid & Energy Router Planner")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Connected Homes", f"{res['n_houses']}")
c2.metric("Closed DC Ring", f"{res['ring_m']} m", f"{round(res['ring_m']/res['n_houses'], 1)} m/hh")
c3.metric("SPV Total CapEx", f"${res['capex_usd']:,}", f"${res['capex_per_hh_usd']:,}/hh")
c4.metric("Self-Sufficiency", f"{res['self_suff_pct']}%", f"{res['p2p_shared_kwh']} kWh/d P2P")
c5.metric("SPV Payback", f"{res['payback_yr']} yrs")
c6.metric("SPV 15-Yr IRR", f"{res['irr_pct']}%")

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ DC Ring Cabling Schematic",
    "🔋 24-Hr DC Droop Simulation",
    "💰 SPV CapEx & Cash Flows",
    "🏢 Multi-Estate Portfolio Roll-Up",
])

with tab1:
    fig1, ax = plt.subplots(figsize=(10, 7.5))
    res["houses"].plot(ax=ax, facecolor="#e8eef5", edgecolor="#4a6fe3", linewidth=1.2)
    coords = res["coords_m"]
    for u, v, d in res["ring_graph"].edges(data=True):
        w = d["weight"]
        is_red = d.get("is_redundancy_tie", False)
        is_road = w > 18.0
        color, lw, ls = ("#7570b3", 2.5, "--") if is_red else (("#d95f02", 2.6, "-") if is_road else ("#1b9e77", 2.0, "-"))
        ax.plot([coords[u, 0], coords[v, 0]], [coords[u, 1], coords[v, 1]], color=color, lw=lw, ls=ls)
        ax.text(
            (coords[u, 0] + coords[v, 0]) / 2,
            (coords[u, 1] + coords[v, 1]) / 2,
            f"{w:.1f}m",
            fontsize=6.5,
            ha="center",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor=color, alpha=0.85),
        )
    ax.scatter(coords[:, 0], coords[:, 1], color="#d90429", s=45, zorder=5)
    for idx, row in res["houses"].iterrows():
        ax.annotate(
            row["house_id"],
            (coords[idx, 0], coords[idx, 1]),
            fontsize=7,
            fontweight="bold",
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
        )

    legend_handles = [
        mlines.Line2D([], [], color="#1b9e77", lw=2.0, ls="-", label="Primary DC Branch (≤ 18m)"),
        mlines.Line2D([], [], color="#d95f02", lw=2.6, ls="-", label="Road / Trench Crossing (> 18m)"),
        mlines.Line2D([], [], color="#7570b3", lw=2.5, ls="--", label="N-1 Redundancy Ring Tie"),
        mlines.Line2D([], [], color="#d90429", marker="o", linestyle="None", markersize=6, label="Energy Router Node"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.5)

    ax.set_title(f"{res['estate_name']} — {int(dc_bus_v)}V DC Closed-Ring Topology ({res['ring_m']} m)")
    ax.set_aspect("equal")
    st.pyplot(fig1)
    plt.close(fig1)

with tab2:
    fig2, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    hrs = res["hours"]
    for i in range(res["n_houses"]):
        axes[0].plot(hrs, res["soc_hist"][i] * 100, color="steelblue", alpha=0.2)
    axes[0].plot(hrs, res["soc_hist"][0] * 100, color="#d90429", lw=2, label="H01 (Low Start: 22%)")
    axes[0].plot(hrs, res["soc_hist"][-1] * 100, color="#2b9348", lw=2, label=f"H{res['n_houses']:02d} (High Start: 85%)")
    axes[0].set_ylabel("Battery SoC (%)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].plot(hrs, res["p_share_hist"][-1], color="#2b9348", lw=1.8, label=f"H{res['n_houses']:02d} (+Export/-Import)")
    axes[1].plot(hrs, res["p_share_hist"][0], color="#d90429", lw=1.8, label="H01 (+Export/-Import)")
    axes[1].set_ylabel("DC Bus Power (kW)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].axhline(dc_bus_v, color="black", ls="--", lw=1)
    axes[2].fill_between(hrs, res["v_bus_hist"].min(axis=0), res["v_bus_hist"].max(axis=0), color="purple", alpha=0.2, label="Bus Voltage Spread")
    axes[2].plot(hrs, res["v_bus_hist"].mean(axis=0), color="purple", lw=2, label="Mean DC Voltage")
    axes[2].set_ylabel("DC Bus (V)")
    axes[2].set_xlabel("Hour of Day")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close(fig2)

with tab3:
    col_a, col_b = st.columns(2)
    with col_a:
        fig3a, ax3a = plt.subplots(figsize=(6, 4.5))
        cb = res["cost_breakdown"]
        bars = ax3a.bar(list(cb.keys()), [v / 1e3 for v in cb.values()], color=["#2a9d8f", "#e9c46a", "#e76f51", "#264653"])
        ax3a.set_ylabel("CapEx ($ Thousands USD)")
        ax3a.set_title("SPV Hardware & Cabling CapEx Breakdown")
        for b in bars:
            ax3a.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"${b.get_height():.1f}k", ha="center", fontsize=8.5, fontweight="bold")
        st.pyplot(fig3a)
        plt.close(fig3a)

    with col_b:
        fig3b, ax3b = plt.subplots(figsize=(6, 4.5))
        yrs = np.arange(16)
        cum = np.cumsum(res["cashflows_m_kes"])
        ax3b.axhline(0, color="black", ls="--", lw=1)
        ax3b.axvline(lease_yrs, color="orange", ls=":", label=f"Transfer Yr {lease_yrs}")
        ax3b.plot(yrs, cum, marker="o", color="#1d3557", lw=2)
        ax3b.fill_between(yrs, 0, cum, where=(cum >= 0), color="#2b9348", alpha=0.2)
        ax3b.set_xlabel("Project Year")
        ax3b.set_ylabel("Cumulative Cash Flow (M KES)")
        ax3b.set_title(f"15-Yr SPV Cash Trajectory (IRR = {res['irr_pct']}%)")
        ax3b.legend()
        st.pyplot(fig3b)
        plt.close(fig3b)

    st.divider()
    st.subheader("📄 Bill of Materials & SPV Financial Summary")

    bom_rows = [
        {
            "Category": "Bill of Materials",
            "Item / Metric": f"Rooftop Solar PV + Hybrid Inverter ({pv_kwp} kWp/hh)",
            "Quantity": f"{round(pv_kwp * res['n_houses'], 1)} kWp",
            "Unit Cost": f"${pv_usd:,.2f} / kWp",
            "Total (USD)": f"${res['cost_breakdown']['PV + Inverter']:,.2f}",
            "Total (KES)": f"KES {res['cost_breakdown']['PV + Inverter'] * 129.0:,.0f}",
        },
        {
            "Category": "Bill of Materials",
            "Item / Metric": f"LiFePO4 Battery Storage ({batt_kwh} kWh/hh)",
            "Quantity": f"{round(batt_kwh * res['n_houses'], 1)} kWh",
            "Unit Cost": f"${batt_usd:,.2f} / kWh",
            "Total (USD)": f"${res['cost_breakdown']['LiFePO4 BESS']:,.2f}",
            "Total (KES)": f"KES {res['cost_breakdown']['LiFePO4 BESS'] * 129.0:,.0f}",
        },
        {
            "Category": "Bill of Materials",
            "Item / Metric": f"Bidirectional DC Energy Router ({int(dc_bus_v)}V Bus)",
            "Quantity": f"{res['n_houses']} units",
            "Unit Cost": f"${router_usd:,.2f} / unit",
            "Total (USD)": f"${res['cost_breakdown']['Energy Routers']:,.2f}",
            "Total (KES)": f"KES {res['cost_breakdown']['Energy Routers'] * 129.0:,.0f}",
        },
        {
            "Category": "Bill of Materials",
            "Item / Metric": f"Underground DC Ring Copper Cable ({cable_mm2} mm²)",
            "Quantity": f"{res['ring_m']} m",
            "Unit Cost": "$10.00 / m",
            "Total (USD)": f"${res['cost_breakdown']['DC Ring Cable']:,.2f}",
            "Total (KES)": f"KES {res['cost_breakdown']['DC Ring Cable'] * 129.0:,.0f}",
        },
        {
            "Category": "Financial Summary",
            "Item / Metric": "Total SPV CapEx",
            "Quantity": f"{res['n_houses']} Homes",
            "Unit Cost": f"${res['capex_per_hh_usd']:,.2f} / home",
            "Total (USD)": f"${res['capex_usd']:,.2f}",
            "Total (KES)": f"KES {res['capex_usd'] * 129.0:,.0f}",
        },
        {
            "Category": "Financial Summary",
            "Item / Metric": "System Self-Sufficiency & P2P Sharing",
            "Quantity": f"{res['self_suff_pct']}%",
            "Unit Cost": f"{res['p2p_shared_kwh']} kWh/day P2P",
            "Total (USD)": "-",
            "Total (KES)": "-",
        },
        {
            "Category": "Financial Summary",
            "Item / Metric": f"Lease-to-Own Terms ({lease_yrs}-Yr Transfer, {discount_pct}% Discount)",
            "Quantity": f"KES {fixed_fee:,}/mo fixed",
            "Unit Cost": f"KES {kplc_tariff * (1 - discount_pct / 100.0):.2f}/kWh retail",
            "Total (USD)": "-",
            "Total (KES)": "-",
        },
        {
            "Category": "Financial Summary",
            "Item / Metric": "SPV Payback Period & 15-Year Unlevered IRR",
            "Quantity": f"{res['payback_yr']} Years",
            "Unit Cost": f"{res['irr_pct']}% IRR",
            "Total (USD)": "-",
            "Total (KES)": f"KES {sum(res['cashflows_m_kes']):.2f}M Net 15-Yr CF",
        },
    ]

    df_export = pd.DataFrame(bom_rows)
    st.dataframe(df_export, width="stretch", hide_index=True)

    safe_filename = res["estate_name"].lower().replace(" ", "_").replace("'", "")
    st.download_button(
        label="⬇️ Download BoM & Financial Summary (CSV)",
        data=df_export.to_csv(index=False).encode("utf-8"),
        file_name=f"{safe_filename}_bom_financials.csv",
        mime="text/csv",
    )

with tab4:
    st.subheader(f"Combined SPV Portfolio Summary ({len(folder_estates)} Estates in Folder)")
    port_rows = []
    for name, path in folder_estates.items():
        if os.path.exists(path):
            r = run_estate_simulation(
                path, name, pv_kwp, batt_kwh, dc_bus_v, cable_mm2,
                kplc_tariff, discount_pct, fixed_fee, anchor_kwh, lease_yrs,
                pv_usd, batt_usd, router_usd
            )
            port_rows.append({
                "Estate": r["estate_name"],
                "Homes": r["n_houses"],
                "DC Ring (m)": r["ring_m"],
                "SPV CapEx ($)": r["capex_usd"],
                "CapEx/Home ($)": r["capex_per_hh_usd"],
                "Self-Sufficiency (%)": r["self_suff_pct"],
                "Payback (Yrs)": r["payback_yr"],
                "15-Yr IRR (%)": r["irr_pct"],
            })
    if port_rows:
        df_port = pd.DataFrame(port_rows)
        st.dataframe(df_port, width="stretch", hide_index=True)
        tot_homes = df_port["Homes"].sum()
        tot_capex = df_port["SPV CapEx ($)"].sum()
        avg_irr = round(df_port["15-Yr IRR (%)"].mean(), 1)
        st.success(f"**Combined Portfolio:** {tot_homes} Homes | Total SPV CapEx: **${tot_capex:,}** | Average Unlevered IRR: **{avg_irr}%**")