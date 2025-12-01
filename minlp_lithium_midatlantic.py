import pandas as pd
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, NonNegativeReals, Binary,
    Objective, Constraint, Expression, SolverFactory, value, minimize
)

# ---------
# 1. Basic settings and file paths
# --------

DATA_FILE = "MINLP_GIS_InputData_MidAtlantic.xlsx"

# Limit number of candidate sites used in the run
MAX_SITES_FOR_RUN = 50

# Cost parameters
TRANSPORT_COST_PER_TKM = 0.5      # $ per tonne-km
CAPITAL_COST_COEFF = 5e4          # capital cost scaling
CAPITAL_EXPONENT = 0.7            # economies of scale exponent
OPERATING_COST_COEFF = 200.0      # operating cost scaling
OPERATING_EXPONENT = 0.9          # non-linear operating exponent

# Large distance used when GIS distance is missing
BIG_DISTANCE = 1e6


# ---------------
# 2. Load data from Excel
# -----------------------------------

demand_df = pd.read_excel(DATA_FILE, sheet_name="Demand nodes")
sites_df_raw = pd.read_excel(DATA_FILE, sheet_name="Candidate_sites")
dist_df_raw = pd.read_excel(DATA_FILE, sheet_name="Distance_Matrix")

print("DEMAND COLUMNS:", demand_df.columns.tolist())
print("SITE COLUMNS:", sites_df_raw.columns.tolist())
print("DISTANCE COLUMNS (first 20):", dist_df_raw.columns.tolist()[:20])


# --------------------------
# 3. Clean demand nodes
# ---------------------------

if "OBJECTID" not in demand_df.columns:
    raise RuntimeError("Expected 'OBJECTID' column in 'Demand nodes' sheet.")

demand_df["Waste_proxy"] = pd.to_numeric(demand_df["Waste_proxy"], errors="coerce").fillna(0.0)
demand_df = demand_df.set_index("OBJECTID")


# ------------------------
# 4. Clean candidate sites
# -----------------

sites_df = sites_df_raw.copy()
sites_df.rename(columns=lambda c: str(c).strip(), inplace=True)  # fix 'OBJECTID ' → 'OBJECTID'

if "OBJECTID" not in sites_df.columns:
    raise RuntimeError("Expected 'OBJECTID' column in 'Candidate_sites' sheet.")

sites_df = sites_df.set_index("OBJECTID")

site_ids_all = sites_df.index.tolist()
site_ids = site_ids_all[:MAX_SITES_FOR_RUN]


# -----------
# 5. Clean distance matrix
# ------------------------------------------

dist_df = dist_df_raw.copy()

# First column should be demand_id
if dist_df.columns[0] != "demand_id":
    dist_df.rename(columns={dist_df.columns[0]: "demand_id"}, inplace=True)

# Drop summary columns if present
for col in ["(blank)", "Grand Total"]:
    if col in dist_df.columns:
        dist_df.drop(columns=[col], inplace=True)


def is_int_like(v):
    try:
        int(v)
        return True
    except (TypeError, ValueError):
        return False


# Keep only numeric demand_id rows
dist_df = dist_df[dist_df["demand_id"].apply(is_int_like)]
dist_df["demand_id"] = dist_df["demand_id"].astype(int)
dist_df = dist_df.set_index("demand_id")

# Make all entries numeric (not strings)
for col in dist_df.columns:
    dist_df[col] = pd.to_numeric(dist_df[col], errors="coerce")

# Keep only site columns that correspond to chosen site IDs
numeric_site_cols = []
for col in dist_df.columns:
    try:
        as_int = int(col)
        if as_int in site_ids:
            numeric_site_cols.append(as_int)
    except (TypeError, ValueError):
        continue

dist_df = dist_df[numeric_site_cols]
dist_df.columns = numeric_site_cols

# Drop rows that are completely NaN
dist_df = dist_df.dropna(how="all")

# Replace remaining NaN distances with a big penalty distance
dist_df = dist_df.fillna(BIG_DISTANCE)

print("DISTANCE MATRIX SHAPE:", dist_df.shape)
print(dist_df.head())


# ----------
# 6. Align demand and sites with distance matrix
# ---------

common_demand_ids = sorted(set(demand_df.index).intersection(set(dist_df.index)))
demand_df = demand_df.loc[common_demand_ids]
dist_df = dist_df.loc[common_demand_ids]

print("Number of demand nodes used:", len(common_demand_ids))

site_ids = sorted(set(site_ids).intersection(set(dist_df.columns)))
sites_df = sites_df.loc[site_ids]
dist_df = dist_df[site_ids]

print("Number of candidate sites used in model:", len(site_ids))

# ------------------
# 7. Build parameter dictionaries
# ------------

demand_dict = demand_df["Waste_proxy"].to_dict()
total_demand = float(demand_df["Waste_proxy"].sum())
print("Total demand (sum of Waste_proxy):", total_demand)

# Simple capacity assumption (can be refined)
if len(site_ids) > 0:
    base_capacity = total_demand / max(1, len(site_ids) / 5.0)
else:
    base_capacity = 0.0

capacity_dict = {i: base_capacity for i in site_ids}


def dist_lookup(j, i):
    return float(dist_df.loc[j, i])


# --------------------------------------
# 8. Build MINLP model in Pyomo
# -------------------------------

model = ConcreteModel()

model.I = Set(initialize=site_ids)             # candidate sites
model.J = Set(initialize=common_demand_ids)    # demand nodes

model.D = Param(model.J, initialize=demand_dict, within=NonNegativeReals)
model.Cap = Param(model.I, initialize=capacity_dict, within=NonNegativeReals)


def distance_init(m, j, i):
    return dist_lookup(j, i)


model.Dist = Param(model.J, model.I, initialize=distance_init, within=NonNegativeReals)

# Decision variables
model.x = Var(model.J, model.I, within=NonNegativeReals)
model.y = Var(model.I, within=Binary)


# Total flow through each site
def flow_rule(m, i):
    return sum(m.x[j, i] for j in m.J)


model.Flow = Expression(model.I, rule=flow_rule)


# --------------
# 9. Constraints
# --

def demand_balance_rule(m, j):
    return sum(m.x[j, i] for i in m.I) == m.D[j]


model.DemandBalance = Constraint(model.J, rule=demand_balance_rule)


def capacity_rule(m, i):
    return m.Flow[i] <= m.Cap[i] * m.y[i]


model.CapacityLimit = Constraint(model.I, rule=capacity_rule)


def logical_flow_rule(m, j, i):
    return m.x[j, i] <= m.D[j] * m.y[i]


model.FlowOnlyIfOpen = Constraint(model.J, model.I, rule=logical_flow_rule)


# ----
# 10. Objective: non-linear cost (MINLP)
# ---------------------

transport_cost_expr = sum(
    TRANSPORT_COST_PER_TKM * model.Dist[j, i] * model.x[j, i]
    for j in model.J for i in model.I
)

capital_cost_expr = sum(
    CAPITAL_COST_COEFF * (model.Flow[i] + 1.0) ** CAPITAL_EXPONENT * model.y[i]
    for i in model.I
)

operating_cost_expr = sum(
    OPERATING_COST_COEFF * (model.Flow[i] + 1.0) ** OPERATING_EXPONENT
    for i in model.I
)

model.TotalCost = Objective(
    expr=transport_cost_expr + capital_cost_expr + operating_cost_expr,
    sense=minimize
)


# ---------------
# 11. Solve with Bonmin
# ---------------

solver = SolverFactory("bonmin")

if not solver.available(False):
    raise RuntimeError(
        "Bonmin solver is not available. "
        "Check that 'bonmin' is on PATH for this Python environment."
    )

results = solver.solve(model, tee=True)


# -------------
# 12. Report results
# ----------------

print("\n==== SOLUTION SUMMARY ====")
print("Solver status:", results.solver.status)
print("Termination condition:", results.solver.termination_condition)

try:
    obj_value = value(model.TotalCost)
    print("Total cost:", obj_value)
except Exception:
    print("Could not evaluate objective value.")

open_sites = [i for i in model.I if value(model.y[i]) > 0.5]
print("Number of open facilities:", len(open_sites))
print("Open facility IDs:", open_sites)

for i in open_sites:
    flow_i = value(model.Flow[i])
    print(f"Site {i}: throughput = {flow_i:.2f} (same units as Waste_proxy)")
# ------------------
# 13. Export results to CSV for GIS and further analysis
# -----------
import os

# Convert open sites and flows into DataFrames
open_site_ids = [i for i in model.I if value(model.y[i]) > 0.5]

# 13.1 Open facilities table
open_sites_records = []
for i in open_site_ids:
    flow_i = float(value(model.Flow[i]))
    # Get site geometry / attributes from the original sites_df
    if i in sites_df.index:
        sx = float(sites_df.loc[i, "x_coord"])
        sy = float(sites_df.loc[i, "y_coord"])
        suit_val = float(sites_df.loc[i, "Suitability"])
    else:
        sx = float("nan")
        sy = float("nan")
        suit_val = float("nan")

    open_sites_records.append({
        "site_id": i,
        "throughput": flow_i,
        "suitability": suit_val,
        "x_coord": sx,
        "y_coord": sy
    })

open_sites_df = pd.DataFrame(open_sites_records)

# 13.2 Flows table (only positive flows)
flow_records = []
for j in model.J:
    for i in model.I:
        x_val = float(value(model.x[j, i]))
        if x_val > 0.0:
            dij = float(model.Dist[j, i])
            flow_records.append({
                "demand_id": j,
                "site_id": i,
                "flow": x_val,
                "distance_km": dij
            })

flows_df = pd.DataFrame(flow_records)

# 13.3 Write CSVs next to the script
output_dir = os.path.dirname(os.path.abspath(__file__))
open_sites_path = os.path.join(output_dir, "open_facilities_results.csv")
flows_path = os.path.join(output_dir, "flows_results.csv")

open_sites_df.to_csv(open_sites_path, index=False)
flows_df.to_csv(flows_path, index=False)

print("\nResults exported:")
print("  Open facilities  →", open_sites_path)
print("  Flows (x_ji)     →", flows_path)
