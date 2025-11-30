import pandas as pd
from pyomo.environ import (
    ConcreteModel, Set, Param, Var,
    NonNegativeReals, Binary,
    Objective, Constraint, minimize,
    value, SolverFactory
)

# -------------------------------------------------------------
# 1. Load Excel sheets
# -------------------------------------------------------------
excel_file = "MINLP_GIS_InputData_MidAtlantic.xlsx"

demand_df = pd.read_excel(excel_file, sheet_name="Demand nodes")
sites_df  = pd.read_excel(excel_file, sheet_name="Candidate_sites")
dist_df   = pd.read_excel(excel_file, sheet_name="Distance_Matrix")

# Strip whitespace from all column names (fixes 'OBJECTID ' issue)
demand_df.columns = demand_df.columns.str.strip()
sites_df.columns  = sites_df.columns.str.strip()
dist_df.columns   = dist_df.columns.str.strip()

print("DEMAND COLUMNS:", demand_df.columns.tolist())
print("SITE COLUMNS:",   sites_df.columns.tolist())
print("RAW DISTANCE COLUMNS (first 20):", dist_df.columns.tolist()[:20])

# -------------------------------------------------------------
# 2. Clean demand data
# -------------------------------------------------------------
# Make Waste_proxy numeric and drop invalid
demand_df["Waste_proxy"] = pd.to_numeric(demand_df["Waste_proxy"], errors="coerce")
demand_df = demand_df.dropna(subset=["Waste_proxy"])

# Use OBJECTID as the demand node id
demand_df = demand_df.set_index("OBJECTID")
demand_df.index = demand_df.index.astype(int)

# -------------------------------------------------------------
# 3. Clean candidate site data
# -------------------------------------------------------------
# OBJECTID as site id
sites_df = sites_df.set_index("OBJECTID")
sites_df.index = sites_df.index.astype(int)

# Suitability numeric; if missing, drop those rows or assign default
if "Suitability" in sites_df.columns:
    sites_df["Suitability"] = pd.to_numeric(sites_df["Suitability"], errors="coerce")
    sites_df = sites_df.dropna(subset=["Suitability"])
else:
    sites_df["Suitability"] = 1.0

# -------------------------------------------------------------
# 4. Clean distance matrix
# -------------------------------------------------------------
# First column should be demand_id; rename if necessary
first_col = dist_df.columns[0]
if first_col != "demand_id":
    dist_df = dist_df.rename(columns={first_col: "demand_id"})

# Drop junk columns like "(blank)" or "Grand Total"
drop_cols = []
for col in dist_df.columns:
    if isinstance(col, str):
        c = col.strip().lower()
        if c in ["(blank)", "grand total"]:
            drop_cols.append(col)
dist_df = dist_df.drop(columns=drop_cols, errors="ignore")

# Demand id as integer index
dist_df["demand_id"] = pd.to_numeric(dist_df["demand_id"], errors="coerce")
dist_df = dist_df.dropna(subset=["demand_id"])
dist_df["demand_id"] = dist_df["demand_id"].astype(int)
dist_df = dist_df.set_index("demand_id")

print("CLEAN DISTANCE COLUMNS (first 20):", dist_df.columns.tolist()[:20])
print("Distance matrix shape:", dist_df.shape)

# -------------------------------------------------------------
# 5. Align demand ids between demand_df and dist_df
# -------------------------------------------------------------
common_demand_ids = sorted(set(demand_df.index).intersection(dist_df.index))
demand_df = demand_df.loc[common_demand_ids]
dist_df   = dist_df.loc[common_demand_ids]

print("Number of demand nodes used:", len(common_demand_ids))

# -------------------------------------------------------------
# 6. Align candidate sites between sites_df and dist_df
# -------------------------------------------------------------
# Distance matrix columns ≈ site ids (often numeric 1,2,3,...)
site_cols_from_dist = []

for col in dist_df.columns:
    # Skip any non-numeric weirdness
    try:
        cid = int(col)
        site_cols_from_dist.append(cid)
    except Exception:
        pass

# Keep only numeric columns
valid_dist_cols = [c for c in dist_df.columns if str(c).isdigit()]
dist_df = dist_df[valid_dist_cols]

# Convert back to integer site IDs
site_cols_from_dist = [int(c) for c in dist_df.columns]

# Intersection of candidate sites from GIS and from distance matrix
candidate_sites = sorted(set(site_cols_from_dist).intersection(sites_df.index))

print("Total site ids in distance matrix:", len(site_cols_from_dist))
print("Candidate sites present in sites_df:", len(candidate_sites))

# -------------------------------------------------------------
# 7. Pick a manageable subset of sites (for solver performance)
# -------------------------------------------------------------
TOP_N_SITES = 30  # tweak this based on your machine

if len(candidate_sites) <= TOP_N_SITES:
    selected_sites = candidate_sites
else:
    subset_sites_df = sites_df.loc[candidate_sites]
    subset_sites_df = subset_sites_df.sort_values("Suitability", ascending=False)
    selected_sites = subset_sites_df.head(TOP_N_SITES).index.tolist()

print("Selected sites for optimization:", len(selected_sites))

# Filter to selected sites
sites_df = sites_df.loc[selected_sites]
# Distance matrix columns are currently strings '1','2',... in valid_dist_cols.
# We want them as integers matching selected_sites.
dist_df.columns = [int(c) for c in dist_df.columns]
dist_df = dist_df[selected_sites]

# -------------------------------------------------------------
# 8. Build sets and parameters
# -------------------------------------------------------------
J = list(demand_df.index)   # demand node ids
I = list(sites_df.index)    # site ids

demand_series = demand_df["Waste_proxy"]
total_demand = float(demand_series.sum())
print("Total demand (sum of Waste_proxy):", total_demand)

if len(I) > 0:
    avg_demand_per_site = total_demand / len(I)
else:
    avg_demand_per_site = 0.0

# Simple capacity assumption: each site can treat twice the average share
capacity_per_site = 2.0 * avg_demand_per_site
capacity_dict = {i: capacity_per_site for i in I}

# Fixed cost based on suitability (higher suitability → cheaper)
max_suit = float(sites_df["Suitability"].max())
min_suit = float(sites_df["Suitability"].min())
base_fixed = 1_000_000.0

fixed_cost = {}
for i in I:
    s = float(sites_df.loc[i, "Suitability"])
    if max_suit > min_suit:
        norm = (s - min_suit) / (max_suit - min_suit)
    else:
        norm = 0.5
    fixed_cost[i] = base_fixed * (1.5 - 0.5 * norm)

# Variable processing cost per unit (same for now)
var_cost = {i: 50.0 for i in I}

# Transport cost per km factor
transport_cost_per_km = 1.0

# Distance dictionary d_ij
distance = {}
for j in J:
    for i in I:
        if i in dist_df.columns:
            d_val = dist_df.loc[j, i]
            try:
                distance[(j, i)] = float(d_val)
            except Exception:
                distance[(j, i)] = 1e6
        else:
            distance[(j, i)] = 1e6

# -------------------------------------------------------------
# 9. Define Pyomo model
# -------------------------------------------------------------
model = ConcreteModel()

model.I = Set(initialize=I)
model.J = Set(initialize=J)

model.D = Param(model.J, initialize={j: float(demand_series.loc[j]) for j in J})
model.C = Param(model.I, initialize=capacity_dict)
model.F = Param(model.I, initialize=fixed_cost)
model.Cvar = Param(model.I, initialize=var_cost)
model.Dist = Param(model.J, model.I, initialize=distance)

model.x = Var(model.J, model.I, domain=NonNegativeReals)
model.y = Var(model.I, domain=Binary)

# Objective
def total_cost_rule(m):
    fixed_part = sum(m.F[i] * m.y[i] for i in m.I)
    flow_part = sum(
        (m.Cvar[i] + transport_cost_per_km * m.Dist[j, i]) * m.x[j, i]
        for j in m.J for i in m.I
    )
    return fixed_part + flow_part

model.TotalCost = Objective(rule=total_cost_rule, sense=minimize)

# Constraints

def demand_satisfaction_rule(m, j):
    return sum(m.x[j, i] for i in m.I) == m.D[j]

model.DemandSatisfaction = Constraint(model.J, rule=demand_satisfaction_rule)

def capacity_rule(m, i):
    return sum(m.x[j, i] for j in m.J) <= m.C[i] * m.y[i]

model.CapacityLimit = Constraint(model.I, rule=capacity_rule)

def logical_rule(m, j, i):
    return m.x[j, i] <= m.D[j] * m.y[i]

model.LogicalLink = Constraint(model.J, model.I, rule=logical_rule)

# -------------------------------------------------------------
# 10. Solve
# -------------------------------------------------------------
solver = SolverFactory("highs")   # make sure HiGHS is installed & on PATH
results = solver.solve(model, tee=True)

print("Solver termination:", results.solver.termination_condition)
print("Solver status:", results.solver.status)

# -------------------------------------------------------------
# 11. Extract results
# -------------------------------------------------------------
opened_sites = [i for i in model.I if value(model.y[i]) > 0.5]

print("Number of opened facilities:", len(opened_sites))
print("Opened facility IDs:", opened_sites)

selected_sites_df = sites_df.copy()
selected_sites_df["open"] = [1 if i in opened_sites else 0 for i in selected_sites_df.index]
selected_sites_df.to_csv("selected_sites.csv", index_label="site_id")

flow_records = []
for j in model.J:
    for i in model.I:
        x_val = value(model.x[j, i])
        if x_val is not None and x_val > 1e-6:
            flow_records.append({
                "demand_id": j,
                "site_id": i,
                "flow": x_val
            })

flows_df = pd.DataFrame(flow_records)
flows_df.to_csv("flows_demand_to_site.csv", index=False)

print("Results written to:")
print("  - selected_sites.csv")
print("  - flows_demand_to_site.csv")
