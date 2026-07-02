import os
import math
import random
import copy
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# =================================================================
# 1. DATA STRUCTURES FOR THE GENETIC ALGORITHM
# =================================================================

class Activity:
    def __init__(self, id_, duration):
        self.id = id_
        self.duration = duration
        self.successors = []
        self.requirements = {}  # {(skill, level): demand}

class Resource:
    def __init__(self, id_):
        self.id = id_
        self.skills = {}  # skill -> level

class Chromosome:
    def __init__(self, AL, PL):
        self.AL = AL  # Activity List
        self.PL = PL  # Priority Rule List
        self.fitness = None
        self.schedule = None
        self.assignments = None  # {act_id: [res_id1, ...]}

class Project:
    def __init__(self):
        self.activities = {}
        self.resources = {}
        self.predecessors = defaultdict(list)

    def compute_predecessors(self):
        self.predecessors.clear()
        for a in self.activities.values():
            for s in a.successors:
                self.predecessors[s].append(a.id)

# =================================================================
# 2. FILE PARSERS & METRIC EXTRACTORS (CRITICAL INTEGRATION)
# =================================================================

def get_msrcp_metrics(file_path):
    """Extracts structural topological features for the XGBoost ML model."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    def find_section_start(header_text):
        for idx, line in enumerate(raw_lines):
            if header_text in line: return idx + 1
        return -1

    proj_start = find_section_start(r"\* Project Module *")
    base_line_idx = proj_start if proj_start != -1 else 1
    base_tokens = list(map(int, raw_lines[base_line_idx].split()))
    num_activities, num_resources, num_skills, _ = base_tokens

    succ_start_idx = base_line_idx + 3
    num_edges = 0
    successors = {}
    durations = []

    for i in range(num_activities):
        line_tokens = list(map(int, raw_lines[succ_start_idx + i].split()))
        durations.append(line_tokens[0])
        num_succ = line_tokens[1]
        successors[i + 1] = line_tokens[2 : 2 + num_succ]
        num_edges += num_succ

    skill_req_start = find_section_start(r"\* Skill Requirements Module *")
    act_skill_reqs = []
    if skill_req_start != -1:
        for i in range(num_activities):
            act_skill_reqs.append(list(map(int, raw_lines[skill_req_start + i].split())))

    workforce_start = find_section_start(r"\* Workforce Module *")
    resource_capacities = []
    if workforce_start != -1:
        for i in range(num_resources):
            resource_capacities.append(sum(list(map(int, raw_lines[workforce_start + i].split()))))
    else:
        resource_capacities = [1] * num_resources

    graph_density = num_edges / (num_activities * (num_activities - 1) / 2) if num_activities > 1 else 0.0

    in_degree = {i: 0 for i in range(1, num_activities + 1)}
    for u in successors:
        for v in successors[u]:
            if v in in_degree: in_degree[v] += 1
    earliest_finish = {i: 0 for i in range(1, num_activities + 1)}
    queue = [i for i in range(1, num_activities + 1) if in_degree.get(i, 0) == 0]
    while queue:
        u = queue.pop(0)
        ef_u = earliest_finish[u] + (durations[u - 1] if (u - 1) < len(durations) else 0)
        earliest_finish[u] = ef_u
        for v in successors.get(u, []):
            if v in earliest_finish:
                earliest_finish[v] = max(earliest_finish[v], ef_u)
                in_degree[v] -= 1
                if in_degree[v] == 0: queue.append(v)
    critical_path_length = max(earliest_finish.values()) if earliest_finish else 0

    flexibility = 1.0 - (num_edges / (num_activities * num_activities)) if num_activities > 0 else 1.0
    total_skill_demands = sum(1 for req in act_skill_reqs for s in req if s > 0)
    skill_scarcity = total_skill_demands / (num_activities * num_skills) if num_skills > 0 else 0.0
    total_skill_load = sum(sum(req) for req in act_skill_reqs)
    total_capacity = sum(resource_capacities)
    avg_resource_load = total_skill_load / (total_capacity * num_activities) if total_capacity > 0 else 0.0

    return (num_activities, num_resources, num_skills, round(graph_density, 4), critical_path_length, round(flexibility, 4), round(skill_scarcity, 4), round(avg_resource_load, 4))


def parse_msrcp_file(file_path):
    """Parses complete structural properties into core runtime entities for the GA Engine."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    def find_section_start(header_text):
        for idx, line in enumerate(lines):
            if header_text in line: return idx + 1
        return -1

    project = Project()
    proj_start = find_section_start(r"\* Project Module *")
    base_idx = proj_start if proj_start != -1 else 1
    header = list(map(int, lines[base_idx].split()))
    num_activities, num_resources, num_skills, _ = header
    
    skill_req_start = find_section_start(r"\* Skill Requirements Module *")
    skill_lvl_start = find_section_start(r"\* Skill Level Requirements Module *")
    workforce_start = find_section_start(r"\* Workforce Module with Skill Levels *")

    raw_lvl_lines = []
    if skill_lvl_start != -1:
        current_idx = skill_lvl_start
        while current_idx < len(lines) and not lines[current_idx].startswith(r"\*"):
            clean_token_line = list(map(int, lines[current_idx].split()))
            if clean_token_line and clean_token_line != [-1]:
                raw_lvl_lines.append(clean_token_line)
            current_idx += 1

    lvl_line_ptr = 0
    succ_start_idx = base_idx + 3
    for act_idx in range(num_activities):
        act_id = act_idx + 1
        act_line = list(map(int, lines[succ_start_idx + act_idx].split()))
        activity = Activity(id_=act_id, duration=act_line[0])
        activity.successors = act_line[2 : 2 + act_line[1]]
        
        if skill_req_start != -1:
            req_tokens = list(map(int, lines[skill_req_start + act_idx].split()))
            if any(qty > 0 for qty in req_tokens) and lvl_line_ptr < len(raw_lvl_lines):
                lvl_tokens = raw_lvl_lines[lvl_line_ptr]
                lvl_line_ptr += 1
                lvl_ptr = 0
                for s in range(1, num_skills + 1):
                    qty = req_tokens[s - 1]
                    if qty > 0:
                        for _ in range(qty):
                            if lvl_ptr < len(lvl_tokens):
                                key = (s, lvl_tokens[lvl_ptr])
                                activity.requirements[key] = activity.requirements.get(key, 0) + 1
                                lvl_ptr += 1
        project.activities[act_id] = activity

    if workforce_start != -1:
        for res_idx in range(num_resources):
            res_id = res_idx + 1
            res_line = list(map(int, lines[workforce_start + res_idx].split()))
            resource = Resource(id_=res_id)
            for s_idx, lvl in enumerate(res_line, start=1):
                if lvl > 0: resource.skills[s_idx] = lvl
            project.resources[res_id] = resource
            
    return project

# =================================================================
# 3. GENETIC ALGORITHM RESOLUTION ENGINE
# =================================================================

def generate_feasible_AL(project):
    preds_count = {act_id: len(project.predecessors[act_id]) for act_id in project.activities}
    eligible = [act_id for act_id, count in preds_count.items() if count == 0]
    AL = []
    while eligible:
        selected = random.choice(eligible)
        eligible.remove(selected)
        AL.append(selected)
        for succ in project.activities[selected].successors:
            preds_count[succ] -= 1
            if preds_count[succ] == 0: eligible.append(succ)
    return AL

def is_precedence_feasible(project, AL):
    seen = set()
    for act_id in AL:
        for p in project.predecessors[act_id]:
            if p not in seen: return False
        seen.add(act_id)
    return True

def resource_can_execute(resource, skill, level):
    return skill in resource.skills and resource.skills[skill] >= level

def sort_resources(resources, rule):
    resources = list(resources)
    if rule == 1: resources.sort(key=lambda r: len(r.skills))
    elif rule == 2: resources.sort(key=lambda r: -len(r.skills))
    elif rule == 3: resources.sort(key=lambda r: -sum(r.skills.values()))
    return resources

def assign_resources(activity, available_resources, rule):
    slots = []
    for (skill, req_level), count in activity.requirements.items():
        for _ in range(count):
            slots.append((skill, req_level))
            
    if not slots:
        return []

    slots.sort(key=lambda x: x[1], reverse=True)
    assigned = []
    used_ids = set()

    def backtrack(slot_idx):
        if slot_idx == len(slots):
            return True  
            
        target_skill, target_level = slots[slot_idx]
        
        for worker in available_resources:
            if worker.id in used_ids:
                continue
                
            if worker.skills.get(target_skill, 0) >= target_level:
                assigned.append(worker)
                used_ids.add(worker.id)
                
                if backtrack(slot_idx + 1):
                    return True
                    
                assigned.pop()
                used_ids.remove(worker.id)
                
        return False

    if backtrack(0):
        return assigned
    return None

def decode(project, chromosome):
    AL, PL = chromosome.AL, chromosome.PL
    activities = project.activities
    finished = set()
    start_time, finish_time, assignments = {}, {}, {}
    resource_usage = defaultdict(list)
    unscheduled = list(AL)

    while unscheduled:
        progress = False
        candidates = [a for a in unscheduled if all(p in finished for p in project.predecessors[a])]
        if not candidates: return float("inf"), None, None
        candidates.sort(key=lambda x: AL.index(x))
        
        for act_id in candidates:
            preds = project.predecessors[act_id]
            t = max([finish_time[p] for p in preds]) if preds else 0
            act_duration = activities[act_id].duration
            
            while t < 99999:
                available = []
                for r in project.resources.values():
                    busy = False
                    for s, e in resource_usage[r.id]:
                        if act_duration == 0 and s <= t < e: busy = True; break
                        elif act_duration > 0 and not (t + act_duration <= s or t >= e): busy = True; break
                    if not busy: available.append(r)
                
                rule = PL[AL.index(act_id)]
                assigned = assign_resources(activities[act_id], available, rule)
                
                if assigned is not None:
                    start_time[act_id] = t
                    finish_time[act_id] = t + act_duration
                    assignments[act_id] = [r.id for r in assigned]
                    if act_duration > 0:
                        for r in assigned: resource_usage[r.id].append((t, t + act_duration))
                    unscheduled.remove(act_id)
                    finished.add(act_id)
                    progress = True
                    break
                
                all_ends = [e for r_id in resource_usage for s, e in resource_usage[r_id] if e > t]
                t = min(all_ends) if all_ends else t + 1
            if progress: break
        if not progress: return float("inf"), None, None
    return (max(finish_time.values()) if finish_time else 0), (start_time, finish_time), assignments

def evaluate(project, chromosome):
    m, sched, assigns = decode(project, chromosome)
    chromosome.fitness, chromosome.schedule, chromosome.assignments = m, sched, assigns
    return m

def tournament(pop):
    a, b = random.choice(pop), random.choice(pop)
    return a if a.fitness < b.fitness else b

def crossover_AL_precedence(p1, p2, project):
    n = len(p1)
    c1, c2 = random.randint(0, n - 2), random.randint(1, n - 1)
    if c1 > c2: c1, c2 = c2, c1
    child = [None] * n
    child[c1:c2] = p1[c1:c2]
    ptr = 0
    for x in p2:
        if x not in child:
            while child[ptr] is not None: ptr += 1
            child[ptr] = x
    return child if is_precedence_feasible(project, child) else generate_feasible_AL(project)

def run_meta_tuned_genetic_algorithm(project, pop_size, cross_rate, mut_rate, max_generations):
    project.compute_predecessors()
    population = []
    for _ in range(pop_size):
        AL = generate_feasible_AL(project)
        population.append(Chromosome(AL, [random.randint(1, 3) for _ in AL]))
        
    for c in population: evaluate(project, c)
    best = min(population, key=lambda x: x.fitness)

    for gen in range(int(max_generations)):
        
        new_pop = [copy.deepcopy(c) for c in sorted(population, key=lambda x: x.fitness)[:2]]
        while len(new_pop) < pop_size:
            p1, p2 = tournament(population), tournament(population)
            AL = crossover_AL_precedence(p1.AL, p2.AL, project) if random.random() < cross_rate else p1.AL[:]
            PL = p1.PL[:]
            if random.random() < cross_rate:
                c1, c2 = sorted(random.sample(range(len(PL)), 2))
                PL[c1:c2] = p2.PL[c1:c2]
            
            child = Chromosome(AL, PL)
            if random.random() < mut_rate:
                idx1, idx2 = random.sample(range(len(child.AL)), 2)
                child.AL[idx1], child.AL[idx2] = child.AL[idx2], child.AL[idx1]
                if not is_precedence_feasible(project, child.AL): child.AL[idx2], child.AL[idx1] = child.AL[idx1], child.AL[idx2]
            if random.random() < mut_rate:
                child.PL[random.randint(0, len(child.PL) - 1)] = random.randint(1, 3)
                
            evaluate(project, child)
            new_pop.append(child)
        population = new_pop
        curr_best = min(population, key=lambda x: x.fitness)
        if curr_best.fitness < best.fitness: best = copy.deepcopy(curr_best)
        
    return best

# =================================================================
# 4. TRAINING PIPELINE & INFERENCE TARGET EXECUTION (UPDATED)
# =================================================================

INSTANCE_FEATURE_NAMES = ["n_act", "n_res", "n_skill", "density", "cp_length", "flexibility", "skill_rarity", "avg_resource_load"]
GA_FEATURES = ["pop_size", "crossover_rate", "mutation_rate", "max_generation"]

print("-> Step 1: Training dual Predictive Engines on past meta-data logs...")
df = pd.read_excel(os.path.join(BASE_DIR, "GA_Optimization_Results_with_Time.xlsx"))
df = df[df["Status"] == "Success"]

# Note: Adjust column name if your Excel file uses "Runtime" or "Time" instead of "Execution Time"
time_col = "Execution Time" if "Execution Time" in df.columns else ("Runtime" if "Runtime" in df.columns else "Time")

X, y_makespan, y_time = [], [], []
for _, row in df.iterrows():
    try:
        metrics = get_msrcp_metrics(os.path.join(BASE_DIR, f"MSLIB4\MSLIB4\{row['File Name']}"))
        X.append(list(metrics) + [row["Population Size"], row["Crossover Rate"], row["Mutation Rate"], row["Max Generations"]])
        y_makespan.append(row["Optimal Makespan"])
        y_time.append(row["Execution Time (s)"])
    except FileNotFoundError: 
        continue

X = pd.DataFrame(X, columns=INSTANCE_FEATURE_NAMES + GA_FEATURES)
y_makespan = pd.Series(y_makespan)
y_time = pd.Series(y_time)

# Train Model 1: Makespan Predictor
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X, y_makespan, test_size=0.2, random_state=42)
model_makespan = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, objective="reg:squarederror", random_state=42)
model_makespan.fit(X_train_m, y_train_m, eval_set=[(X_test_m, y_test_m)], verbose=False)

# Train Model 2: Execution Time Predictor
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X, y_time, test_size=0.2, random_state=42)
model_time = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, objective="reg:squarederror", random_state=42)
model_time.fit(X_train_t, y_train_t, eval_set=[(X_test_t, y_test_t)], verbose=False)

# import shap

# explainer = shap.TreeExplainer(model_makespan)
# shap_values = explainer.shap_values(X_test_m)

# shap.summary_plot(shap_values, X_test_m)

def recommend_best_ga_params(file_path):
    metrics = get_msrcp_metrics(file_path)
    test_pop_sizes = [20, 40, 60, 80, 100, 200.250]
    test_crossover = [0.6, 0.7, 0.8, 0.9, 0.95]
    test_mutation = [0.02, 0.05, 0.1, 0.15, 0.2]
    test_generations = [100, 500, 750,1000,1500]
    
    candidate_rows, candidate_configs = [], []
    for pop in test_pop_sizes:
        for cross in test_crossover:
            for mut in test_mutation:
                for gen in test_generations:
                    candidate_rows.append(list(metrics) + [pop, cross, mut, gen])
                    candidate_configs.append({"pop_size": pop, "crossover_rate": cross, "mutation_rate": mut, "max_generation": gen})
                    
    scoring_df = pd.DataFrame(candidate_rows, columns=INSTANCE_FEATURE_NAMES + GA_FEATURES)
    
    # Predict both criteria for all candidates
    pred_makespans = model_makespan.predict(scoring_df)
    pred_times = model_time.predict(scoring_df)
    
    # Lexicographical Selection: Priority goes to best makespan. 
    # Adding a minuscule fraction (1e-6) of execution time acts strictly as a tie-breaker.
    composite_scores = pred_makespans + (1e-6 * pred_times)
    best_idx = np.argmin(composite_scores)
    
    return candidate_configs[best_idx]

# =================================================================
# MAIN PIPELINE EXECUTION ENTRY
# =================================================================
if __name__ == "__main__":
    file_name="MSLIB_Set4_1.msrcp"
    target_file =os.path.join(BASE_DIR, f"MSLIB4\MSLIB4\{file_name}")
    
    print("\n-> Step 2: Running XGBoost parameter scanning on the target instance...")
    recommended_config = recommend_best_ga_params(target_file)
    print(f"   [XGBoost Selection Complete] Optimized Parameters: {recommended_config}")
    
    print("\n-> Step 3: Initializing Meta-Tuned Genetic Algorithm using ML-selected configurations...")
    project_instance = parse_msrcp_file(target_file)
    best_solution = run_meta_tuned_genetic_algorithm(
        project=project_instance,
        pop_size=recommended_config["pop_size"],
        cross_rate=recommended_config["crossover_rate"],
        mut_rate=recommended_config["mutation_rate"],
        max_generations=recommended_config["max_generation"]
    )
    
    print(f"\n=======================================================")
    print(f"RUN COMPLETE. Realized Optimal Makespan: {best_solution.fitness}")
    print(f"=======================================================\n")