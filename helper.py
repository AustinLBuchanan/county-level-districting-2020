import networkx as nx
import geopandas as gpd
import gurobipy as gp
from gurobipy import GRB

# FIPS codes
state_code = {
    'WA': '53', 'DE': '10', 'WI': '55', 'WV': '54', 'HI': '15',
    'FL': '12', 'WY': '56', 'NJ': '34', 'NM': '35', 'TX': '48',
    'LA': '22', 'NC': '37', 'ND': '38', 'NE': '31', 'TN': '47', 'NY': '36',
    'PA': '42', 'AK': '02', 'NV': '32', 'NH': '33', 'VA': '51', 'CO': '08',
    'CA': '06', 'AL': '01', 'AR': '05', 'VT': '50', 'IL': '17', 'GA': '13',
    'IN': '18', 'IA': '19', 'MA': '25', 'AZ': '04', 'ID': '16', 'CT': '09',
    'ME': '23', 'MD': '24', 'OK': '40', 'OH': '39', 'UT': '49', 'MO': '29',
    'MN': '27', 'MI': '26', 'RI': '44', 'KS': '20', 'MT': '30', 'MS': '28',
    'SC': '45', 'KY': '21', 'OR': '41', 'SD': '46'
}

def draw_districts( G, districts, df ):

    k = len(districts)
    assignment = [ -1 for u in G.nodes ]
    labeling = { i : j for j in range(k) for i in districts[j] }
    node_with_this_geoid = { G.nodes[i]["GEOID10"] : i for i in G.nodes }

    # pick a position u in the dataframe
    for u in range(len(G.nodes)):
        geoID = df['GEOID10'][u]

        # what node in G has the same geoid?
        i = node_with_this_geoid[geoID]

        # position u in the dataframe should be given 
        #  the same district label as i has in labeling
        assignment[u] = labeling[i]

    # Now add the assignments to a column of the dataframe and map it
    df['assignment'] = assignment
    my_fig = df.plot(column='assignment').get_figure()
    

def add_2020_populations( G, state ):
    
    code = state_code[state]
    
    # Open the file and create a dictionary that maps "<county> County" to its 2020 (DAS-noised) population.
    with open('data\\population_2020.csv') as file:
        # columns of this csv are: state-code (0), county-name(1), population(2)
        pop_by_name_2020 = { line.split(',')[1] : int( line.split(',')[2] ) for line in file if line.split(',')[0] == code }

    # create a dictionary that maps each node to its DAS-noised population
    pop_by_node_2020 = { i : None for i in G.nodes }
    for i in G.nodes:
        name = G.nodes[i]['NAMELSAD10']
        
        # https://en.wikipedia.org/wiki/Talk:LaSalle_Parish,_Louisiana
        if name == 'La Salle Parish':
            name = 'LaSalle Parish'
            G.nodes[i]['NAMELSAD10'] = name
        
        pop_by_node_2020[i] = pop_by_name_2020[name]
        
    nx.set_node_attributes(G, pop_by_node_2020, "POP20")
    

def check_feasibility( G, L, U, k ):
    
    # check for overt infeasibility
    max_pop = max( G.nodes[i]['POP20'] for i in G.nodes )
    if max_pop > U:
        print("This state is not county-level feasible.")
        return False
    
    # create model 
    m = gp.Model()
    m.Params.OutputFlag = 0

    # create x[i,j] variable which equals one when county i 
    #    is assigned to (the district centered at) county j
    x = m.addVars(G.nodes, G.nodes, vtype=GRB.BINARY)
    
    # What are the graph-based distances between vertices?
    n = G.number_of_nodes()
    dist = { (i,j) : n for i in G.nodes for j in G.nodes }
    for i in G.nodes:
        d = nx.shortest_path_length( G, source=i )
        for j in d.keys():
            dist[i,j] = d[j]
    
    # Let's minimize the moment of inertia: d^2 * p * x
    # (The choice of objective doesn't really matter when testing feasibility)
    m.setObjective( gp.quicksum( dist[i,j] * dist[i,j] * G.nodes[i]['POP20'] * x[i,j] for i in G.nodes for j in G.nodes), GRB.MINIMIZE )
    
    # add constraints saying that each county i is assigned to one district
    m.addConstrs( gp.quicksum( x[i,j] for j in G.nodes ) == 1 for i in G.nodes)

    # add constraint saying there should be k district centers
    m.addConstr( gp.quicksum( x[j,j] for j in G.nodes ) == k )

    # add constraints that say: if j roots a district, then its population is between L and U.
    m.addConstrs( gp.quicksum( G.nodes[i]['POP20'] * x[i,j] for i in G.nodes) >= L * x[j,j] for j in G.nodes )
    m.addConstrs( gp.quicksum( G.nodes[i]['POP20'] * x[i,j] for i in G.nodes) <= U * x[j,j] for j in G.nodes )

    # add coupling constraints saying that if i is assigned to j, then j is a center.
    m.addConstrs( x[i,j] <= x[j,j] for i in G.nodes for j in G.nodes )

    # Add contiguity constraints
    DG = nx.DiGraph(G)

    # Add variable f[j,u,v] which equals the amount of flow (originally from j) that is sent across arc (u,v)
    f = m.addVars( DG.nodes, DG.edges, vtype=GRB.CONTINUOUS)
    M = DG.number_of_nodes() - 1

    # Add constraint saying that node j cannot receive flow of its own type
    m.addConstrs( gp.quicksum( f[j,u,j] for u in DG.neighbors(j) ) == 0 for j in DG.nodes )

    # Add constraints saying that node i can receive flow of type j only if i is assigned to j
    m.addConstrs( gp.quicksum( f[j,u,i] for u in DG.neighbors(i)) <= M * x[i,j] for i in DG.nodes for j in DG.nodes if i != j )

    # If i is assigned to j, then i should consume one unit of j flow. 
    #    Otherwise, i should consume no units of j flow.
    m.addConstrs( gp.quicksum( f[j,u,i] - f[j,i,u] for u in DG.neighbors(i)) == x[i,j] for i in DG.nodes for j in DG.nodes if i != j )

    # For computational niceties, impose that districts are centered at their most populous county
    for i in G.nodes:
        for j in G.nodes:
            if G.nodes[i]['POP20'] > G.nodes[j]['POP20']:
                x[i,j].UB = 0
    
    # Tighten tolerances because of numerical instability, then solve.
    m.Params.IntFeasTol = 1.e-9
    m.Params.FeasibilityTol = 1.e-9
    m.optimize()
    
    if m.status == GRB.INFEASIBLE:
        print("This state is not county-level feasible.")
        return False
    
    if m.status == GRB.OPTIMAL:
        print("Found a feasible districting plan, as follows:")
        
        # retrieve the districts and their populations
        centers = [j for j in G.nodes if x[j,j].x > 0.5 ]
        districts = [ [i for i in G.nodes if x[i,j].x > 0.5] for j in centers]
        district_counties = [ [ G.nodes[i]["NAME10"] for i in districts[j] ] for j in range(k)]
        district_populations = [ sum(G.nodes[i]["POP20"] for i in districts[j]) for j in range(k) ]

        # print district info
        for j in range(k):
            print("District",j+1,"has population",district_populations[j],"and contains counties",district_counties[j])

        return districts
