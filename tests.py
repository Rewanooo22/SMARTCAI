"""Algorithm tests that use the real algorithms.py APIs.

Graphs are built here from data.py so these tests don't import any Streamlit UI module.
Run with: python -m unittest tests
"""
import itertools
import math
import random
import unittest

import networkx as nx
import pandas as pd

from data import (all_nodes, df_existing, df_potential, df_traffic, df_neighborhoods,
                  neighborhoods, facilities, roads, transit_routes)
from algorithms import (a_star, time_dependent_dijkstra, modified_kruskal, MissingMandatoryEdgeError,
                        public_transport_dp, optimize_maintenance, greedy_maintenance,
                        route_population_shares, TRANSFER_PENALTY)

NAME_TO_ID = {row['Name']: row['ID'] for _, row in all_nodes.iterrows()}
ID_TO_NAME = {v: k for k, v in NAME_TO_ID.items()}


def build_emergency_graph():
    """Same construction as emergency_routing.G_emergency."""
    G = nx.Graph()
    for n in list(neighborhoods.values()) + list(facilities.values()):
        G.add_node(n['name'], x=n['x'], y=n['y'])
    for r in roads.values():
        weight = r['distance'] + r['traffic'] * 0.1
        if r['condition'] == 'poor':
            weight *= 1.5
        G.add_edge(r['from'], r['to'], weight=weight)
    return G


def build_traffic_graph():
    """Same construction as urban_planning.build_traffic_graph."""
    G = nx.Graph()
    for _, row in all_nodes.iterrows():
        G.add_node(row['ID'])
    for _, row in df_existing.iterrows():
        G.add_edge(row['FromID'], row['ToID'], distance=row['Distance'])
    return G


def build_all_edges():
    existing = df_existing[['FromID', 'ToID']].assign(Cost=df_existing['Maintenance_Cost'], Type='Existing')
    potential = df_potential[['FromID', 'ToID']].assign(Cost=df_potential['Cost'], Type='Potential')
    return pd.concat([existing, potential])


def road_cost(u, v):
    match = df_existing[((df_existing['FromID'] == u) & (df_existing['ToID'] == v)) |
                        ((df_existing['FromID'] == v) & (df_existing['ToID'] == u))]
    return match['Maintenance_Cost'].iloc[0]


class TestAStar(unittest.TestCase):
    def test_path_endpoints_and_weight(self):
        G = build_emergency_graph()
        path = a_star('Maadi', 'El Salam Hospital', G, priority=False)
        self.assertGreater(len(path), 1)
        self.assertEqual(path[0], 'Maadi')
        self.assertEqual(path[-1], 'El Salam Hospital')
        weight = sum(G[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1))
        self.assertGreater(weight, 0)

    def test_priority_mode_finds_path(self):
        G = build_emergency_graph()
        self.assertEqual(a_star('Maadi', 'El Salam Hospital', G, priority=True)[-1], 'El Salam Hospital')


class TestDijkstra(unittest.TestCase):
    def test_finds_route(self):
        G = build_traffic_graph()
        total_time, path = time_dependent_dijkstra(G, 'Downtown Cairo', 'Nasr City', 'Morning',
                                                   df_traffic, df_existing, NAME_TO_ID, ID_TO_NAME)
        self.assertGreater(len(path), 1)
        self.assertEqual((path[0], path[-1]), ('Downtown Cairo', 'Nasr City'))
        self.assertTrue(0 < total_time < float('inf'))


class TestKruskal(unittest.TestCase):
    def test_mandatory_edge_included_and_tree_valid(self):
        all_edges = build_all_edges()
        mandatory = [(1, 4)]  # exists in potential_roads; not chosen by plain MST cost order
        mst_edges, total_cost = modified_kruskal(all_edges, all_nodes, mandatory)
        pairs = {frozenset((u, v)) for u, v, _ in mst_edges}
        self.assertIn(frozenset((1, 4)), pairs)
        self.assertAlmostEqual(total_cost, sum(c for _, _, c in mst_edges))
        # A spanning forest has no cycles: edges == nodes - components.
        T = nx.Graph()
        T.add_nodes_from(all_nodes['ID'])
        T.add_edges_from((u, v) for u, v, _ in mst_edges)
        self.assertEqual(len(mst_edges), T.number_of_nodes() - nx.number_connected_components(T))

    def test_missing_mandatory_edge_raises(self):
        with self.assertRaises(MissingMandatoryEdgeError) as ctx:
            modified_kruskal(build_all_edges(), all_nodes, [('F9', 1), (1, 4)])
        self.assertEqual(ctx.exception.missing, [('F9', 1)])

    def test_mandatory_cycle_raises(self):
        edges = pd.DataFrame({'FromID': [1, 2, 1], 'ToID': [2, 3, 3], 'Cost': [1, 1, 1]})
        nodes = pd.DataFrame({'ID': [1, 2, 3]})
        with self.assertRaises(MissingMandatoryEdgeError):
            modified_kruskal(edges, nodes, [(1, 2), (2, 3), (1, 3)])


class TestMaintenance(unittest.TestCase):
    def test_optimize_maintenance_budget(self):
        improvement, selected_roads = optimize_maintenance(df_existing, 500)
        self.assertGreaterEqual(improvement, 0)
        self.assertLessEqual(sum(road_cost(u, v) for u, v in selected_roads), 500)

    def test_greedy_maintenance_budget_and_total(self):
        improvement, selected_roads = greedy_maintenance(df_existing, 500)
        self.assertLessEqual(sum(road_cost(u, v) for u, v in selected_roads), 500)
        conditions = {(r.FromID, r.ToID): r.Condition for r in df_existing.itertuples()}
        self.assertEqual(improvement, sum(10 - conditions[p] for p in selected_roads))

    def test_greedy_handles_zero_cost_roads(self):
        df = pd.DataFrame({'FromID': [1, 2], 'ToID': [2, 3], 'Maintenance_Cost': [0, 10], 'Condition': [10, 5]})
        improvement, selected_roads = greedy_maintenance(df, 10)
        self.assertEqual(improvement, 5)
        self.assertEqual(set(selected_roads), {(1, 2), (2, 3)})


def roads_df(rows):
    return pd.DataFrame(rows, columns=['FromID', 'ToID', 'Maintenance_Cost', 'Condition'])


def brute_force_best(df, budget):
    rows = df.to_dict('records')
    best = 0
    for mask in itertools.product([0, 1], repeat=len(rows)):
        chosen = [r for r, m in zip(rows, mask) if m]
        if sum(r['Maintenance_Cost'] for r in chosen) <= budget:
            best = max(best, sum(10 - r['Condition'] for r in chosen))
    return best


class TestOptimizeMaintenanceDP(unittest.TestCase):
    def check(self, df, budget):
        improvement, selected = optimize_maintenance(df, budget)
        by_pair = {(r.FromID, r.ToID): r for r in df.itertuples()}
        self.assertEqual(len(selected), len(set(selected)))
        self.assertLessEqual(sum(by_pair[p].Maintenance_Cost for p in selected), budget)
        self.assertEqual(improvement, sum(10 - by_pair[p].Condition for p in selected))
        self.assertEqual(improvement, brute_force_best(df, budget))
        return improvement, selected

    def test_skipping_road_keeps_accumulated_value(self):
        # Road (2,3) is unaffordable after (1,2); skipping it must keep (1,2)'s gain.
        improvement, selected = self.check(roads_df([[1, 2, 5, 4], [2, 3, 10, 1], [3, 4, 5, 5]]), 10)
        self.assertEqual((improvement, set(selected)), (11, {(1, 2), (3, 4)}))

    def test_beats_greedy_ratio_choice(self):
        # Greedy takes the best ratio (1,2) and then can't afford anything else; optimum is the other two.
        df = roads_df([[1, 2, 6, 3], [2, 3, 5, 5], [3, 4, 5, 5]])
        improvement, selected = self.check(df, 10)
        self.assertEqual(set(selected), {(2, 3), (3, 4)})
        self.assertGreater(improvement, greedy_maintenance(df, 10)[0])

    def test_exact_budget_and_nothing_affordable(self):
        self.check(roads_df([[1, 2, 4, 6], [2, 3, 6, 2]]), 10)
        self.assertEqual(optimize_maintenance(roads_df([[1, 2, 50, 2]]), 10), (0, []))

    def test_random_combinations_match_brute_force(self):
        rng = random.Random(42)
        for _ in range(30):
            rows = [[i, i + 1, rng.randint(0, 20), rng.randint(1, 10)] for i in range(rng.randint(1, 8))]
            self.check(roads_df(rows), rng.randint(0, 40))

    def test_real_data_not_worse_than_greedy(self):
        self.assertGreaterEqual(optimize_maintenance(df_existing, 500)[0], greedy_maintenance(df_existing, 500)[0])


class TestTransferPenalty(unittest.TestCase):
    STATIONS = {'Maadi': (0, 0), 'Downtown Cairo': (1, 0), 'Zamalek': (2, 0), 'Dokki': (1, 1), 'Helwan': (0, 1)}
    ROUTES = {('Maadi', 'Downtown Cairo'): 10, ('Downtown Cairo', 'Zamalek'): 10,
              ('Downtown Cairo', 'Dokki'): 10, ('Downtown Cairo', 'Helwan'): 10}
    LINES = [{'route_id': 1, 'stops': ['Maadi', 'Downtown Cairo', 'Zamalek'], 'frequency': 10, 'capacity': 500},
             {'route_id': 3, 'stops': ['Giza', 'Dokki', 'Downtown Cairo'], 'frequency': 12, 'capacity': 400}]

    def setUp(self):
        self.dp, self.paths, _ = public_transport_dp(self.STATIONS, self.ROUTES, 3, 'Maadi', 1, self.LINES)

    def test_same_line_no_penalty(self):
        self.assertEqual(self.dp['Zamalek'][2], 20)

    def test_line_change_penalized(self):
        self.assertEqual(self.dp['Dokki'][2], 20 + TRANSFER_PENALTY)
        self.assertEqual(self.paths['Dokki'][2], 'Downtown Cairo')

    def test_non_transit_hop_no_penalty(self):
        self.assertEqual(self.dp['Helwan'][2], 20)


class TestTransferTracking(unittest.TestCase):
    """Synthetic network: two lines meet at B; the cheapest arrival at B is on the 'wrong' line."""
    STATIONS = {s: (0, 0) for s in 'SPQBTU'}
    ROUTES = {('S', 'P'): 10, ('P', 'B'): 10,   # line 1: S-P-B-T
              ('S', 'Q'): 10, ('Q', 'B'): 9,    # line 2: S-Q-B-U
              ('B', 'T'): 10, ('B', 'U'): 10}
    LINES = [{'route_id': 1, 'stops': ['S', 'P', 'B', 'T'], 'frequency': 10, 'capacity': 100},
             {'route_id': 2, 'stops': ['S', 'Q', 'B', 'U'], 'frequency': 10, 'capacity': 100}]

    def run_dp(self, lines=None, routes=None):
        return public_transport_dp(self.STATIONS, routes or self.ROUTES, 4, 'S', 1, lines or self.LINES)

    def test_direct_route_multiple_hops(self):
        dp, paths, _ = self.run_dp()
        self.assertEqual(dp['T'][3], 30)  # S-P-B-T entirely on line 1
        self.assertEqual(dp['U'][3], 29)  # S-Q-B-U entirely on line 2

    def test_slower_arrival_kept_to_avoid_transfer(self):
        dp, paths, _ = self.run_dp()
        self.assertEqual(dp['B'][2], 19)            # fastest arrival at B is via line 2
        self.assertEqual(dp['T'][3], 30)            # but T is reached via the line-1 arrival (20 + 10)
        self.assertEqual(paths['T'][3], 'B')
        self.assertLess(dp['T'][3], 19 + 10 + TRANSFER_PENALTY)

    def test_one_transfer_when_unavoidable(self):
        routes = {k: v for k, v in self.ROUTES.items() if k[0] != 'P' and k[1] != 'P'}  # line 1 only from B on
        dp, _, _ = self.run_dp(routes=routes)
        self.assertEqual(dp['T'][3], 19 + 10 + TRANSFER_PENALTY)

    def test_hop_shared_by_two_lines_no_penalty(self):
        lines = [dict(self.LINES[0], stops=['S', 'P', 'B', 'T']),
                 dict(self.LINES[1], stops=['S', 'Q', 'B', 'T'])]  # B-T served by both lines
        dp, _, _ = self.run_dp(lines=lines)
        self.assertEqual(dp['T'][3], 29)


class TestRouteCoverageShares(unittest.TestCase):
    def test_shares_sum_to_unique_population(self):
        stops = {s for r in transit_routes for s in r['stops']}
        unique_pop = df_neighborhoods[df_neighborhoods['Name'].isin(stops)]['Population'].sum()
        self.assertAlmostEqual(sum(route_population_shares(transit_routes)), unique_pop)


class TestPublicTransportDP(unittest.TestCase):
    STATIONS = {'Maadi': (0, 0), 'Zamalek': (1, 1)}
    ROUTES = {('Maadi', 'Zamalek'): 10}

    def run_dp(self, transit, time_slots=6, vehicles=2):
        return public_transport_dp(self.STATIONS, self.ROUTES, time_slots, 'Maadi', vehicles, transit)

    def test_real_routes_coverage_bounded(self):
        _, _, coverage = self.run_dp(transit_routes)
        self.assertGreater(coverage, 0)
        self.assertLess(coverage, df_neighborhoods['Population'].sum())

    def test_capacity_limits_coverage(self):
        route = {'stops': ['Maadi'], 'frequency': 10, 'capacity': 500}
        self.assertAlmostEqual(self.run_dp([route], vehicles=2)[2], 2 * 500 / 10)

    def test_shared_population_not_double_counted(self):
        pop = df_neighborhoods.set_index('Name')['Population']['Zamalek']
        big = {'stops': ['Zamalek'], 'frequency': 10, 'capacity': 10**9}
        self.assertAlmostEqual(self.run_dp([big, dict(big)], vehicles=2)[2], pop / 10)

    def test_headway_in_minutes_vs_hour_slots(self):
        route = {'stops': ['Maadi'], 'frequency': 30, 'capacity': 100}
        self.assertGreater(self.run_dp([route], time_slots=1)[2], 0)       # 30 min fits in 60
        self.assertEqual(self.run_dp([dict(route, frequency=90)], time_slots=1)[2], 0)  # 90 min doesn't

    def test_travel_times(self):
        dp, _, _ = self.run_dp(transit_routes)
        self.assertEqual(dp['Maadi'][0], 0)
        self.assertFalse(math.isinf(min(dp['Zamalek'])))


if __name__ == '__main__':
    unittest.main()
