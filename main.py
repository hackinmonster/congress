from bridges.bridges import Bridges
from bridges.graph_adj_list import GraphAdjList
from bridges.color import Color
import pandas as pd
from pathlib import Path
from itertools import combinations
from collections import defaultdict

DATA_DIR = Path("/home/eric/Documents/bridges/congress/data/csv")

def compute_degrees(adj):
    #adj: dict[int, set[int]] or dict[str, set[str]]
    return {u: len(neighs) for u, neighs in adj.items()}

def color_for_degree(d, dmin, dmax):
    #simple blue→red ramp
    if dmax == dmin:
        t = 0.0
    else:
        t = (d - dmin) / (dmax - dmin)
    r = int(255 * t)
    b = int(255 * (1 - t))
    return Color(r, 80, b)

def visualize_degree_centrality(adj, labels, parties, assignment_id=100):

    deg = compute_degrees(adj)
    dmin, dmax = (min(deg.values()) if deg else 0, max(deg.values()) if deg else 1)

    g = GraphAdjList()
    for u in adj:
        g.add_vertex(u, labels.get(u, str(u)))
        g.get_vertex(u).size = 5 + (20 * (deg[u] - dmin) / (dmax - dmin or 1))
        #node color by party
        p = (parties.get(u) or "").strip()
        p_up = p.upper()
        if p_up in ("R", "REPUBLICAN"):
            g.get_vertex(u).color = "red"
        elif p_up in ("D", "DEMOCRAT", "DEMOCRATIC"):
            g.get_vertex(u).color = "blue"
        elif p_up in ("L", "LIBERTARIAN"):
            g.get_vertex(u).color = "yellow"
        elif p_up in ("G", "GREEN"):
            g.get_vertex(u).color = "green"
        elif p_up in ("I", "INDEPENDENT"):
            g.get_vertex(u).color = "gray"
        else:
            g.get_vertex(u).color = "lightgray"
        g.get_vertex(u).label = f"{labels.get(u, u)} ({deg[u]})"

    #add undirected edges once
    for u, neighs in adj.items():
        for v in neighs:
            if u <= v:  # prevent duplicates if adj is symmetric
                g.add_edge(u, v)
                lv = g.get_link_visualizer(u, v)
                lv.thickness = 0.5
                lv.color = "purple"

    bridges = Bridges(assignment_id, "efackelm", "1036607715596")
    bridges.set_title("Degree Centrality: Hubs (size/color ∝ degree)")
    bridges.set_description("Nodes sized/colored by degree. Label shows name (degree).")
    bridges._element_label_flag = True
    bridges.set_data_structure(g)
    bridges.visualize()

def build_adj_and_labels_from_csvs(
    data_dir: Path,
    max_bills: int = 500,
    min_common_bills: int = 6,
    chamber_filter: str | None = None,
    max_cosponsor_count: int | None = None,
):
    """Build an undirected co-sponsorship adjacency and labels from CSVs.

    - Reads people.csv (for names) and sponsors.csv (bill_id, people_id).
    - Limits to the first max_bills by bill_id order for faster visualization.
    - Forms edges only for pairs with at least min_common_bills shared bills (default 6).
    """
    people = pd.read_csv(data_dir / "people.csv")
    sponsors = pd.read_csv(data_dir / "sponsors.csv")

    #filter out committee entries if present in people.csv (committee_id != 0)
    people = people.fillna(0)
    people["committee_id"] = people["committee_id"].astype(int)
    people = people[people["committee_id"] == 0]

    #optional filter by chamber (House/Rep or Senate/Sen)
    if chamber_filter:
        cf = str(chamber_filter).strip().lower()
        target_role = None
        if cf in ("house", "rep", "h"):
            target_role = "Rep"
        elif cf in ("senate", "sen", "s"):
            target_role = "Sen"
        if target_role is not None:
            people = people[people["role"].astype(str).str.strip().eq(target_role)]

    #labels and party maps
    labels = {int(r.people_id): str(r.name) for r in people[["people_id", "name"]].itertuples(index=False)}
    parties = {int(r.people_id): str(r.party) for r in people[["people_id", "party"]].itertuples(index=False)}

    # Keep only sponsor rows that correspond to real people (not committees)
    sponsors = sponsors.merge(people[["people_id"]], on="people_id", how="inner")

    #limit bills
    bill_ids = (
        sponsors[["bill_id"]]
        .drop_duplicates()
        .sort_values("bill_id")
        .head(max_bills)
        ["bill_id"].tolist()
    )
    sponsors_small = sponsors[sponsors["bill_id"].isin(bill_ids)]

    #optionally filter by maximum number of cosponsors per bill
    if max_cosponsor_count is not None:
        cosz = (
            sponsors_small[["bill_id", "people_id"]]
            .drop_duplicates()
            .groupby("bill_id")["people_id"].nunique()
        )
        allowed = set(cosz[cosz <= int(max_cosponsor_count)].index.tolist())
        sponsors_small = sponsors_small[sponsors_small["bill_id"].isin(allowed)]

    #count pair co-sponsorships across bills
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    for bid, group in sponsors_small.groupby("bill_id"):
        people_ids = sorted(set(int(x) for x in group["people_id"].tolist()))
        for u, v in combinations(people_ids, 2):
            a, b = (u, v) if u < v else (v, u)
            pair_counts[(a, b)] += 1

    #build filtered adjacency by threshold
    adj: dict[int, set[int]] = {}
    for (u, v), cnt in pair_counts.items():
        if cnt >= min_common_bills:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

    #ensure every labeled node that appears in adj has a label
    for u in adj.keys():
        labels.setdefault(u, str(u))
        parties.setdefault(u, "")

    return adj, labels, parties


if __name__ == "__main__":
    CHAMBER = "House"
    MAX_COSPONSOR_COUNT = 50
    adj, labels, parties = build_adj_and_labels_from_csvs(
        DATA_DIR,
        max_bills=3000,
        min_common_bills=30,
        chamber_filter=CHAMBER,
        max_cosponsor_count=MAX_COSPONSOR_COUNT,
    )
    visualize_degree_centrality(adj, labels, parties, assignment_id=10000)