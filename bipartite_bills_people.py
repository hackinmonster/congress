from bridges.bridges import Bridges
from bridges.graph_adj_matrix import GraphAdjMatrix
from bridges.color import Color
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/home/eric/Documents/bridges/congress/data/csv")

PARTY_COLOR = {
    "R": "red",
    "REPUBLICAN": "red",
    "D": "blue",
    "DEMOCRAT": "blue",
    "DEMOCRATIC": "blue",
    "L": "yellow",
    "LIBERTARIAN": "yellow",
    "G": "green",
    "GREEN": "green",
    "I": "gray",
    "INDEPENDENT": "gray",
}


def load_filtered(data_dir: Path, chamber_filter: str | None, max_bills: int):
    people = pd.read_csv(data_dir / "people.csv")
    sponsors = pd.read_csv(data_dir / "sponsors.csv")
    bills = pd.read_csv(data_dir / "bills.csv")

    # Drop committees
    people = people.fillna({"committee_id": 0})
    people["committee_id"] = people["committee_id"].astype(int)
    people = people[people["committee_id"] == 0]

    # Optional chamber filter
    if chamber_filter:
        cf = str(chamber_filter).strip().lower()
        target_role = None
        if cf in ("house", "rep", "h"):
            target_role = "Rep"
        elif cf in ("senate", "sen", "s"):
            target_role = "Sen"
        if target_role is not None:
            people = people[people["role"].astype(str).str.strip().eq(target_role)]

    # Keep only sponsors that are in filtered people set
    sponsors = sponsors.merge(people[["people_id"]], on="people_id", how="inner")

    # Limit number of bills for visualization
    bill_ids = (
        sponsors[["bill_id"]]
        .drop_duplicates()
        .sort_values("bill_id")
        .head(max_bills)["bill_id"].tolist()
    )
    sponsors_small = sponsors[sponsors["bill_id"].isin(bill_ids)]
    bills_small = bills[bills["bill_id"].isin(bill_ids)]

    return people, bills_small, sponsors_small


def build_bipartite_matrix(people: pd.DataFrame, bills: pd.DataFrame, sponsors: pd.DataFrame) -> GraphAdjMatrix:
    g = GraphAdjMatrix()

    # Maps for labels and parties
    ppl_name = {int(r.people_id): str(r.name) for r in people[["people_id", "name"]].itertuples(index=False)}
    ppl_party = {int(r.people_id): str(r.party) for r in people[["people_id", "party"]].itertuples(index=False)}
    bill_title = {int(r.bill_id): str(r.bill_number) for r in bills[["bill_id", "bill_number"]].itertuples(index=False)}

    # Add vertices for people and bills
    people_ids = sorted(set(int(x) for x in sponsors["people_id"].tolist()))
    bill_ids = sorted(set(int(x) for x in sponsors["bill_id"].tolist()))

    for pid in people_ids:
        key = f"P:{pid}"
        g.add_vertex(key, None)
        vis = g.get_visualizer(key)
        # Color by party
        p = (ppl_party.get(pid, "") or "").strip().upper()
        vis.color = PARTY_COLOR.get(p, "lightgray")
        vis.size = 12

    for bid in bill_ids:
        key = f"B:{bid}"
        g.add_vertex(key, None)
        vis = g.get_visualizer(key)
        vis.color = "teal"
        vis.size = 10

    # Now that all vertices are added, set human-readable labels
    for pid in people_ids:
        key = f"P:{pid}"
        g.vertices[key].label = ppl_name.get(pid, str(pid))
    for bid in bill_ids:
        key = f"B:{bid}"
        g.vertices[key].label = bill_title.get(bid, str(bid))

    # Add edges person ↔ bill
    for r in sponsors.itertuples(index=False):
        pid = int(r.people_id)
        bid = int(r.bill_id)
        pk = f"P:{pid}"
        bk = f"B:{bid}"
        # Add both directions for clarity
        g.add_edge(pk, bk, 1)
        g.add_edge(bk, pk, 1)
        try:
            lv1 = g.get_link_visualizer(pk, bk)
            lv1.color = "purple"
            lv1.thickness = 0.6
            lv2 = g.get_link_visualizer(bk, pk)
            lv2.color = "purple"
            lv2.thickness = 0.6
        except Exception:
            pass

    return g


def main():
    CHAMBER = "House"  # "House", "Senate", or None
    MAX_BILLS = 500

    people, bills, sponsors = load_filtered(DATA_DIR, chamber_filter=CHAMBER, max_bills=MAX_BILLS)
    g = build_bipartite_matrix(people, bills, sponsors)

    bridges = Bridges(101, "efackelm", "1036607715596")
    bridges.set_title("Bipartite: Legislators ↔ Bills (Adjacency Matrix)")
    bridges.set_description("People colored by party; bills teal; edges purple. Using GraphAdjMatrix. See tutorial: https://bridgesuncc.github.io/tutorials/Graph_AM.html")
    bridges._element_label_flag = True
    bridges.set_data_structure(g)
    bridges.visualize()


if __name__ == "__main__":
    main()
