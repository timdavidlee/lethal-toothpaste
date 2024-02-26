import json
import re
import uuid
from functools import partial
from collections import defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REMASK = re.compile(r"\"([A-Za-z0-9 ]+)\"")
ISNUM = re.compile(r"^[0-9]+")


@dataclass
class AdCampaign:
    name: str
    category: str
    adview_rate: float
    adclick_rate: float
    prodview_rate: float
    cartadd_rate: float
    purchase_rate: float
    start_date: str
    duration_days: int


@dataclass
class PersonProfile:
    name: str
    top_categories: list[str]
    probs: list[float]


class UserAdStage(StrEnum):
    ADVIEW = "ad-view"
    ADCLICK = "ad-click"
    PRODVIEW = "view-product"
    PRODADD = "add-cart-product"
    PURCHASE = "product-purchase"


def load_json(fl: Path):
    with open(fl, "r") as f:
        return json.load(f)


def clean_names(frag: str):
    f = frag.replace("\"", "")
    f = ISNUM.sub("", f)
    f = f.replace(".", "").strip()
    return f


def load_adnames():
    adnames = defaultdict(list)
    datadir = "/Users/timlee/myrepos/lethal-toothpaste/lethaltoothpaste/adcampaigns/llm_gen/"
    dd = Path(datadir)
    for fl in dd.glob("*.json"):
        data = load_json(fl)
        cat = data["category"]
        adnames[cat] = [clean_names(r) for r in data["ad_names"]]
    return adnames


def create_adcampaign_rates(adnames: dict):
    campaigns = []
    this_year = datetime.now().year
    datelist = pd.date_range(
        start=f"{this_year}-01-01",
        end=f"{this_year}-12-31",
        freq="d"
    )
    datelist = [r.strftime("%Y-%m-%d") for r in datelist]
    for cat in adnames:
        names = adnames[cat]
        for n in names:
            r1, r2, r3, r4, r5 = np.round(np.random.rand(5), 4)
            c = AdCampaign(
                name=n,
                category=cat,
                adview_rate=r1,
                adclick_rate=r2,
                prodview_rate=r3,
                cartadd_rate=r4,
                purchase_rate=r5,
                start_date=np.random.choice(datelist),
                duration_days=np.random.randint(14, 90)
            )
            campaigns.append(c)
    return campaigns


def generate_names(n: int = 500_000):
    with open("./lethaltoothpaste/names/boy_names.txt", "r") as f:
        boy_first_names = f.read().split("\n")
        boy_first_names = [b.lower() for b in boy_first_names]

    with open("./lethaltoothpaste/names/girl_names.txt", "r") as f:
        girl_first_names = f.read().split("\n")
        girl_first_names = [g.lower() for g in girl_first_names]

    with open("./lethaltoothpaste/names/last_names.txt", "r") as f:
        last_names = f.read().split("\n")
        last_names = [l.lower() for l in last_names]

    last_names = np.random.choice(last_names, size=n)
    first_names = np.random.choice(boy_first_names + girl_first_names, size=n)
    return ["{} {}".format(f, l) for f, l in zip(first_names, last_names)]


def generate_ppl_profiles(all_categories, n: int = 20_000):
    ppl_names = generate_names(n=n)

    ppl_profiles = []
    for p in tqdm(ppl_names):
        n_top_categories = np.random.randint(4, 7)
        top_categories = np.random.choice(all_categories, size=n_top_categories, replace=False)
        probs = np.random.rand(n_top_categories)**2
        probs = probs / probs.sum()

        ppl = PersonProfile(
            name=p,
            top_categories=top_categories,
            probs=probs
        )
        ppl_profiles.append(ppl)
    return ppl_profiles


def generate_trxn(profile: PersonProfile, campaigns: list[AdCampaign]):
    filename = "/tmp/lp/name-{}.parquet".format(profile.name.replace(" ", "_"))
    df = pd.DataFrame([asdict(c) for c in campaigns])

    adview = df["adview_rate"] > np.random.rand(800)
    adclick = pd.concat([df["adclick_rate"] > np.random.rand(800), adview], axis=1).min(axis=1)
    prod_view = pd.concat([df["prodview_rate"] > np.random.rand(800), adclick], axis=1).min(axis=1)
    add_cart = pd.concat([df["cartadd_rate"] > np.random.rand(800), prod_view], axis=1).min(axis=1)
    purchase = pd.concat([df["purchase_rate"] > np.random.rand(800), add_cart], axis=1).min(axis=1)

    df["adview"] = adview * 1
    df["adclick"] = adclick * 1
    df["prod_view"] = prod_view * 1
    df["add_cart"] = add_cart * 1
    df["purchase"] = purchase * 1

    m = df["category"].isin(profile.top_categories)
    inscope_df = df[m].copy()

    date_collector = []
    for row in df.loc[m, ["start_date", "duration_days"]].values:
        start_date, duration = row
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        daydiff = timedelta(days=np.random.randint(0, duration))
        rec_date = start_date + daydiff
        date_collector.append(rec_date.strftime("%Y-%m-%d"))

    inscope_df["transaction_dt"] = date_collector
    inscope_df["session_id"] = [str(uuid.uuid4()) for _ in range(inscope_df.shape[0])]
    inscope_df = inscope_df.melt(id_vars=["name", "category", "transaction_dt", "session_id"], value_vars=["adview", "adclick", "prod_view", "add_cart", "purchase"])
    inscope_df = inscope_df[inscope_df["value"] != 0].copy()
    inscope_df["username"] = profile.name
    inscope_df.to_parquet(filename)


adnames = load_adnames()
all_categories = list(adnames.keys())
campaigns = create_adcampaign_rates(adnames)
ppl_profiles = generate_ppl_profiles(all_categories)