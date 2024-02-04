import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from functools import partial

import pandas as pd
import numpy as np
from tqdm import tqdm

from .generate_rates import (
    load_adnames,
    create_adcampaign_rates,
    AdCampaign
)


@dataclass
class PersonProfile:
    name: str
    top_categories: list[str]
    probs: list[float]


def generate_names(n: int = 500_000):
    with open("./lethaltoothpaste/names/boy_names.txt", "r") as f:
        boy_first_names = f.read().split("\n")
        boy_first_names = [b.lower() for b in boy_first_names]

    with open("./lethaltoothpaste/names/girl_names.txt", "r") as f:
        girl_first_names = f.read().split("\n")
        girl_first_names = [g.lower() for g in girl_first_names]

    with open("./lethaltoothpaste/names/last_names.txt", "r") as f:
        last_names = f.read().split("\n")
        last_names = [ln.lower() for ln in last_names]

    last_names = np.random.choice(last_names, size=n)
    first_names = np.random.choice(boy_first_names + girl_first_names, size=n)
    return ["{} {}".format(f, l) for f, l in zip(first_names, last_names)]


def generate_ppl_profiles(all_categories, n: int = 20_000):
    ppl_names = generate_names(n=n)

    ppl_profiles = []
    for p in tqdm(ppl_names):
        n_top_categories = np.random.randint(4, 7)
        top_categories = np.random.choice(
            all_categories,
            size=n_top_categories,
            replace=False)
        probs = np.random.rand(n_top_categories) ** 2
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
    adclick = pd.concat(
        [df["adclick_rate"] > np.random.rand(800), adview],
        axis=1
    ).min(axis=1)
    prod_view = pd.concat(
        [df["prodview_rate"] > np.random.rand(800), adclick],
        axis=1
    ).min(axis=1)
    add_cart = pd.concat(
        [df["cartadd_rate"] > np.random.rand(800), prod_view],
        axis=1
    ).min(axis=1)
    purchase = pd.concat(
        [df["purchase_rate"] > np.random.rand(800), add_cart],
        axis=1
    ).min(axis=1)

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
    inscope_df["session_id"] = [
        uuid.uuid4() for _ in range(inscope_df.shape[0])
    ]
    inscope_df = inscope_df.melt(
        id_vars=["name", "category", "transaction_dt"],
        value_vars=["adview", "adclick", "prod_view", "add_cart", "purchase"]
    )
    inscope_df = inscope_df[inscope_df["value"] != 0].copy()
    inscope_df["username"] = profile.name
    inscope_df.to_parquet(filename)


def generate_trxn_per_person(
    ppl_profiles: list[PersonProfile],
    campaigns: list[AdCampaign]
):
    shutil.rmtree("/tmp/lp/")
    os.makedirs("/tmp/lp/", exist_ok=True)

    collector = []
    futures = []
    singlefunc = partial(generate_trxn, campaigns=campaigns)
    with ThreadPoolExecutor(max_workers=10) as pool:
        for p in ppl_profiles:
            futures.append(pool.submit(singlefunc, profile=p))
        for future in tqdm(as_completed(futures), total=len(ppl_profiles)):
            collector.append(future.result())


def main():
    adnames = load_adnames()
    all_categories = list(adnames.keys())
    campaigns = create_adcampaign_rates(adnames)

    ppl_profiles = generate_ppl_profiles(all_categories)
    generate_trxn_per_person(ppl_profiles, campaigns)
