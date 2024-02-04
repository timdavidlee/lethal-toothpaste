import json
import re

from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd

REMASK = re.compile(r"\"([A-Za-z0-9 ]+)\"")
ISNUM = re.compile(r"^[0-9]+")


@dataclass
class AdCampaign:
    name: str
    category: str
    adview_rate: float
    adclick_rate: float
    prodview_rate: float
    purchase_rate: float
    start_date: str
    duration_days: int


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
    datadir = "/Users/timlee/myrepos/lethal-toothpaste/lethaltoothpaste/adcampaigns/llm_gen/"  # noqa
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
            r1, r2, r3, r4, r5 = np.random.rand(5)
            c = AdCampaign(
                name=n,
                category=cat,
                adview_rate=r1,
                adclick_rate=r2,
                prodview_rate=r3,
                purchase_rate=r4,
                start_date=np.random.choice(datelist),
                duration_days=np.random.randint(14, 90)
            )
            campaigns.append(c)
    return campaigns


def main():
    adnames = load_adnames()
    campaigns = create_adcampaign_rates(adnames)
    print(campaigns)
