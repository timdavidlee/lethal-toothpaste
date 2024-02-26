from datetime import timedelta
from uuid import uuid4

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from functools import partial

from lethaltoothpaste.datasets.purchase_join_dataset.util import (
    generate_items_df, 
    generate_rand_ints,
    PurchasingUser,
    UserJourney,
    generate_user_journeys,
)

from lethaltoothpaste.names.generate_names import generate_names
from lethaltoothpaste.stats.profiles import gen_country_profile
from multiprocessing.pool import ThreadPool


def journeys2df(journey_collector: list[UserJourney]) -> pd.DataFrame:
    row_collector = []
    for jc in journey_collector:
        for e in jc.events:
            row_collector.append(dict(
                journey_id=jc.journey_id,
                user_id=jc.user_id,
                status=jc.status,
                item_id=e[0],
                action=e[1],
                timestamp=e[2],
            ))
    trxn_df = pd.DataFrame(row_collector)
    return trxn_df


def mt_generate_journeys(users, cat2item, item2item, item2viewprob, item2purchaseprob):
    single_func = partial(
        generate_user_journeys,
        cat2item=cat2item,
        item2item=item2item,
        item2viewprob=item2viewprob,
        item2purchaseprob=item2purchaseprob,
    )

    quantities = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60])
    quant_prob = (1 / quantities) ** 0.5
    quant_prob = quant_prob / quant_prob.sum()

    qty = np.random.choice(quantities, p=quant_prob, size=len(users))
    tups = list(zip(users, qty))
    n_count = len(tups)

    journey_collector = []
    with ThreadPool(10) as pool:
        for result in tqdm(pool.imap(single_func, tups), total=n_count):
            journey_collector.extend(result)
    return journey_collector


def trxn2shipping_df(trxn_df, countries, country_probs):
    trxn_df["datetime"] = pd.to_datetime(trxn_df["timestamp"], unit="s")
    trxn_df["week_datetime"] = trxn_df["datetime"].dt.round("d").map(lambda x: x - timedelta(days=x.dayofweek) + timedelta(days=8))

    shipping_df = trxn_df.groupby(["user_id", "week_datetime"], as_index=False).agg(
        journey_id=pd.NamedAgg("journey_id", "unique"),
        status=pd.NamedAgg("status", "first"),
    )
    rowct_shipping = shipping_df.shape[0]
    shipping_df["shipping_id"] = [uuid4() for _ in range(rowct_shipping)]
    shipping_df["shipping_destination_country"] = np.random.choice(countries, p=country_probs, size=rowct_shipping)
    shipping_df = shipping_df.explode("journey_id").drop(["user_id"], axis=1)
    return shipping_df


def generate_upload_files(n_users: int = 1000):
    sample_dir = Path("./sample_files/")

    usernames = generate_names(n_users)
    user_ids = generate_rand_ints(size=n_users)
    countries, country_probs = gen_country_profile()
    user_countries = np.random.choice(countries, p=country_probs, size=n_users)

    SELECT_CATS = [
        "bakeware & accessories",
        "bathroom accessories",
        "bedding sets & components",
        "car parts",
        "casual & dress shoes",
        "cooking utensils",
        "decorative pillows & cushions",
        "hand tools",
        "home decor",
        "mobile phone accessories",
        "nails, screws & fixings",
        "office supplies",
        "paint & supplies",
    ]

    users = [PurchasingUser(j, k, m, categories=SELECT_CATS) for j, k, m in zip(user_ids, usernames, user_countries)]
    userfile = sample_dir / "user_table.tsv"
    pd.DataFrame([u.as_dict() for u in users]).to_csv(userfile, index=False, sep="\t")
    logger.info(f"{len(users):,}: {userfile}")

    items_df = generate_items_df(SELECT_CATS, force=False)
    items_df["item_price"] = items_df["item_price"].map(lambda x: float(x.replace(",", "")))
    itemsfile = sample_dir / "items.tsv"
    items_df[["item_id", "item_price", "item_currency", "item_brand_name", "item_title"]].to_csv(itemsfile, index=False, sep="\t")
    logger.info(f"{items_df.shape[0]:,}: {itemsfile}")

    item2categoryfile = sample_dir / "item_category.tsv"
    items_df[["item_id", "item_category_name"]].to_csv(item2categoryfile, index=False, sep="\t")
    logger.info(f"{items_df.shape[0]:,}: {item2categoryfile}")

    cat2item = items_df.groupby("item_category_name")["item_id"].agg(set)

    item2item = {k: v for k, v in items_df[["item_id", "top10_sim_id"]].values}
    item2viewprob = {k: v for k, v in items_df[["item_id", "view_prob"]].values}
    item2purchaseprob = {k: v for k, v in items_df[["item_id", "purchase_prob"]].values}

    journey_collector = mt_generate_journeys(users, cat2item, item2item, item2viewprob, item2purchaseprob)
    trxn_df = journeys2df(journey_collector)
    trxn_file = sample_dir / "transactions.tsv"
    trxn_df[["journey_id", "user_id", "item_id", "action", "timestamp"]].to_csv(trxn_file, index=False, sep="\t")
    logger.info(f"{trxn_df.shape[0]:,}: {trxn_file}")

    shipping_df = trxn2shipping_df(trxn_df, countries, country_probs)
    shipping_file = sample_dir / "shipping.tsv"
    shipping_df.rename(columns={"week_datetime": "ship_date"}).to_csv(shipping_file, index=False, sep="\t")
    logger.info(f"{shipping_df.shape[0]:,}: {shipping_file}")


generate_upload_files()