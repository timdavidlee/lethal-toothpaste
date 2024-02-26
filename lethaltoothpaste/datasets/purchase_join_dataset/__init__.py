
import json
import os

import numpy as np
import pandas as pd
import torch
from uuid import uuid4
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from lethaltoothpaste.names.generate_names import generate_names


def generate_rand_ints(size: int, low: int = 100, high: int = 10_000_000):
    rem = size
    rand_ints = pd.Series(np.random.randint(low, high, rem))
    rand_ints = rand_ints.drop_duplicates()
    rem = size - rand_ints.shape[0]

    while rem > 0:
        z = pd.Series(np.random.randint(low, high, rem * 2))
        rand_ints = pd.concat([rand_ints, z], axis=0)
        rand_ints = rand_ints.drop_duplicates()
        rem = size - rand_ints.shape[0]

    rand_ints = rand_ints.head(size)
    return rand_ints.values


def load_item_list():
    df = pd.read_csv("/Users/timlee/Documents/data/amazon-global-store-us/Souq_Saudi_Amazon_Global_Store_US.csv")  # noqa
    df.insert(0, "item_id", generate_rand_ints(df.shape[0]))
    return df


def generate_similarity(items_df: pd.DataFrame, force: bool = False):
    file = "/Users/timlee/Documents/data/amazon-global-store-us/limited-with-similarity.json"   # noqa
    if os.path.exists(file) and (not force):
        logger.info("cached version found + loaded")
        return pd.read_json(file, orient="records", lines=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(device)
    logger.info("encoding")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vecs = model.encode(
        items_df["item_title"],
        show_progress_bar=True,
        batch_size=128
    )

    logger.info("calculating sim")
    batch_size = 128
    score_collector = []
    for k in tqdm(range(0, items_df.shape[0] - 1, batch_size)):
        subset = vecs[k: k + batch_size, :]
        if subset.shape[0] == 0:
            continue
        scores = cosine_similarity(subset, vecs)
        score_collector.append(scores)

    cossim = np.concatenate(score_collector, axis=0)

    logger.info("calculating top scores")
    scores, inds = torch.topk(torch.FloatTensor(cossim), k=10)

    ind2item_id = items_df["item_id"].values
    items_df["top10_sim_index"] = inds.numpy().tolist()
    items_df["top10_sim_id"] = items_df["top10_sim_index"].map(
        lambda x: ind2item_id[x]
    )
    items_df.to_json(file, orient="records", lines=True)
    logger.info(file)
    return items_df


def generate_items_df(
    inscope_category_list: list[str]
):
    items_df = load_item_list()

    orig_ct = items_df.shape[0]
    mask = items_df["item_category_name"].isin(inscope_category_list)
    items_df = items_df[mask].reset_index(drop=True)

    rowct =  items_df.shape[0]
    items_df["view_prob"] = np.random.rand(rowct)
    items_df["purchase_prob"] = np.random.rand(rowct)

    logger.info("{:,} -> {:,}".format(orig_ct, items_df.shape[0]))

    _ = generate_similarity(items_df, force=False)


class PurchasingUser:
    def __init__(self, user_id: int, name: str, categories: list[str]):
        self.name = name
        self.user_id = user_id
        self.cat_probs = None
        self.cat_list = None
        self.np_random = np.random.RandomState()
        self._total_categories = categories

    def _gen_profile(self):
        self.cat_probs = self.np_random.rand(np.random.randint(2, 5))
        self.cat_list = self.np_random.choice(
            self._total_categories,
            size=self.cat_probs.shape[0]
        )

    def get_rand_cat(self):
        return self.np_random.choice(self.cat_list, p=self.cat_probs)


class UserJourney:
    START_EPOCH = 1672537374
    END_EPOCH = 1703986974

    def __init__(self, user_id: str):
        self.user_id = int(user_id)
        self.journey_id = str(uuid4())
        self.start_epoch = int(np.random.randint(self.START_EPOCH, self.END_EPOCH))
        self.current_time = self.start_epoch
        self.events = []

    def add_step(self, item_id: int, event_type: str):
        self.current_time += int(np.random.randint(30_000, 300_000))
        self.events.append(
            [int(item_id), event_type, self.current_time]
        )

    def __str__(self):
        dictdata = {
            "user_id": self.user_id,
            "journey_id": self.journey_id,
            "events": self.events
        }
        return json.dumps(dictdata, indent=2)


def generate_user_journeys(
    user_id: int,
    n_journeys: int,
    item2item: dict,
    item2viewprob: dict[int, float],
    item2purchaseprob: dict[int, float],
):
    journey_collector = []
    for j in tqdm(range(n_journeys)):
        journey = Journey(user_id=1)
        continue_flag = True
    
        while continue_flag:
            item_id = np.random.choice(list(item2item.keys()))
            p1, p2 = np.random.rand(2)
        
            if p1 < item2viewprob[item_id]:
                journey.add_step(item_id, "search")
                item_id = np.random.choice(item2item[item_id])
                continue
            
            journey.add_step(item_id, "view")
            if p2 < item2purchaseprob[item_id]:
                journey.add_step(item_id, "search")
                item_id = np.random.choice(item2item[item_id])
                continue
            
            journey.add_step(item_id, "purchase")
            continue_flag = False
        journey_collector.append(journey)
    return journey_collector