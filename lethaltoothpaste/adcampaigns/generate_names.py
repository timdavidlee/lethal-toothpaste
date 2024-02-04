import os
import json
from pathlib import Path

from openai import OpenAI
import pandas as pd
import numpy as np

from loguru import logger
from caseconverter import snakecase


def load_openai_api_key():
    with open("/Users/timlee/Dropbox/keys/openai_key.txt", "r") as f:
        return f.read()


def load_products(force: bool = False):
    cache_file = "/tmp/products.feather"
    if os.path.exists(cache_file) and (not force):
        return pd.read_feather(cache_file)
 
    df = pd.read_csv("/Users/timlee/Documents/data/amazon-global-store-us/Souq_Saudi_Amazon_Global_Store_US.csv")  # noqa
    df["item_id"] = df["item_ean"]
    df["popularity"] = np.random.pareto(20, size=df.shape[0])
    df["popularity"] = df["popularity"] \
        / df.groupby("item_category_name")["popularity"].transform("sum")
    df.to_feather(cache_file)
    return df


def main():
    df = load_products(force=True)

    client = OpenAI(
        api_key=load_openai_api_key()
    )

    SAVEDIR = Path("/Users/timlee/myrepos/lethal-toothpaste/lethaltoothpaste/adcampaigns/llm_gen/")  # noqa
    for cat in df["item_category_name"].unique():
        filename = SAVEDIR / "{}.json".format(snakecase(cat))
        if filename.exists():
            logger.info("loaded: {}".format(filename))
            continue

        completion = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": f"Create 20 names of advertising campaigns for this product category: {cat}"  # noqa
            }],
            model="gpt-3.5-turbo"
        )
        names = completion.choices[0].message.content
        names = names.split("\n")
        with open(filename, "w") as f:
            data = {
                "category": cat,
                "ad_names": names,
            }
            f.write(json.dumps(data, indent=2))
        logger.info("saved: {}".format(filename))
