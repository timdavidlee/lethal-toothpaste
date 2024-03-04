# lethal-toothpaste

## Where did the name come from?

I used a random phrase generator online. This one might have actually been directly from `github` repo name generator, I don't remember

## What is this repo?

I got sick of looking for **teaching datasets**:

- Datasets that were modest in size (100-1000) rows
- More realistic data (real countries, real products, real money values)
- That had some interesting patterns built into them

And while kaggle.com had a number of interesting datasets, (some of which serve as some base information):

- many of them were too confusing to students
- or lacking some basic information to make it easily readable

## What techniques are used

**Logical Explainable Patterns:** Instead of uniform random everything, datasets are "logically generated" for example:

- Generate an item catalog with the following:
    - proposity to view (p = 0 - 1.0)
    - proposity to buy (p = 0 - 1.0)
- generate a list of random users
- give each user a top 2-5 shopping categories, with a uneven distribution (say 0.5, 0.3, 0.2)
- Use that probability profile to select some items from a catalogy under that category

**Realistic Profiles:**

- For example, randomly selecting a country is based on the GDP distribution
- Selecting a random quantity order follows an inverse relationship:
    ```
    1 items = 1 / 1 = 1.0
    2 items = 1 / 2 = 0.5
    5 items = 1 / 5 = 0.2
    ...
    50 items = 1 / 50 = 0.02
    Then normalize all of them so total == 1
    ```

**Trend Control**

- Once a group of transactions are created, we can apply `sin` and `cos` curves against the data to ensure dips in the middle of the year, or up-trends near the end.

