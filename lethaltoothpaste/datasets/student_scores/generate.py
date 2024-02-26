import pandas as pd
import numpy as np

from lethaltoothpaste.names.generate_names import generate_names


def main():
    n_students = 90
    number_of_exams = 12
    subjects = ["economics", "chemistry", "writing", "math", "physics"]
    test_dates = pd.date_range("2023-01-15", "2023-06-15", freq="7d")
    test_dates = test_dates[:number_of_exams]

    names = generate_names(n=n_students)
    class_ids = np.random.randint(1, 5, size=n_students)

    scores = np.random.rand(number_of_exams, n_students, len(subjects))
    scores = (scores * 100).astype(int)

    score_collector = []
    for j in range(number_of_exams):
        day = test_dates[j]
        df = pd.DataFrame(scores[j, :, :], columns=subjects)
        df["test_date"] = day
        df.insert(0, "class_id", class_ids)
        df.insert(0, "student", names)
        score_collector.append(df)


    df = pd.concat(score_collector, axis=0)
    flatten = pd.melt(
        df,
        id_vars=["test_date", "class_id", "student"],
        var_name="subject",
        value_name="test_score"
    )

    flatten.to_csv("./sample_files/multi-class-student-scores.tsv", index=False, sep="\t")


if __name__ == "__main__":
    main()
