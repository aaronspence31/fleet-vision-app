from collections import Counter


def classify_main_batch(action_batch):
    return Counter(action_batch).most_common(1)[0][0]
