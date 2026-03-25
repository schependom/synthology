import math
import random
import re
from typing import Optional

from loguru import logger

from owl2bench.bfs import build_bfs_subgraph_samples

UNIVERSITY_ID_RES = [
    re.compile(r"\bU(\d+)"),
    re.compile(r"\bUniversity(\d+)\b", re.IGNORECASE),
    re.compile(r"\.University(\d+)", re.IGNORECASE),
]


def split_samples(all_triples: list[tuple], num_samples: int, base_sample_id: int) -> list[tuple[int, list[tuple]]]:
    chunk_size = math.ceil(len(all_triples) / max(1, num_samples))
    samples = []
    for i in range(0, len(all_triples), chunk_size):
        chunk = all_triples[i : i + chunk_size]
        samples.append((base_sample_id + len(samples), chunk))
    random.shuffle(samples)
    return samples


def extract_university_id(value: str) -> Optional[int]:
    for regex in UNIVERSITY_ID_RES:
        match = regex.search(value)
        if match:
            return int(match.group(1))
    return None


def partition_triples_by_university(
    triples: list[tuple[str, str, str]],
) -> tuple[dict[int, list[tuple[str, str, str]]], list[tuple[str, str, str]]]:
    grouped: dict[int, list[tuple[str, str, str]]] = {}
    unassigned: list[tuple[str, str, str]] = []

    for s, p, o in triples:
        uid = extract_university_id(s)
        if uid is None:
            uid = extract_university_id(o)

        if uid is None:
            unassigned.append((s, p, o))
            continue

        grouped.setdefault(uid, []).append((s, p, o))

    return grouped, unassigned


def build_samples_by_university(
    base_clean: list[tuple[str, str, str]],
    inferred_clean: list[tuple[str, str, str, int]],
    base_sample_id: int,
) -> tuple[
    list[tuple[int, list[tuple[str, str, str]]]],
    dict[int, list[tuple[str, str, str, int]]],
]:
    inferred_triples_only = [(s, p, o) for (s, p, o, _hops) in inferred_clean]

    base_by_uid, base_unassigned = partition_triples_by_university(base_clean)
    inferred_by_uid_raw, inferred_unassigned = partition_triples_by_university(inferred_triples_only)

    uid_set = sorted(set(base_by_uid.keys()) | set(inferred_by_uid_raw.keys()))
    samples: list[tuple[int, list[tuple[str, str, str]]]] = []
    inferred_by_sample: dict[int, list[tuple[str, str, str, int]]] = {}

    inferred_hops_index: dict[tuple[str, str, str], list[int]] = {}
    for s, p, o, hops in inferred_clean:
        inferred_hops_index.setdefault((s, p, o), []).append(hops)

    for uid in uid_set:
        sid = base_sample_id + uid
        sample_base = base_by_uid.get(uid, [])
        samples.append((sid, sample_base))

        sample_inferred: list[tuple[str, str, str, int]] = []
        for s, p, o in inferred_by_uid_raw.get(uid, []):
            hops_list = inferred_hops_index.get((s, p, o), [1])
            hops = hops_list.pop(0)
            sample_inferred.append((s, p, o, hops))
        inferred_by_sample[sid] = sample_inferred

    if samples and base_unassigned:
        logger.warning(
            "University partitioning: {} base triples had no university prefix and were assigned to first sample",
            len(base_unassigned),
        )
        first_sid = samples[0][0]
        first_triples = samples[0][1]
        first_triples.extend(base_unassigned)
        samples[0] = (first_sid, first_triples)

    if samples and inferred_unassigned:
        logger.warning(
            "University partitioning: {} inferred triples had no university prefix and were assigned to first sample",
            len(inferred_unassigned),
        )
        first_sid = samples[0][0]
        for s, p, o in inferred_unassigned:
            inferred_by_sample[first_sid].append((s, p, o, 1))

    random.shuffle(samples)
    return samples, inferred_by_sample


def build_partitioned_samples(
    partition_mode: str,
    base_clean: list[tuple[str, str, str]],
    inferred_clean: list[tuple[str, str, str, int]],
    base_sample_id: int,
    num_samples: int,
    bfs_sample_count: int,
    bfs_max_individuals_per_sample: int,
) -> tuple[
    list[tuple[int, list[tuple[str, str, str]]]],
    dict[int, list[tuple[str, str, str, int]]],
]:
    if partition_mode == "university_prefix":
        samples, inferred_by_sample = build_samples_by_university(base_clean, inferred_clean, base_sample_id)
        logger.info("Partitioned graph into {} university-based samples", len(samples))
        return samples, inferred_by_sample

    if partition_mode == "bfs_subgraphs":
        samples, inferred_by_sample = build_bfs_subgraph_samples(
            base_clean,
            inferred_clean,
            base_sample_id,
            bfs_sample_count,
            bfs_max_individuals_per_sample,
        )
        logger.info(
            "Partitioned graph into {} BFS subgraph samples (target_count={}, max_individuals={})",
            len(samples),
            bfs_sample_count,
            bfs_max_individuals_per_sample,
        )
        return samples, inferred_by_sample

    samples = split_samples(base_clean, num_samples, base_sample_id)
    inferred_by_sample = {sid: [] for sid, _ in samples}
    if samples:
        inferred_by_sample[samples[0][0]] = inferred_clean
    logger.warning(
        "Using fallback chunk partitioning mode='{}'; this may cut graph chains.",
        partition_mode,
    )
    return samples, inferred_by_sample


def compute_split_counts(
    total_samples: int,
    split_ratios: dict[str, float],
    require_multiple_graphs_per_csv: bool,
) -> tuple[int, int, int]:
    train_ratio = split_ratios.get("train", 0.8)
    val_ratio = split_ratios.get("val", 0.1)
    test_ratio = split_ratios.get("test", 0.1)
    total_ratio = train_ratio + val_ratio + test_ratio

    if total_samples <= 0 or total_ratio <= 0:
        return 0, 0, 0

    train_count = int(total_samples * (train_ratio / total_ratio))
    val_count = int(total_samples * (val_ratio / total_ratio))
    test_count = total_samples - train_count - val_count

    if require_multiple_graphs_per_csv and total_samples >= 6:
        minimum = 2
        counts = {"train": train_count, "val": val_count, "test": test_count}

        for name in ["train", "val", "test"]:
            while counts[name] < minimum:
                donor = max(counts, key=lambda k: counts[k])
                if counts[donor] <= minimum:
                    break
                counts[donor] -= 1
                counts[name] += 1

        train_count = counts["train"]
        val_count = counts["val"]
        test_count = counts["test"]

    return train_count, val_count, test_count
