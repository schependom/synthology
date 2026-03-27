"""
Random-base + forward-chaining baseline generator.

Generates train/val/test splits by sampling random base ABox facts from ontology
schema declarations and materializing inferred facts with owlrl.
"""

import csv
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from owlrl import DeductiveClosure, OWLRL_Semantics
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS

Triple = Tuple[URIRef, URIRef, URIRef]


class FCBaselineGenerator:
    """Forward-chaining baseline with random base fact generation."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        if cfg.dataset.seed is not None:
            random.seed(cfg.dataset.seed)

        self.ontology_path = cfg.ontology.path
        self.output_dir = cfg.dataset.output_dir

        # Load ontology graph once; reused for all samples.
        self.ontology_graph = Graph()
        self.ontology_graph.parse(self.ontology_path)

        self.classes = self._discover_classes()
        self.object_properties = self._discover_object_properties()
        self.domains, self.ranges = self._discover_domain_range()

        if not self.classes:
            raise ValueError("No classes discovered in ontology; cannot generate baseline data.")
        if not self.object_properties:
            raise ValueError("No object properties discovered in ontology; cannot generate baseline data.")

        logger.info(
            f"Discovered schema: {len(self.classes)} classes, {len(self.object_properties)} object properties, "
            f"{len(self.domains)} domains, {len(self.ranges)} ranges"
        )

    def _discover_classes(self) -> List[URIRef]:
        classes = set()
        for cls in self.ontology_graph.subjects(RDF.type, OWL.Class):
            if isinstance(cls, URIRef):
                classes.add(cls)
        for cls in self.ontology_graph.subjects(RDF.type, RDFS.Class):
            if isinstance(cls, URIRef):
                classes.add(cls)

        # Filter built-ins for cleaner generated data.
        blocked_prefixes = (str(OWL), str(RDF), str(RDFS))
        return sorted([c for c in classes if not str(c).startswith(blocked_prefixes)], key=str)

    def _discover_object_properties(self) -> List[URIRef]:
        props = set()
        for prop in self.ontology_graph.subjects(RDF.type, OWL.ObjectProperty):
            if isinstance(prop, URIRef):
                props.add(prop)

        blocked_prefixes = (str(OWL), str(RDF), str(RDFS))
        return sorted([p for p in props if not str(p).startswith(blocked_prefixes)], key=str)

    def _discover_domain_range(self) -> Tuple[Dict[URIRef, Set[URIRef]], Dict[URIRef, Set[URIRef]]]:
        domains: Dict[URIRef, Set[URIRef]] = {}
        ranges: Dict[URIRef, Set[URIRef]] = {}

        for prop, _, cls in self.ontology_graph.triples((None, RDFS.domain, None)):
            if isinstance(prop, URIRef) and isinstance(cls, URIRef):
                domains.setdefault(prop, set()).add(cls)

        for prop, _, cls in self.ontology_graph.triples((None, RDFS.range, None)):
            if isinstance(prop, URIRef) and isinstance(cls, URIRef):
                ranges.setdefault(prop, set()).add(cls)

        return domains, ranges

    @staticmethod
    def _clean_name(uri: URIRef) -> str:
        s = str(uri)
        if "#" in s:
            return s.split("#")[-1]
        return s.split("/")[-1]

    def _individual_uri(self, split_name: str, sample_index: int, ind_index: int) -> URIRef:
        base = self.cfg.generator.namespace_base
        return URIRef(f"{base}{split_name}_s{sample_index}_i{ind_index}")

    def _sample_typed_individuals(
        self, split_name: str, sample_index: int
    ) -> Tuple[List[URIRef], Dict[URIRef, Set[URIRef]], Set[Triple]]:
        n_individuals = random.randint(self.cfg.generator.min_individuals, self.cfg.generator.max_individuals)
        individuals = [self._individual_uri(split_name, sample_index, i) for i in range(n_individuals)]

        ind_types: Dict[URIRef, Set[URIRef]] = {}
        base_type_triples: Set[Triple] = set()

        min_types = self.cfg.generator.min_types_per_individual
        max_types = self.cfg.generator.max_types_per_individual

        for ind in individuals:
            n_types = random.randint(min_types, max_types)
            sampled = set(random.sample(self.classes, k=min(n_types, len(self.classes))))
            ind_types[ind] = sampled
            for cls in sampled:
                base_type_triples.add((ind, RDF.type, cls))

        return individuals, ind_types, base_type_triples

    def _pick_individual_for_constraints(
        self,
        candidates: List[URIRef],
        required_types: Optional[Set[URIRef]],
        ind_types: Dict[URIRef, Set[URIRef]],
    ) -> Optional[URIRef]:
        if not required_types:
            return random.choice(candidates)

        valid = [ind for ind in candidates if required_types.issubset(ind_types.get(ind, set()))]
        if not valid:
            return None
        return random.choice(valid)

    def _sample_relation_triples(
        self,
        individuals: List[URIRef],
        ind_types: Dict[URIRef, Set[URIRef]],
    ) -> Set[Triple]:
        relation_triples: Set[Triple] = set()

        fixed_base_relations = self.cfg.generator.get("base_relations_per_sample", None)
        if fixed_base_relations is not None:
            n_target = int(fixed_base_relations)
        else:
            n_target = random.randint(self.cfg.generator.min_base_relations, self.cfg.generator.max_base_relations)
        max_attempts = max(100, n_target * 25)
        attempts = 0

        while len(relation_triples) < n_target and attempts < max_attempts:
            attempts += 1
            prop = random.choice(self.object_properties)

            if self.cfg.generator.enforce_domain_range:
                subj = self._pick_individual_for_constraints(
                    individuals,
                    self.domains.get(prop),
                    ind_types,
                )
                obj = self._pick_individual_for_constraints(
                    individuals,
                    self.ranges.get(prop),
                    ind_types,
                )
                if subj is None or obj is None:
                    continue
            else:
                subj = random.choice(individuals)
                obj = random.choice(individuals)

            if not self.cfg.generator.allow_reflexive and subj == obj:
                continue

            relation_triples.add((subj, prop, obj))

        return relation_triples

    def _materialize(self, base_triples: Set[Triple]) -> Set[Triple]:
        g = Graph()
        for t in self.ontology_graph:
            g.add(t)

        for triple in base_triples:
            g.add(triple)

        before = set(g)
        DeductiveClosure(
            OWLRL_Semantics,
            rdfs_closure=self.cfg.materialization.rdfs_closure,
            axiomatic_triples=self.cfg.materialization.axiomatic_triples,
            datatype_axioms=self.cfg.materialization.datatype_axioms,
        ).expand(g)
        after = set(g)

        inferred = after - before
        return {
            (s, p, o)
            for s, p, o in inferred
            if isinstance(s, URIRef) and isinstance(p, URIRef) and isinstance(o, URIRef)
        }

    def _filter_instance_triples(
        self,
        triples: Set[Triple],
        individuals: Set[URIRef],
    ) -> Set[Triple]:
        filtered: Set[Triple] = set()

        for s, p, o in triples:
            # Keep class memberships for generated individuals.
            if p == RDF.type and s in individuals and o in set(self.classes):
                filtered.add((s, p, o))
                continue

            # Keep object-property facts over generated individuals.
            if p in set(self.object_properties) and s in individuals and o in individuals:
                filtered.add((s, p, o))

        return filtered

    def _random_negative_from_positive(
        self,
        triple: Triple,
        individuals: List[URIRef],
    ) -> Triple:
        s, p, o = triple

        if p == RDF.type:
            # Corrupt subject for memberships.
            return random.choice(individuals), p, o

        # Corrupt subject or object for relation triples.
        if random.random() < 0.5:
            return random.choice(individuals), p, o
        return s, p, random.choice(individuals)

    def _make_rows_for_sample(
        self,
        sample_id: str,
        individuals: List[URIRef],
        base_triples: Set[Triple],
        inferred_triples: Set[Triple],
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        facts_rows: List[Dict[str, str]] = []
        targets_rows: List[Dict[str, str]] = []

        positive_meta: Dict[Triple, Tuple[str, int]] = {}

        for t in sorted(base_triples, key=lambda x: (str(x[0]), str(x[1]), str(x[2]))):
            positive_meta[t] = ("base_fact", 0)

        for t in sorted(inferred_triples, key=lambda x: (str(x[0]), str(x[1]), str(x[2]))):
            if t not in positive_meta:
                positive_meta[t] = ("inf_root", 1)

        for (s, p, o), (fact_type, hops) in positive_meta.items():
            row = {
                "sample_id": sample_id,
                "subject": self._clean_name(s),
                "predicate": "rdf:type" if p == RDF.type else self._clean_name(p),
                "object": self._clean_name(o),
                "label": 1,
                "truth_value": "true",
                "type": fact_type,
                "hops": hops,
                "corruption_method": "none",
            }
            targets_rows.append(row)

            if fact_type == "base_fact":
                facts_rows.append(
                    {
                        "sample_id": sample_id,
                        "subject": row["subject"],
                        "predicate": row["predicate"],
                        "object": row["object"],
                    }
                )

        # Random corruption negatives (baseline style).
        positives = list(positive_meta.items())
        n_negatives = int(len(positives) * self.cfg.neg_sampling.ratio)
        max_attempts = max(1, self.cfg.neg_sampling.max_attempts_per_negative) * max(1, n_negatives)

        individuals_set = set(individuals)
        positive_set = set(positive_meta.keys())
        negatives: Set[Triple] = set()

        attempts = 0
        while len(negatives) < n_negatives and attempts < max_attempts and positives:
            attempts += 1
            (pos_triple, (pos_type, pos_hops)) = random.choice(positives)
            neg = self._random_negative_from_positive(pos_triple, individuals)

            if neg in positive_set or neg in negatives:
                continue

            s, p, o = neg
            # Ensure negatives remain in the same instance universe.
            if s not in individuals_set:
                continue
            if p != RDF.type and o not in individuals_set:
                continue

            negatives.add(neg)
            neg_type = "neg_base_fact" if pos_type == "base_fact" else "neg_inf_root"
            targets_rows.append(
                {
                    "sample_id": sample_id,
                    "subject": self._clean_name(s),
                    "predicate": "rdf:type" if p == RDF.type else self._clean_name(p),
                    "object": self._clean_name(o),
                    "label": 0,
                    "truth_value": "false",
                    "type": neg_type,
                    "hops": pos_hops,
                    "corruption_method": "random",
                }
            )

        return facts_rows, targets_rows

    def _write_split_csv(
        self, split_dir: Path, facts_rows: List[Dict[str, str]], targets_rows: List[Dict[str, str]]
    ) -> None:
        split_dir.mkdir(parents=True, exist_ok=True)

        facts_path = split_dir / "facts.csv"
        targets_path = split_dir / "targets.csv"

        with open(facts_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "subject", "predicate", "object"])
            writer.writeheader()
            writer.writerows(facts_rows)

        with open(targets_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sample_id",
                    "subject",
                    "predicate",
                    "object",
                    "label",
                    "truth_value",
                    "type",
                    "hops",
                    "corruption_method",
                ],
            )
            writer.writeheader()
            writer.writerows(targets_rows)

        logger.success(
            f"Saved split {split_dir.name}: {len(facts_rows)} facts, {len(targets_rows)} targets to {split_dir}"
        )

    def _generate_split(self, split_name: str, n_samples: int) -> None:
        facts_rows: List[Dict[str, str]] = []
        targets_rows: List[Dict[str, str]] = []
        retained_samples = 0
        split_fact_cap = self.cfg.dataset.get(f"{split_name}_fact_cap", None)
        split_target_cap = self.cfg.dataset.get(f"{split_name}_target_cap", None)

        for i in range(n_samples):
            sample_id = str(1000 + i)

            individuals, ind_types, type_triples = self._sample_typed_individuals(split_name, i)
            rel_triples = self._sample_relation_triples(individuals, ind_types)
            base_triples = type_triples | rel_triples

            inferred = self._materialize(base_triples)

            individual_set = set(individuals)
            filtered_base = self._filter_instance_triples(base_triples, individual_set)
            filtered_inferred = self._filter_instance_triples(inferred, individual_set) - filtered_base

            sample_facts, sample_targets = self._make_rows_for_sample(
                sample_id=sample_id,
                individuals=individuals,
                base_triples=filtered_base,
                inferred_triples=filtered_inferred,
            )

            # Keep complete samples when applying cap so facts/targets remain aligned.
            if (
                split_fact_cap is not None
                and retained_samples > 0
                and (len(facts_rows) + len(sample_facts)) > split_fact_cap
            ):
                break
            if (
                split_target_cap is not None
                and retained_samples > 0
                and (len(targets_rows) + len(sample_targets)) > split_target_cap
            ):
                break

            facts_rows.extend(sample_facts)
            targets_rows.extend(sample_targets)
            retained_samples += 1

            if (i + 1) % 100 == 0 or (i + 1) == n_samples:
                logger.info(f"{split_name}: generated {i + 1}/{n_samples} samples")

        split_dir = Path(self.output_dir) / split_name

        if split_fact_cap is not None and len(facts_rows) > split_fact_cap:
            facts_rows = facts_rows[:split_fact_cap]

        if split_target_cap is not None and len(targets_rows) > split_target_cap:
            targets_rows = targets_rows[:split_target_cap]

        if split_fact_cap is not None or split_target_cap is not None:
            fact_sample_ids = {row["sample_id"] for row in facts_rows}
            target_sample_ids = {row["sample_id"] for row in targets_rows}
            retained_sample_ids = fact_sample_ids & target_sample_ids
            facts_rows = [row for row in facts_rows if row.get("sample_id") in retained_sample_ids]
            targets_rows = [row for row in targets_rows if row.get("sample_id") in retained_sample_ids]

        self._write_split_csv(split_dir, facts_rows, targets_rows)

        if split_fact_cap is not None or split_target_cap is not None:
            logger.info(
                "Applied caps for %s: fact_cap=%s, target_cap=%s, kept_samples=%s, kept_facts=%s, kept_targets=%s"
                % (
                    split_name,
                    split_fact_cap,
                    split_target_cap,
                    len({row["sample_id"] for row in facts_rows} & {row["sample_id"] for row in targets_rows}),
                    len(facts_rows),
                    len(targets_rows),
                )
            )

    def generate(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        self._generate_split("train", self.cfg.dataset.n_train)
        self._generate_split("val", self.cfg.dataset.n_val)
        self._generate_split("test", self.cfg.dataset.n_test)


REPO_ROOT = os.environ.get("SYNTHOLOGY_ROOT", "../../../..")


@hydra.main(version_base=None, config_path=f"{REPO_ROOT}/configs/fc_baseline", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Running FC baseline generator with configuration:\n" + OmegaConf.to_yaml(cfg))

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=cfg.logging.level, colorize=True)

    generator = FCBaselineGenerator(cfg)
    generator.generate()


if __name__ == "__main__":
    main()
