"""
Random-base + forward-chaining baseline generator.

Generates train/val/test splits by sampling random base ABox facts from ontology
schema declarations and materializing inferred facts with Apache Jena.
"""

import csv
import json
import os
import random
import shlex
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS

Triple = Tuple[URIRef, URIRef, URIRef]


class JenaMaterializer:
    """Small bridge to a local Maven-based Jena materializer helper."""

    def __init__(self) -> None:
        self.project_root = Path(os.environ.get("SYNTHOLOGY_ROOT", Path(__file__).resolve().parents[4]))
        udm_java_dir = self.project_root / "apps" / "udm_baseline" / "java"
        udm_java_dir = self.project_root / "apps" / "udm_baseline" / "java"
        self.java_project_dir = udm_java_dir if udm_java_dir.exists() else udm_java_dir
        self.jar_path = self.java_project_dir / "target" / "jena-materializer-1.0.0-shaded.jar"
        self._ensured = False
        self.last_run_timing: Dict[str, Any] = {}

    def _ensure_built(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        pre_exists = self.jar_path.exists()

        if self._ensured and self.jar_path.exists():
            return {
                "status": "already_ensured",
                "jar_preexisting": True,
                "jar_built_in_call": False,
                "elapsed_seconds": time.perf_counter() - t0,
            }
        if self.jar_path.exists():
            self._ensured = True
            return {
                "status": "jar_preexisting",
                "jar_preexisting": True,
                "jar_built_in_call": False,
                "elapsed_seconds": time.perf_counter() - t0,
            }

        if not self.java_project_dir.exists():
            raise FileNotFoundError(
                f"Jena helper project not found at {self.java_project_dir}. Expected apps/udm_baseline/java to exist."
            )

        logger.info("Building Jena helper JAR (one-time setup) via Maven")
        cmd = ["mvn", "-q", "-DskipTests", "package"]
        try:
            subprocess.run(cmd, cwd=self.java_project_dir, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError("Maven (mvn) is not installed or not in PATH.") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to build Jena helper JAR.\nstdout:\n{exc.stdout}\nstderr:\n{exc.stderr}"
            ) from exc

        if not self.jar_path.exists():
            raise RuntimeError(f"Jena helper JAR build completed but {self.jar_path} was not produced.")

        self._ensured = True
        return {
            "status": "built",
            "jar_preexisting": pre_exists,
            "jar_built_in_call": True,
            "elapsed_seconds": time.perf_counter() - t0,
        }

    def materialize(
        self, ontology_path: str, base_triples: Set[Triple], jena_profile: str = "owl_mini", derivation_logging: bool = True
    ) -> Tuple[Set[Triple], Dict[Triple, int]]:
        """Materialize closure with Jena and return all URI-only closure triples and hop depths."""
        run_start = time.perf_counter()
        ensure_stats = self._ensure_built()

        with tempfile.TemporaryDirectory(prefix="jena_mat_") as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            base_path = tmp_dir_path / "base.nt"
            out_path = tmp_dir_path / "closure.nt"

            serialize_start = time.perf_counter()
            base_graph = Graph()
            for triple in base_triples:
                base_graph.add(triple)
            base_graph.serialize(base_path, format="nt", encoding="utf-8")
            serialize_seconds = time.perf_counter() - serialize_start

            cmd = [
                "java",
            ]

            # Allow runtime heap tuning without code changes, e.g.:
            # SYNTHOLOGY_UDM_BASELINE_XMX_MB=4096 uv run invoke exp3-generate-baseline --universities=1
            # Backward compatible fallback: SYNTHOLOGY_JENA_XMX_MB
            xmx_mb = (
                os.environ.get("SYNTHOLOGY_UDM_BASELINE_XMX_MB")
                or os.environ.get("SYNTHOLOGY_JENA_XMX_MB")
                or os.environ.get("SYNTHOLOGY_HEAP_MB")
            )
            if xmx_mb:
                cmd.append(f"-Xmx{xmx_mb}m")

            java_env = os.environ.copy()
            if xmx_mb:
                # Ensure explicit SYNTHOLOGY_JENA_XMX_MB is not shadowed by inherited JVM heap flags.
                for env_key in ("JAVA_TOOL_OPTIONS", "_JAVA_OPTIONS", "JDK_JAVA_OPTIONS"):
                    raw_opts = java_env.get(env_key)
                    if not raw_opts:
                        continue
                    try:
                        tokens = shlex.split(raw_opts)
                    except ValueError:
                        tokens = raw_opts.split()

                    filtered = [tok for tok in tokens if not tok.startswith("-Xmx")]
                    if filtered != tokens:
                        if filtered:
                            java_env[env_key] = " ".join(filtered)
                        else:
                            java_env.pop(env_key, None)
                        logger.warning(
                            "Removed inherited -Xmx option(s) from {} to honor UDM baseline heap setting (Xmx={}m).",
                            env_key,
                            xmx_mb,
                        )

            cmd += [
                "-jar",
                str(self.jar_path),
                str(Path(ontology_path).resolve()),
                str(base_path),
                str(out_path),
                jena_profile,
                "true" if derivation_logging else "false",
            ]

            try:
                java_start = time.perf_counter()
                subprocess.run(cmd, check=True, capture_output=True, text=True, env=java_env)
                java_seconds = time.perf_counter() - java_start
            except FileNotFoundError as exc:
                raise RuntimeError("Java runtime (java) is not installed or not in PATH.") from exc
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"Jena materialization failed.\nstdout:\n{exc.stdout}\nstderr:\n{exc.stderr}"
                ) from exc

            parse_start = time.perf_counter()
            closure_graph = Graph()
            closure_graph.parse(out_path, format="nt")
            parse_seconds = time.perf_counter() - parse_start

            closure = {
                (s, p, o)
                for s, p, o in closure_graph
                if isinstance(s, URIRef) and isinstance(p, URIRef) and isinstance(o, URIRef)
            }

            hop_depths = {}
            hops_path = tmp_dir_path / "hops.csv"
            if hops_path.exists():
                import csv

                with open(hops_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Extract strings directly; row data might already match URIRef strings
                        # Assuming the Java stringifier outputs just the URI.
                        s = URIRef(row["subject"])
                        p = URIRef(row["predicate"])
                        o = URIRef(row["object"])
                        if (s, p, o) in closure:
                            hop_depths[(s, p, o)] = int(row["depth"])

            total_seconds = time.perf_counter() - run_start
            self.last_run_timing = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "ontology_path": str(Path(ontology_path).resolve()),
                "jena_profile": jena_profile,
                "base_triples": len(base_triples),
                "closure_triples": len(closure),
                "ensure": ensure_stats,
                "serialize_seconds": serialize_seconds,
                "java_seconds": java_seconds,
                "total_seconds": total_seconds,
            }

            return closure, hop_depths


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
        self.jena_materializer = JenaMaterializer()
        self._timing_cfg = dict(self.cfg.materialization.get("timing", {}))
        self._timing_enabled = bool(self._timing_cfg.get("enabled", False))
        self._timing_output_dir = Path(str(self._timing_cfg.get("output_dir", "data/exp2/timings")))
        self._timing_run_tag = str(self._timing_cfg.get("run_tag", "udm_jena"))
        self._timing_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._timing_event_count = 0
        self._timing_jsonl_path = self._timing_output_dir / f"{self._timing_run_tag}_jena_events.jsonl"
        self._timing_csv_path = self._timing_output_dir / f"{self._timing_run_tag}_jena_events.csv"

        if self._timing_enabled:
            self._timing_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Timing recorder enabled | run_id={} | run_tag={} | output_dir={} | jsonl={} | csv={} "
                "| recorded_metrics=[event_type,split,sample_id,iteration,working_triples,base_triples,closure_triples,"
                "newly_inferred,serialize_seconds,java_seconds,parse_seconds,total_seconds,iteration_wall_seconds,function_total_seconds]",
                self._timing_run_id,
                self._timing_run_tag,
                self._timing_output_dir.resolve(),
                self._timing_jsonl_path.resolve(),
                self._timing_csv_path.resolve(),
            )

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

    def _append_timing_event(self, event: Dict[str, Any]) -> None:
        if not self._timing_enabled:
            return

        enriched = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": self._timing_run_id,
            "run_tag": self._timing_run_tag,
            "dataset_output_dir": str(Path(self.output_dir).resolve()),
            **event,
        }

        with open(self._timing_jsonl_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(enriched, sort_keys=True) + "\n")

        csv_exists = self._timing_csv_path.exists()
        with open(self._timing_csv_path, "a", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "timestamp_utc",
                "run_id",
                "run_tag",
                "dataset_output_dir",
                "event_type",
                "split",
                "sample_id",
                "iteration",
                "working_triples",
                "base_triples",
                "closure_triples",
                "newly_inferred",
                "java_seconds",
                "serialize_seconds",
                "parse_seconds",
                "total_seconds",
                "iteration_wall_seconds",
                "function_total_seconds",
                "details_json",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not csv_exists:
                writer.writeheader()
            row = {key: enriched.get(key, "") for key in fieldnames}
            row["details_json"] = json.dumps(enriched, sort_keys=True)
            writer.writerow(row)

        self._timing_event_count += 1
        logger.info(
            "Timing event recorded | event_type={} | split={} | sample_id={} | iteration={} "
            "| working={} | base={} | closure={} | new={} | java_s={} | serialize_s={} | parse_s={} | total_s={} "
            "| jsonl={} | csv={}",
            enriched.get("event_type", ""),
            enriched.get("split", ""),
            enriched.get("sample_id", ""),
            enriched.get("iteration", ""),
            enriched.get("working_triples", ""),
            enriched.get("base_triples", ""),
            enriched.get("closure_triples", ""),
            enriched.get("newly_inferred", ""),
            enriched.get("java_seconds", ""),
            enriched.get("serialize_seconds", ""),
            enriched.get("parse_seconds", ""),
            enriched.get("total_seconds", ""),
            self._timing_jsonl_path.resolve(),
            self._timing_csv_path.resolve(),
        )

    def _materialize(
        self,
        base_triples: Set[Triple],
        split_name: str = "",
        sample_id: str = "",
    ) -> Tuple[Set[Triple], Dict[Triple, int]]:
        """
        Materialize inferred facts from base triples using forward-chaining reasoning.

        Returns:
            Tuple of (inferred_triples, hop_depths)
            where hop_depths[triple] = iteration at which triple was first inferred
        """
        jena_profile = str(self.cfg.materialization.get("jena_profile", "owl_mini"))
        return self._materialize_singlepass_jena(base_triples, split_name, sample_id, jena_profile)

    def _schema_uri_triples(self) -> Set[Triple]:
        return {
            (s, p, o)
            for s, p, o in self.ontology_graph
            if isinstance(s, URIRef) and isinstance(p, URIRef) and isinstance(o, URIRef)
        }

    def _materialize_singlepass_jena(
        self,
        base_triples: Set[Triple],
        split_name: str,
        sample_id: str,
        jena_profile: str,
    ) -> Tuple[Set[Triple], Dict[Triple, int]]:
        """Single-pass Jena materialization."""
        t0 = time.perf_counter()
        closure, native_hop_depths = self.jena_materializer.materialize(
            self.ontology_path, base_triples, jena_profile=jena_profile
        )
        schema_uri_triples = self._schema_uri_triples()
        inferred = closure - base_triples - schema_uri_triples

        # Hydrate unlogged inferred triples (or single-pass overrides) with hops=1,
        # but keep the deep native hop depths extracted from the Java engine.
        hop_depths = {t: native_hop_depths.get(t, 1) for t in inferred}

        self._append_timing_event(
            {
                "event_type": "jena_singlepass",
                "split": split_name,
                "sample_id": sample_id,
                "iteration": 1,
                "jena_profile": jena_profile,
                "base_triples": len(base_triples),
                "closure_triples": len(closure),
                "newly_inferred": len(inferred),
                "function_total_seconds": time.perf_counter() - t0,
                **self.jena_materializer.last_run_timing,
            }
        )
        return inferred, hop_depths

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
        hop_depths: Dict[Triple, int],
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        facts_rows: List[Dict[str, str]] = []
        targets_rows: List[Dict[str, str]] = []

        positive_meta: Dict[Triple, Tuple[str, int]] = {}

        for t in sorted(base_triples, key=lambda x: (str(x[0]), str(x[1]), str(x[2]))):
            positive_meta[t] = ("base_fact", 0)

        for t in sorted(inferred_triples, key=lambda x: (str(x[0]), str(x[1]), str(x[2]))):
            if t not in positive_meta:
                # Use actual hop depth from materialization, or default to 1 if not in mapping
                hop_depth = hop_depths.get(t, 1)
                positive_meta[t] = ("inf_root", hop_depth)

        for (s, p, o), (fact_type, hops) in positive_meta.items():
            row = {
                "sample_id": sample_id,
                "subject": self._clean_name(s),
                "predicate": "rdf:type" if p == RDF.type else self._clean_name(p),
                "object": self._clean_name(o),
                "label": 1,
                "truth_value": "True",
                "type": fact_type,
                "hops": hops,
                "corruption_method": "",
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

        # Exclude symmetric/identity predicates (owl:differentFrom, owl:sameAs)
        # from the corruption pool.  In OWL2Bench all individuals are pairwise
        # differentFrom, so any random corruption of such a triple hits an
        # existing positive and is rejected, wasting nearly all attempts.
        _SKIP_CORRUPTION = {OWL.differentFrom, OWL.sameAs}
        corruptible = [(t, m) for t, m in positives if t[1] not in _SKIP_CORRUPTION]
        if not corruptible:
            corruptible = positives

        attempts = 0
        while len(negatives) < n_negatives and attempts < max_attempts and corruptible:
            attempts += 1
            (pos_triple, (pos_type, pos_hops)) = random.choice(corruptible)
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
                    "truth_value": "False",
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

            inferred, hop_depths = self._materialize(base_triples, split_name=split_name, sample_id=sample_id)

            individual_set = set(individuals)
            filtered_base = self._filter_instance_triples(base_triples, individual_set)
            filtered_inferred = self._filter_instance_triples(inferred, individual_set) - filtered_base

            sample_facts, sample_targets = self._make_rows_for_sample(
                sample_id=sample_id,
                individuals=individuals,
                base_triples=filtered_base,
                inferred_triples=filtered_inferred,
                hop_depths=hop_depths,
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
        timing = {}
        phase_start = time.perf_counter()
        os.makedirs(self.output_dir, exist_ok=True)

        # --- Ontology Parsing Phase (already done in __init__) ---
        # If you want to time parsing, move ontology_graph.parse timing here.
        # For now, just record a dummy value if needed.
        # timing["parsing_seconds"] = ...

        # --- Base Fact Generation Phase ---
        base_fact_start = time.perf_counter()
        self._generate_split("train", self.cfg.dataset.n_train)
        self._generate_split("val", self.cfg.dataset.n_val)
        self._generate_split("test", self.cfg.dataset.n_test)
        base_fact_end = time.perf_counter()
        timing["base_fact_generation_seconds"] = base_fact_end - base_fact_start

        # --- Materialization Phase ---
        # Already included in _materialize/_materialize_singlepass_jena per sample, but can sum if needed.
        # For now, not separated out globally.

        # --- Reporting Phase ---
        reporting_start = time.perf_counter()
        # Could add more reporting logic here if needed
        reporting_end = time.perf_counter()
        timing["reporting_seconds"] = reporting_end - reporting_start

        total_runtime = time.perf_counter() - phase_start
        timing["total_seconds"] = total_runtime

        # Save timing metrics
        metrics_path = os.path.join(self.output_dir, "generation_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"timing": timing}, f, indent=2)
        logger.info(f"Saved generation metrics to: {metrics_path}")

        if self._timing_enabled:
            logger.info(
                "Timing recorder summary | run_id={} | events={} | jsonl={} | csv={} "
                "| includes_metrics=[per_call:serialize_seconds,java_seconds,parse_seconds,total_seconds;"
                "per_iteration:working_triples,closure_triples,newly_inferred,iteration_wall_seconds]",
                self._timing_run_id,
                self._timing_event_count,
                self._timing_jsonl_path.resolve(),
                self._timing_csv_path.resolve(),
            )


REPO_ROOT = os.environ.get("SYNTHOLOGY_ROOT", "../../../..")


@hydra.main(version_base=None, config_path=f"{REPO_ROOT}/configs/udm_baseline", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Running FC baseline generator with configuration:\n" + OmegaConf.to_yaml(cfg))

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=cfg.logging.level, colorize=True)

    generator = FCBaselineGenerator(cfg)
    generator.generate()


if __name__ == "__main__":
    main()
