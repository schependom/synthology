from pathlib import Path
import shlex
import subprocess

import rdflib


def main() -> None:
    base_path = Path("data/lubm/raw/lubm_1/Universities.ttl")
    tbox_path = Path("data/ont/input/lubm-rl.ttl")
    abox_out = Path("/tmp/lubm_abox_filtered.ttl")
    out_ttl = Path("/tmp/lubm_jena_materialized_rl.ttl")

    raw = rdflib.Graph()
    raw.parse(base_path, format="turtle")

    abox = rdflib.Graph()
    for s, p, o in raw:
        if isinstance(s, rdflib.BNode) or isinstance(o, rdflib.BNode):
            continue
        if isinstance(o, rdflib.Literal):
            continue
        abox.add((s, p, o))

    abox.serialize(destination=str(abox_out), format="turtle")

    cmd = (
        "java --class-path 'vendor/apache-jena-6.0.0/lib/*' "
        "apps/LUBM/src/lubm/vendor/JenaMaterializer.java {input_tbox} {input_abox} {output_ttl}"
    ).format(
        input_tbox=shlex.quote(str(tbox_path)),
        input_abox=shlex.quote(str(abox_out)),
        output_ttl=shlex.quote(str(out_ttl)),
    )

    print("running:", cmd)
    proc = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print("returncode=", proc.returncode)
    print("stdout:\n", proc.stdout)
    print("stderr:\n", proc.stderr)

    mat = rdflib.Graph()
    mat.parse(out_ttl, format="turtle")

    tbox = rdflib.Graph()
    tbox.parse(tbox_path, format="turtle")

    base_set = set(abox)
    mat_set = set(mat)
    tbox_set = set(tbox)

    novel = [t for t in mat_set if t not in base_set and t not in tbox_set]

    print("counts:")
    print("  abox_filtered=", len(base_set))
    print("  tbox_rl=", len(tbox_set))
    print("  materialized_ttl=", len(mat_set))
    print("  novel_excluding_base_tbox=", len(novel))

    for i, (s, p, o) in enumerate(sorted(novel, key=lambda x: (str(x[0]), str(x[1]), str(x[2])))):
        if i >= 20:
            break
        reasons = []
        if isinstance(s, rdflib.BNode):
            reasons.append("subject_bnode")
        if isinstance(o, rdflib.BNode):
            reasons.append("object_bnode")
        if isinstance(o, rdflib.Literal):
            reasons.append("object_literal")

        label = ";".join(reasons) if reasons else "uri_uri"
        print(f"  {i + 1} | {label} | {s.n3()} {p.n3()} {o.n3()}")


if __name__ == "__main__":
    main()
