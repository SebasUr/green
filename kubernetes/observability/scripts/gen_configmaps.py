#!/usr/bin/env python3
"""Regenerate the two ConfigMaps from their source files.

  dashboards/green-observatory-overview.json  ->  dashboards/dashboard-configmap.yaml
  rte-carbon-exporter/exporter.py             ->  rte-carbon-exporter/configmap.yaml

Both embed the source as a YAML literal block so the Grafana sidecar / the
exporter Pod receive byte-identical files. Run after editing either source.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))


def _block(text: str, indent: str = "    ") -> str:
    return "\n".join(indent + line for line in text.rstrip("\n").split("\n"))


def gen_dashboard_configmap() -> None:
    """Embed every dashboard JSON as its own key; the sidecar writes one file per key."""
    src_dir = os.path.join(ROOT, "dashboards")
    out = os.path.join(src_dir, "dashboard-configmap.yaml")
    names = sorted(n for n in os.listdir(src_dir) if n.endswith(".json"))
    if not names:
        raise SystemExit("no dashboard JSON found")

    data_blocks = []
    for name in names:
        body = open(os.path.join(src_dir, name)).read()
        json.loads(body)  # fail loudly if any dashboard JSON is invalid
        data_blocks.append(f"  {name}: |\n{_block(body)}")

    cm = f"""# Auto-generated from the dashboard JSON files — do not edit by hand.
# Regenerate:  python scripts/gen_configmaps.py
#
# The Grafana sidecar (grafana-sc-dashboard) discovers this ConfigMap by the
# label grafana_dashboard="1" and loads each data key as a dashboard into the
# folder named by the annotation. Apply into the namespace where Grafana runs.
#
# Dashboards: {", ".join(names)}
apiVersion: v1
kind: ConfigMap
metadata:
  name: green-observatory-dashboard
  namespace: monitoring
  labels:
    grafana_dashboard: "1"
    app.kubernetes.io/part-of: green-observatory
  annotations:
    grafana_folder: "Green Observatory"
data:
{chr(10).join(data_blocks)}
"""
    open(out, "w").write(cm)
    print("wrote", os.path.relpath(out, ROOT),
          f"({len(cm.splitlines())} lines, {len(names)} dashboards: {', '.join(names)})")


def gen_exporter_configmap() -> None:
    src = os.path.join(ROOT, "rte-carbon-exporter", "exporter.py")
    out = os.path.join(ROOT, "rte-carbon-exporter", "configmap.yaml")
    body = open(src).read()
    compile(body, "exporter.py", "exec")  # fail loudly if the exporter is broken
    cm = f"""# Auto-generated from exporter.py — do not edit by hand.
# Regenerate:  python scripts/gen_configmaps.py
apiVersion: v1
kind: ConfigMap
metadata:
  name: rte-carbon-exporter-code
  namespace: monitoring
  labels:
    app.kubernetes.io/name: rte-carbon-exporter
    app.kubernetes.io/part-of: green-observatory
data:
  exporter.py: |
{_block(body)}
"""
    open(out, "w").write(cm)
    print("wrote", os.path.relpath(out, ROOT), f"({len(cm.splitlines())} lines)")


if __name__ == "__main__":
    gen_dashboard_configmap()
    gen_exporter_configmap()
