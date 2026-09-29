import csv
import statistics


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def column(rows, name):
    return [parse_float(row.get(name)) for row in rows]


def summarize(values):
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def group_by(rows, key):
    groups = {}
    for row in rows:
        groups.setdefault(row.get(key, ""), []).append(row)
    return groups


def report(path, key, metric):
    rows = load_rows(path)
    lines = []
    for name, group in sorted(group_by(rows, key).items()):
        s = summarize(column(group, metric))
        lines.append(f"{name}: n={s['count']} mean={s['mean']:.2f} median={s['median']:.2f}")
    return "\n".join(lines)


def write_summary(path, out_path, key, metric):
    text = report(path, key, metric)
    with open(out_path, "w") as f:
        f.write(text + "\n")
    return len(text.splitlines())
