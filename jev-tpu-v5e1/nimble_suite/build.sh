#!/usr/bin/env bash
# Rebuild Bespoke Labs' 13-subset public suite (3,880 records) from the public
# sources with Nimble's own converters at a pinned commit, selecting exactly the
# ids in its committed manifests, and verify every subset's dataset_sha256.
#
#   bash nimble_suite/build.sh <workdir>      # writes <workdir>/public/<subset>/all.jsonl
set -euo pipefail
W=${1:?workdir}; HERE=$(cd "$(dirname "$0")" && pwd)
NIMBLE_COMMIT=0e67403
mkdir -p "$W"; cd "$W"
[ -d nimble ] || git clone -q https://github.com/bespokelabsai/nimble.git
git -C nimble checkout -q $NIMBLE_COMMIT
R=$W/raw; O=$W/public; M=$W/nimble/docs/assets/public-benchmarks/subsets
mkdir -p $R/{boolq,squad_v2,paws,civil_comments,aegis2,helpsteer2,summeval,pubmedqa,vitaminc,massive,multinli} $O
H="python3 $HERE/hf2jsonl.py"
[ -s $R/boolq/validation.jsonl ]        || $H google/boolq default validation $R/boolq/validation.jsonl
[ -s $R/squad_v2/validation.jsonl ]     || $H rajpurkar/squad_v2 squad_v2 validation $R/squad_v2/validation.jsonl
[ -s $R/paws/test.jsonl ]               || $H google-research-datasets/paws labeled_final test $R/paws/test.jsonl
[ -s $R/civil_comments/test.jsonl ]     || $H google/civil_comments default test $R/civil_comments/test.jsonl
[ -s $R/aegis2/test.jsonl ]             || $H nvidia/Aegis-AI-Content-Safety-Dataset-2.0 default test $R/aegis2/test.jsonl
[ -s $R/helpsteer2/validation.jsonl ]   || $H nvidia/HelpSteer2 default validation $R/helpsteer2/validation.jsonl
[ -s $R/summeval/test.jsonl ]           || $H mteb/summeval default test $R/summeval/test.jsonl
[ -s $R/pubmedqa/train.jsonl ]          || $H qiaojin/PubMedQA pqa_labeled train $R/pubmedqa/train.jsonl
[ -s $R/vitaminc/dev.jsonl ] || { curl -sfL -o $R/vitaminc/vitaminc.zip https://github.com/TalSchuster/talschuster.github.io/raw/master/static/vitaminc.zip
  python3 -c "import zipfile,sys;z=zipfile.ZipFile('$R/vitaminc/vitaminc.zip');open('$R/vitaminc/dev.jsonl','wb').write(z.read('vitaminc/dev.jsonl'))"; }
[ -s $R/massive/amazon-massive-dataset-1.1.tar.gz ] || curl -sfL -o $R/massive/amazon-massive-dataset-1.1.tar.gz https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz
[ -s $R/multinli/multinli_1.0/multinli_1.0_dev_matched.jsonl ] || { curl -sfL -o $R/multinli/multinli_1.0.zip https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip
  python3 -c "import zipfile;zipfile.ZipFile('$R/multinli/multinli_1.0.zip').extractall('$R/multinli')"; }
cd $W/nimble
b() { ds=$1; src=$2; out=$3; shift 3; rm -rf $O/$out; python3 -m nimble.datasets.public_benchmarks --dataset $ds --source $src --output-dir $O/$out --ids-from $M/$out-manifest.json "$@" >/dev/null; }
b vitaminc $R/vitaminc/dev.jsonl vitaminc-dev
b massive $R/massive/amazon-massive-dataset-1.1.tar.gz massive-en-US --subset en-US
b massive $R/massive/amazon-massive-dataset-1.1.tar.gz massive-de-DE --subset de-DE
b boolq $R/boolq/validation.jsonl boolq
b squad2 $R/squad_v2/validation.jsonl squad2
b paws $R/paws/test.jsonl paws
b multinli $R/multinli/multinli_1.0/multinli_1.0_dev_matched.jsonl multinli
b civil_comments $R/civil_comments/test.jsonl civil_comments
b aegis2 $R/aegis2/test.jsonl aegis2
b summeval $R/summeval/test.jsonl summeval-relevance --subset relevance
b summeval $R/summeval/test.jsonl summeval-consistency --subset consistency
b pubmedqa $R/pubmedqa/train.jsonl pubmedqa
# HelpSteer2 was selected with a prompt-length filter that dropped one record of a
# family; build the unfiltered 250 and keep exactly the manifest's 249 ids.
rm -rf $O/hs2full; python3 -m nimble.datasets.public_benchmarks --dataset helpsteer2 --source $R/helpsteer2/validation.jsonl --output-dir $O/hs2full --limit 250 >/dev/null
python3 - "$O" "$M" <<'PY'
import json, os, sys, hashlib
O, M = sys.argv[1:3]
m = json.load(open(f"{M}/helpsteer2-manifest.json")); ids = set(m["ids"])
lines = [l for l in open(f"{O}/hs2full/all.jsonl", "rb") if json.loads(l)["id"] in ids]
os.makedirs(f"{O}/helpsteer2", exist_ok=True); open(f"{O}/helpsteer2/all.jsonl", "wb").writelines(lines)
bad = 0
for f in sorted(os.listdir(M)):
    sub = f[: -len("-manifest.json")]
    want = json.load(open(f"{M}/{f}"))["dataset_sha256"]
    got = hashlib.sha256(open(f"{O}/{sub}/all.jsonl", "rb").read()).hexdigest()
    n = sum(1 for _ in open(f"{O}/{sub}/all.jsonl"))
    print(("MATCH" if got == want else "DIFF "), sub, n); bad += got != want
sys.exit(1 if bad else 0)
PY
