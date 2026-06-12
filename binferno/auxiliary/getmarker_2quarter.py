#!/usr/bin/env python3
# filepath: /data/binferno/binferno/auxiliary/test_getmarker_2quarter.py

import re
import sys
from collections import defaultdict

COV_CUTOFF = 0.4
CONTIG_RE = re.compile(r"^(.+)_\d+_\d+_[+-]$")


def trim(s: str) -> str:
    t = s.rstrip()
    i = t.find(" ")
    if i >= 0:
        t = t[:i]
    return t


def check_marker(marker: str) -> str:
    if marker == "TIGR00388":
        return "TIGR00389"
    if marker == "TIGR00471":
        return "TIGR00472"
    if marker == "TIGR00408":
        return "TIGR00409"
    if marker == "TIGR02386":
        return "TIGR02387"
    return marker


def _read_fasta_lengths(fasta_f: str):
    lengths = {}
    current = ""
    with open(fasta_f, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                current = trim(line[1:])
                lengths[current] = 0
            elif line:
                lengths[current] = lengths.get(current, 0) + len(line)
    return lengths


def gethmmmarker(
    hmm: str, fasta_f: str, min_seq_len: int, out: str, max_effort: int = 0
) -> int:
    lenhash = _read_fasta_lengths(fasta_f)

    current = ""
    tmpset = set()
    queryhash = {}
    queryseq = []
    queryseqnum = []
    querylen = []
    querycount = 0

    with open(hmm, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            arr = re.split(r"[ ]+", line.strip())
            if len(arr) <= 16:
                continue

            inlinehmm = check_marker(arr[3])
            m = CONTIG_RE.match(arr[0])
            if not m:
                continue

            contig = m.group(1)
            if contig not in lenhash or lenhash[contig] < int(min_seq_len):
                continue

            if current != inlinehmm:
                if current != "" and len(tmpset) > 0:
                    queryhash[current] = querycount
                    q = sorted(tmpset)
                    queryseq.append(q)
                    queryseqnum.append(len(q))
                    querycount += 1
                current = inlinehmm
                querylen.append(int(float(arr[5])))
                tmpset = set()

            cov_len = int(float(arr[16])) - int(float(arr[15]))
            hmm_len = float(arr[5])
            if hmm_len > 0 and (cov_len / hmm_len) >= COV_CUTOFF:
                tmpset.add(contig)

    if len(tmpset) > 0:
        queryhash[current] = querycount
        q = sorted(tmpset)
        queryseq.append(q)
        queryseqnum.append(len(q))
        querycount += 1

    if len(queryhash) == 0:
        return -1

    if max_effort == 1:
        arr = sorted(queryseqnum, reverse=True)
        i = arr[int(querycount / 2)]
        if i == 1:
            for j in range(int(querycount / 2), -1, -1):
                if arr[j] > i:
                    i = arr[j]
                    break
    else:
        arr = sorted(queryseqnum)
        i = arr[int(querycount / 2)]

    if i <= 1:
        if max_effort == 0:
            return -1
        longest = ""
        second = ""
        for c in sorted(lenhash.keys(), key=lambda x: lenhash[x], reverse=True):
            if longest == "":
                longest = c
            elif second == "":
                second = c
            else:
                break

        with open(out, "w", encoding="utf-8") as fout:
            if i == 0:
                fout.write(f"{longest}\n{second}\n")
            else:
                jbest = 999999
                chosen = ""
                for marker, idx in queryhash.items():
                    if queryseqnum[idx] == i and jbest > querylen[idx]:
                        jbest = querylen[idx]
                        chosen = marker
                first = queryseq[queryhash[chosen]][0]
                fout.write(f"{first}\n")
                fout.write(f"{second if longest == first else longest}\n")
    else:
        jbest = 999999
        chosen = ""
        for marker, idx in queryhash.items():
            if queryseqnum[idx] == i and jbest > querylen[idx]:
                jbest = querylen[idx]
                chosen = marker

        with open(out, "w", encoding="utf-8") as fout:
            idx = queryhash[chosen]
            for c in queryseq[idx]:
                fout.write(f"{c}\n")
    return 0

if __name__ == "__main__":
    # Same default behavior as Perl:
    # gethmmmarker($ARGV[0], $ARGV[1], $ARGV[2], $ARGV[3], optional $ARGV[4])
    if len(sys.argv) < 5:
        print(
            "Usage: test_getmarker_2quarter.py <hmmout> <fasta> <min_seq_len> <out> [max_effort]",
            file=sys.stderr,
        )
        sys.exit(1)

    hmm = sys.argv[1]
    fasta = sys.argv[2]
    min_seq_len = int(sys.argv[3])
    out = sys.argv[4]
    max_effort = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    rc = gethmmmarker(hmm, fasta, min_seq_len, out, max_effort)
    sys.exit(0 if rc == 0 else 1)
