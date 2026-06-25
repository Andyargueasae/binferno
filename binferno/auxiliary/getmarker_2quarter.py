#!/usr/bin/env python3
# filepath: /data/binferno/binferno/auxiliary/test_getmarker_2quarter.py

import re
import sys
from collections import defaultdict

COV_CUTOFF = 0.4
CONTIG_RE = re.compile(r"^(.+)_\d+_\d+_[+-]$")
PYRODIGAL_RE = re.compile(r"^(.+)_\d+$")


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


def extract_contig_base(target_name: str):
    m = CONTIG_RE.match(target_name)
    if m:
        return m.group(1)

    m = PYRODIGAL_RE.match(target_name)
    if m:
        return m.group(1)

    return None


def resolve_target_name(target_name: str, known_names):
    if target_name in known_names:
        return target_name

    base_name = extract_contig_base(target_name)
    if base_name is not None and base_name in known_names:
        return base_name

    return None


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
            contig = resolve_target_name(arr[0], lenhash)
            if contig is None:
                continue
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


def countmarker(
    hmm: str, maxbinout_f: str, maxbinoutname: str, markerhmm_f: str, out: str
):
    ret = []
    totalmarker = []
    uniquemarker = []
    binhash = {}
    numhash = {}
    binarr = []
    markerhash = {}
    markerarr = []
    resultarr = defaultdict(dict)

    with open(f"{maxbinout_f}.summary", "r", encoding="utf-8") as fh:
        binnum = 0
        for line in fh:
            m = re.match(r"^Bin \[([^\t]+)\]\t", line)
            if not m:
                continue
            curr = m.group(1)
            currname = curr
            m2 = re.search(r"([0-9]+)\.fasta$", curr)
            if m2:
                idx = m2.group(1)
                with open(f"{maxbinout_f}.{idx}.fasta", "r", encoding="utf-8") as ff:
                    for fline in ff:
                        if fline.startswith(">"):
                            hdr = trim(fline[1:])
                            binhash[hdr] = curr
                currname = f"{maxbinoutname}.{idx}.fasta"
            numhash[curr] = binnum
            binnum += 1
            binarr.append(currname)

    i = 0
    with open(markerhmm_f, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("NAME"):
                arr = re.split(r"[ ]+", line.strip())
                if len(arr) > 1:
                    curr = check_marker(arr[1])
                    if curr not in markerhash:
                        markerhash[curr] = i
                        markerarr.append(curr)
                        i += 1
    genenum = i

    curr = ""
    with open(hmm, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            arr = re.split(r"[ ]+", line.strip())
            if len(arr) <= 16:
                continue
            inlinehmm = check_marker(arr[3])
            contig = resolve_target_name(arr[0], binhash)
            if contig is None:
                continue
            if curr != inlinehmm:
                curr = inlinehmm
                if curr not in markerhash:
                    continue
                i = markerhash[curr]
            j = int(float(arr[16])) - int(float(arr[15]))
            hlen = float(arr[5])
            if contig in binhash and hlen > 0 and (j / hlen) >= COV_CUTOFF:
                b = numhash[binhash[contig]]
                resultarr[i][b] = resultarr[i].get(b, 0) + 1

    for j in range(binnum):
        total = 0
        uniq = 0
        for i in range(genenum + 1):
            if j in resultarr.get(i, {}):
                total += resultarr[i][j]
                uniq += 1
        totalmarker.append(total)
        uniquemarker.append(uniq)
        ret.append((uniq / genenum) if genenum > 0 else 0.0)

    with open(out, "w", encoding="utf-8") as fout:
        fout.write("\tTotal marker\tUnique marker")
        for j in range(genenum):
            fout.write(f"\t{markerarr[j]}")
        fout.write("\n")
        for j in range(binnum):
            fout.write(f"{binarr[j]}\t{totalmarker[j]}\t{uniquemarker[j]}")
            for i in range(genenum + 1):
                fout.write(
                    f"\t{resultarr[i][j]}" if j in resultarr.get(i, {}) else "\t"
                )
            fout.write("\n")

    return ret


def listmarker_bybin(
    hmm: str, maxbinout_f: str, contigfaa_f: str, markerhmm_f: str, out_f: str
):
    binhash = {}
    markerhash = {}
    resulthash = {}
    hmmclasshash = {}
    faahash = {}

    with open(f"{maxbinout_f}.summary", "r", encoding="utf-8") as fh:
        for line in fh:
            m = re.match(r"^([^\t]+)\t", line)
            if not m:
                continue
            curr = m.group(1)
            m2 = re.search(r"([0-9]+)\.fasta$", curr)
            if m2:
                currbin = m2.group(1)
                with open(
                    f"{maxbinout_f}.{currbin}.fasta", "r", encoding="utf-8"
                ) as ff:
                    for fline in ff:
                        if fline.startswith(">"):
                            hdr = trim(fline[1:])
                            binhash[hdr] = currbin

    with open(hmm, "r", encoding="utf-8") as fh:
        curr = ""
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            arr = re.split(r"[ ]+", line.strip())
            if len(arr) <= 16:
                continue
            inlinehmm = check_marker(arr[3])
            contig = resolve_target_name(arr[0], binhash)
            if contig is None:
                continue
            if curr != inlinehmm:
                curr = inlinehmm
            j = int(float(arr[16])) - int(float(arr[15]))
            hlen = float(arr[5])
            if contig in binhash and hlen > 0 and (j / hlen) >= COV_CUTOFF:
                resulthash[arr[0]] = contig
                markerhash[arr[0]] = inlinehmm
                hmmclasshash.setdefault(binhash[contig], 0)

    with open(contigfaa_f, "r", encoding="utf-8") as fh:
        key = None
        for line in fh:
            if line.startswith(">"):
                key = trim(line[1:])
                faahash[key] = ""
            else:
                line = line.strip()
                if line and key is not None:
                    faahash[key] += line

    for cls in hmmclasshash.keys():
        with open(f"{out_f}.{cls}.marker.fasta", "w", encoding="utf-8") as fout:
            for gene in sorted(markerhash.keys(), key=lambda x: markerhash[x]):
                contig = resulthash[gene]
                if binhash[contig] == cls:
                    fout.write(
                        f">{maxbinout_f}.{binhash[contig]}.{markerhash[gene]}\n{faahash.get(gene, '')}\n"
                    )


def listmarker_bymarker(
    hmm: str, maxbinout_f: str, contigfaa_f: str, markerhmm_f: str, out_f: str
):
    binhash = {}
    markerhash = {}
    resulthash = {}
    hmmclasshash = {}
    faahash = {}

    with open(f"{maxbinout_f}.summary", "r", encoding="utf-8") as fh:
        for line in fh:
            m = re.match(r"^([^\t]+)\t", line)
            if not m:
                continue
            curr = m.group(1)
            m2 = re.search(r"([0-9]+)\.fasta$", curr)
            if m2:
                idx = m2.group(1)
                with open(f"{maxbinout_f}.{idx}.fasta", "r", encoding="utf-8") as ff:
                    for fline in ff:
                        if fline.startswith(">"):
                            hdr = trim(fline[1:])
                            binhash[hdr] = curr

    with open(hmm, "r", encoding="utf-8") as fh:
        curr = ""
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            arr = re.split(r"[ ]+", line.strip())
            if len(arr) <= 16:
                continue
            inlinehmm = check_marker(arr[3])
            contig = resolve_target_name(arr[0], binhash)
            if contig is None:
                continue
            if curr != inlinehmm:
                curr = inlinehmm
            j = int(float(arr[16])) - int(float(arr[15]))
            hlen = float(arr[5])
            if contig in binhash and hlen > 0 and (j / hlen) >= COV_CUTOFF:
                resulthash[arr[0]] = contig
                markerhash[arr[0]] = inlinehmm
                hmmclasshash.setdefault(inlinehmm, 0)

    with open(contigfaa_f, "r", encoding="utf-8") as fh:
        key = None
        for line in fh:
            if line.startswith(">"):
                key = trim(line[1:])
                faahash[key] = ""
            else:
                line = line.strip()
                if line and key is not None:
                    faahash[key] += line

    for cls in hmmclasshash.keys():
        with open(f"{out_f}.{cls}", "w", encoding="utf-8") as fout:
            for gene in sorted(resulthash.keys()):
                if markerhash[gene] == cls:
                    contig = resulthash[gene]
                    fout.write(
                        f">{binhash[contig]}.{markerhash[gene]}\n{faahash.get(gene, '')}\n"
                    )


if __name__ == "__main__":
    # Same default behavior as Perl:
    # gethmmmarker($ARGV[0], $ARGV[1], $ARGV[2], $ARGV[3], optional $ARGV[4])
    if len(sys.argv) < 5:
        print(
            "Usage: test_getmarker_2quarter.py <hmmout> <fasta> <min_seq_len> <out> [max_effort]",
            file=sys.stderr,
        )
        raise RuntimeError("Not enough arguments provided")

    hmm = sys.argv[1]
    fasta = sys.argv[2]
    min_seq_len = int(sys.argv[3])
    out = sys.argv[4]
    max_effort = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    rc = gethmmmarker(hmm, fasta, min_seq_len, out, max_effort)
    if rc != 0:
        print("No suitable marker found.", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Marker(s) written to {out}.")
        sys.exit(0)
