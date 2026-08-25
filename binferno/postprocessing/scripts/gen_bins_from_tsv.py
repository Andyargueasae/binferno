#!/usr/bin/env python

from __future__ import print_function
import argparse
import gzip
import os
from collections import defaultdict


def read_fasta(fastafile):
    """Read FASTA file (supports gzipped)."""
    sequences = {}
    if fastafile.endswith("gz"):
        open_func = gzip.open
        mode = "rt"
    else:
        open_func = open
        mode = "r"
    
    with open_func(fastafile, mode) as f:
        current_seq = None
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if " " in line:
                    seq_id, _ = line[1:].split(" ", 1)
                else:
                    seq_id = line[1:]
                sequences[seq_id] = ""
                current_seq = seq_id
            else:
                if current_seq:
                    sequences[current_seq] += line
    
    return sequences


def gen_bins(fastafile, resultfile, outputdir):
    """Generate bins from clustering results."""
    print("Processing file:\t{}".format(fastafile))
    sequences = read_fasta(fastafile)
    
    print("Reading Map:\t{}".format(resultfile))
    bin_to_contigs = defaultdict(list)
    
    with open(resultfile, "r") as f:
        for line in f:
            contig_name, cluster_name = line.strip().split("\t")
            bin_to_contigs[cluster_name].append(contig_name)
    
    print("Writing bins:\t{}".format(outputdir))
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    
    bin_num = 0
    for _cluster_name, contigs in bin_to_contigs.items():
        binfile = os.path.join(outputdir, "{}.fa".format(bin_num))
        with open(binfile, "w") as f:
            for contig_name in contigs:
                if contig_name in sequences:
                    f.write(">{}\n".format(contig_name))
                    f.write("{}\n".format(sequences[contig_name]))
        bin_num += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fasta", help="FASTA file")
    parser.add_argument("result", help="Clustering result TSV")
    parser.add_argument("output", help="Output directory")
    args = parser.parse_args()
    
    gen_bins(args.fasta, args.result, args.output)