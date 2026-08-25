"""Tests for gen_bins_from_tsv.py - bin generation from TSV clustering results."""

import os
import pytest
import gzip
from pathlib import Path
from binferno.postprocessing.scripts.gen_bins_from_tsv import gen_bins


@pytest.fixture
def fasta_file(tmp_path):
    """Create test FASTA file."""
    fasta_path = tmp_path / "test_contigs.fasta"
    fasta_path.write_text(
        ">contig_1\nACGTACGTACGTACGT\n"
        ">contig_2\nTGCATGCATGCATGCA\n"
        ">contig_3\nAAAAAAAAAAGGGGGGGG\n"
        ">contig_4\nCCCCCCCCTTTTTTTT\n"
    )
    return str(fasta_path)


@pytest.fixture
def gzipped_fasta_file(tmp_path):
    """Create gzipped FASTA file."""
    fasta_path = tmp_path / "test_contigs.fasta.gz"
    content = ">contig_1\nACGTACGTACGTACGT\n>contig_2\nTGCATGCATGCATGCA\n"
    with gzip.open(fasta_path, 'wt') as f:
        f.write(content)
    return str(fasta_path)


@pytest.fixture
def tsv_file(tmp_path):
    """Create test TSV."""
    tsv_path = tmp_path / "clustering.tsv"
    tsv_path.write_text(
        "contig_1\tbin_0\n"
        "contig_2\tbin_0\n"
        "contig_3\tbin_1\n"
        "contig_4\tbin_1\n"
    )
    return str(tsv_path)


def test_gen_bins_basic(fasta_file, tsv_file, tmp_path):
    """Test basic bin generation."""
    output_dir = tmp_path / "bins"
    gen_bins(fasta_file, tsv_file, str(output_dir))
    
    assert output_dir.exists()
    bin_files = list(output_dir.glob("*.fa"))
    assert len(bin_files) == 2


def test_gen_bins_creates_output_directory(fasta_file, tsv_file, tmp_path):
    """Test output directory creation."""
    output_dir = tmp_path / "new_bins_dir"
    assert not output_dir.exists()
    
    gen_bins(fasta_file, tsv_file, str(output_dir))
    assert output_dir.exists()


def test_gen_bins_with_gzipped_fasta(gzipped_fasta_file, tmp_path):
    """Test with gzipped FASTA."""
    tsv_path = tmp_path / "clustering.tsv"
    tsv_path.write_text("contig_1\tbin_0\ncontig_2\tbin_1\n")
    
    output_dir = tmp_path / "bins"
    gen_bins(gzipped_fasta_file, str(tsv_path), str(output_dir))
    
    assert output_dir.exists()
    assert len(list(output_dir.glob("*.fa"))) > 0


def test_gen_bins_single_contig_per_bin(fasta_file, tmp_path):
    """Test one contig per bin."""
    tsv_path = tmp_path / "clustering.tsv"
    tsv_path.write_text(
        "contig_1\tbin_0\ncontig_2\tbin_1\n"
        "contig_3\tbin_2\ncontig_4\tbin_3\n"
    )
    
    output_dir = tmp_path / "bins"
    gen_bins(fasta_file, str(tsv_path), str(output_dir))
    
    bin_files = sorted(output_dir.glob("*.fa"))
    assert len(bin_files) == 4
    
    for bin_file in bin_files:
        assert bin_file.read_text().count(">") == 1


def test_gen_bins_multiple_contigs_per_bin(fasta_file, tmp_path):
    """Test multiple contigs in one bin."""
    tsv_path = tmp_path / "clustering.tsv"
    tsv_path.write_text(
        "contig_1\tbin_0\ncontig_2\tbin_0\n"
        "contig_3\tbin_0\ncontig_4\tbin_0\n"
    )
    
    output_dir = tmp_path / "bins"
    gen_bins(fasta_file, str(tsv_path), str(output_dir))
    
    bin_files = list(output_dir.glob("*.fa"))
    assert len(bin_files) == 1
    assert bin_files[0].read_text().count(">") == 4


def test_gen_bins_missing_contig(tmp_path):
    """Test contig in TSV but not in FASTA."""
    fasta_path = tmp_path / "test.fasta"
    fasta_path.write_text(">contig_1\nACGTACGT\n>contig_2\nTGCATGCA\n")
    
    tsv_path = tmp_path / "clustering.tsv"
    tsv_path.write_text("contig_1\tbin_0\ncontig_missing\tbin_0\n")
    
    output_dir = tmp_path / "bins"
    gen_bins(str(fasta_path), str(tsv_path), str(output_dir))
    
    assert output_dir.exists()


def test_gen_bins_multiline_sequences(tmp_path):
    """Test FASTA with multiline sequences."""
    fasta_path = tmp_path / "test.fasta"
    fasta_path.write_text(
        ">contig_1\nACGTACGT\nACGTACGT\n"
        ">contig_2\nTGCATGCA\nTGCATGCA\n"
    )
    
    tsv_path = tmp_path / "clustering.tsv"
    tsv_path.write_text("contig_1\tbin_0\ncontig_2\tbin_0\n")
    
    output_dir = tmp_path / "bins"
    gen_bins(str(fasta_path), str(tsv_path), str(output_dir))
    
    content = list(output_dir.glob("*.fa"))[0].read_text()
    assert "ACGTACGTACGTACGT" in content
    assert "TGCATGCATGCATGCA" in content


def test_gen_bins_with_descriptions(tmp_path):
    """Test FASTA with description lines."""
    fasta_path = tmp_path / "test.fasta"
    fasta_path.write_text(
        ">contig_1 length=1000 coverage=50\nACGTACGT\n"
        ">contig_2 description\nTGCATGCA\n"
    )
    
    tsv_path = tmp_path / "clustering.tsv"
    tsv_path.write_text("contig_1\tbin_0\ncontig_2\tbin_0\n")
    
    output_dir = tmp_path / "bins"
    gen_bins(str(fasta_path), str(tsv_path), str(output_dir))
    
    assert len(list(output_dir.glob("*.fa"))) == 1


def test_gen_bins_sequence_integrity(tmp_path):
    """Test sequences are preserved correctly."""
    fasta_path = tmp_path / "test.fasta"
    seq1 = "ACGTACGTACGTACGTACGTACGT"
    seq2 = "TGCATGCATGCATGCATGCATGCA"
    
    fasta_path.write_text(f">contig_1\n{seq1}\n>contig_2\n{seq2}\n")
    
    tsv_path = tmp_path / "clustering.tsv"
    tsv_path.write_text("contig_1\tbin_0\ncontig_2\tbin_0\n")
    
    output_dir = tmp_path / "bins"
    gen_bins(str(fasta_path), str(tsv_path), str(output_dir))
    
    content = list(output_dir.glob("*.fa"))[0].read_text()
    assert seq1 in content and seq2 in content


def test_gen_bins_large_sequences(tmp_path):
    """Test with large sequences."""
    fasta_path = tmp_path / "large.fasta"
    large_seq = "A" * 100000
    fasta_path.write_text(f">contig_1\n{large_seq}\n>contig_2\n{large_seq}\n")
    
    tsv_path = tmp_path / "clustering.tsv"
    tsv_path.write_text("contig_1\tbin_0\ncontig_2\tbin_0\n")
    
    output_dir = tmp_path / "bins"
    gen_bins(str(fasta_path), str(tsv_path), str(output_dir))
    
    content = list(output_dir.glob("*.fa"))[0].read_text()
    assert len(content) > 200000