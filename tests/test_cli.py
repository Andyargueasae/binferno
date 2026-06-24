import pytest
import subprocess
import os


def test_cli():
    """
    run binferno --help and check that it returns 0
    """
    result = subprocess.run(["binferno", "--help"], capture_output=True)
    assert (
        result.returncode == 0
    ), f"binferno --help failed with return code {result.returncode}"
    assert (
        b"usage" in result.stdout
    ), "binferno --help did not contain usage information"
    assert (
        "--contigs" in result.stdout.decode()
    ), "binferno --help did not mention --contigs option"
    assert (
        "--bam" in result.stdout.decode()
    ), "binferno --help did not mention --bam option"
    assert (
        "--output-dir" in result.stdout.decode()
    ), "binferno --help did not mention --output-dir option"
    assert (
        "--threads" in result.stdout.decode()
    ), "binferno --help did not mention --threads option"
    assert (
        "--train-only" in result.stdout.decode()
    ), "binferno --help did not mention --train-only option"

    assert (
        "--training.use-simclr" in result.stdout.decode()    ), "binferno --help did not mention --training.use-simclr option"
    assert (
        "--training.use-barlowtwins" in result.stdout.decode()    ), "binferno --help did not mention --training.use-barlowtwins option"
    assert (
        "--postprocessor.checkm-database" in result.stdout.decode()    ), "binferno --help did not mention --postprocessor.checkm-database option"
    
    return


def test_cli_version():
    assert (
        True
    ), "This test is a placeholder for testing the --version flag of the CLI. Implement this test to check that binferno --version returns the expected version string."
    return
