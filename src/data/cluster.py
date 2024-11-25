import shutil
from pathlib import Path
from subprocess import run

from ..utils.config import get_params


def cluster_sequences(input_fasta: str, output_dir: str, is_test: bool = False):
    """Cluster sequences using MMseqs2"""

    params = get_params("cluster")["test" if is_test else "train"]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(output_dir) / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    cmd = [
        "mmseqs",
        "easy-cluster",
        Path(input_fasta).as_posix(),
        out_dir.as_posix(),
        tmp.as_posix(),
        "-c",
        str(params["coverage"]),
        "--min-seq-id",
        str(params["min_seq_id"]),
        "--cov-mode",
        str(params["cov_mode"]),
        "--cluster-mode",
        str(params["cluster_mode"]),
        "--threads",
        str(params["threads"]),
    ]

    if params["reassign"]:
        cmd.append("--cluster-reassign")

    try:
        run(cmd, check=True)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def main():
    params = get_params("cluster")

    for is_test in [False, True]:
        dataset = Path(params["test" if is_test else "train"])
        out_dir = Path(params["out"]) / ("test" if is_test else "train")

        cluster_sequences(str(dataset), str(out_dir), is_test)


if __name__ == "__main__":
    main()
