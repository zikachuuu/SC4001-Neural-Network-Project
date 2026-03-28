import os
import subprocess
import sys


def main() -> None:
    """Compatibility wrapper to keep old command usable.

    Delegates to the unified preprocessing script with speaker normalization enabled.
    """
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    unified_script = os.path.join(curr_dir, "1_extract_features_emodb.py")
    default_output = os.path.join(curr_dir, "processed_emodb_speaker_norm")

    cmd = [
        sys.executable,
        unified_script,
        "--normalize-speaker",
        "--output-dir",
        default_output,
    ]

    # Preserve custom CLI overrides (for example --interactive or --data-dir).
    cmd.extend(sys.argv[1:])
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()