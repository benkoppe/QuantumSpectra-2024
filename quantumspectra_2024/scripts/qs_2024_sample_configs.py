from argparse import ArgumentParser

from quantumspectra_2024.common.Config import copy_sample_configs


def main():
    parser = ArgumentParser(
        description="Copy sample configuration files to a given directory."
    )

    parser.add_argument(
        "destination_path", type=str, help="Path to the destination dierctory."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files."
    )
    args = parser.parse_args()

    destination_path = args.destination_path
    overwrite = args.overwrite

    copy_sample_configs(destination_path, overwrite=overwrite)

    print("Sample configuration files copied successfully.")


if __name__ == "__main__":
    main()
