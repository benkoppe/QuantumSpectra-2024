import subprocess

SCRIPT_NAME = "outputs/hpc/run_everything.py"


def run_script(script_name):
    # run the external script and capture output
    process = subprocess.Popen(
        ["python3", script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # wait for the process to complete and get the output
    stdout, stderr = process.communicate()

    # check if there was an error
    if process.returncode != 0:
        error = stderr.decode()
        print("Error running script.")
        # return None

    # decode the stdout and split into lines
    output_lines = stdout.decode().splitlines()

    # get the last line of output
    last_line = output_lines[-1] if output_lines else None

    return last_line


def main():
    while True:
        last_line = run_script(SCRIPT_NAME)

        if last_line is not None:
            print(last_line)

            if "/" in last_line:
                last_line_numbers = last_line.split("/")
                if last_line_numbers[0] == last_line_numbers[1]:
                    break

    print(f"Last line of output: {last_line}")


if __name__ == "__main__":
    main()
