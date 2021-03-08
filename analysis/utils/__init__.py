import subprocess
def run_bash_command(bashCommand):
    print(bashCommand)
    try:
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        return output
    except:
        print("couldn't run bash command, try running it manually")