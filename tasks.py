import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "synthology"
PYTHON_VERSION = "3.12"


# Project commands
@task
def gen_ft_asp(ctx: Context):
    """Generates family tree datasets with ASP solver."""

    # Run __main__.py in uv environment
    ctx.run("uv run --python 3.9 -m ftdatagen")


# Data version control with DVC+SSH
#
#   1. uv add dvc[ssh] --dev
#
#   2. `dvc init`
#
#   3. add SSH information to ~/.ssh/config file
#
#   4. dvc remote modify --local <host-name> auth ssh
#
#         e.g. `dvc remote modify --local gbar1 auth ssh`
#
#   5. Add the folder you want to track with DVC
#      to .gitignore
#
#   6. Remove folder you want to track with DVC from git tracking:
#
#         git rm -r --cached <folder>
#
#   7. Commit changes to git
#
#   8. Now, `uv run invoke dvc` to add, commit and push data changes
#      as being done below:
#
@task
def dvc(ctx, folder="data", message="Add new data"):
    ctx.run(f"dvc add {folder}")
    ctx.run(f"git add {folder}.dvc .gitignore")
    ctx.run(f"git commit -m '{message}'")
    ctx.run("git push")
    ctx.run("dvc push")
