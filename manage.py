import argparse
import pathlib
import re
import subprocess

channels = ["conda-forge"]
dev_packages_conda = [
    "anaconda-client",
    "codecov",
    "conda-build",
    "conda-verify",
    "git",
    "latexmk",
    "make",
    "poetry",
    "sphinx",
    "sphinx-autodoc-typehints",
]
dev_packages_py = ["black", "flake8", "poetry2conda", "ruamel.yaml", "toml"]
test_packages_py = ["pytest", "pytest-cov"]

# HELPER FUNCTIONS


def _interleave(s, args):
    return [val for a in args for val in (s, a)]


def _hash_tarball(tarball_path):
    import hashlib

    m = hashlib.sha256()
    with open(tarball_path, "rb") as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            m.update(data)

    return m.hexdigest()


def _remove_paths(self, *paths):
    import shutil

    for p in paths:
        try:
            p.unlink()
        except OSError:
            try:
                shutil.rmtree(p.resolve())
            except OSError:
                continue


def _changelog_helper(tag, repo, previous_tag=None):
    if previous_tag is None:
        return []

    version_header = (
        "`Version "
        + tag[1:]
        + " <"
        + repo
        + "/compare/"
        + previous_tag
        + "..."
        + tag
        + ">`__"
    )
    gitlog = subprocess.check_output(
        [
            "git",
            "log",
            previous_tag + "..." + tag,
            "--pretty=* %s (`%h <" + repo + "/commit/%H>`__)",
            "--no-merges",
        ],
        text=True,
    )

    lines = []
    lines.append(version_header)
    lines.append("-" * len(version_header) + "\n")
    lines.extend(
        [line for line in gitlog.splitlines() if re.search(r"feat:|fix:", line)]
    )
    lines.append("\n")

    return lines


# COMMAND LINE FUNCTIONS


def setup(env_name):
    subprocess.run(
        ["conda", "create", "-n", env_name, "python=3"]
        + dev_packages_conda
        + _interleave("-c", channels)
        + ["-y"]
    )


def init(repo, homepage=None, conda_sub={}, readme="README.rst"):
    # INTERACTIVE POETRY INIT
    subprocess.run(["poetry", "init"])

    subprocess.run(["poetry", "add", "--dev"] + dev_packages_py + test_packages_py)

    # MANUALLY MODIFY PYPROJECT.TOML

    p = pathlib.Path("pyproject.toml")

    import toml

    with open(p, "r") as f:
        d = toml.load(f)

    # POETRY OPTIONAL ENTRIES

    d["tool"]["poetry"]["readme"] = readme
    d["tool"]["poetry"]["repository"] = repo
    d["tool"]["poetry"]["homepage"] = (
        d["tool"]["poetry"]["repository"] if homepage is None else homepage
    )

    # PYTEST

    d["tool"]["pytest"] = dict(
        ini_options=dict(
            minversion=6.0, addopts="-x -v --cov --cov-report=xml", testpaths="tests"
        )
    )

    # POETRY2CONDA

    d["tool"]["poetry2conda"] = dict(name=d["tool"]["poetry"]["name"])
    d["tool"]["poetry2conda"]["dependencies"] = conda_sub

    with open(p, "w") as f:
        toml.dump(d, f)


def lint():
    subprocess.run(["black", "src", "tests", "examples", "manage.py"])
    subprocess.check_call(
        [
            "flake8",
            "--max-line-length",
            "88",
            "--extend-ignore",
            "E203,W503",
            "--per-file-ignores",
            "__init__.py:F401,F403",
        ]
    )


def install():
    subprocess.run(["poetry", "install"])


def test(codecov_token=None):
    subprocess.check_call(["pytest"])
    if codecov_token is not None:
        subprocess.run(["codecov", "-t", codecov_token])


def build():
    subprocess.run(["poetry", "build"])


def build_conda(*build_channels):
    import ruamel.yaml
    import toml

    # READ PYPROJECT.TOML

    with open(pathlib.Path("pyproject.toml"), "r") as f:
        d = toml.load(f)

    d = d["tool"]["poetry"]

    # TARBALL PATH

    p = pathlib.Path("dist", d["name"] + "-" + d["version"] + ".tar.gz")

    # RUNTIME DEPENDENCIES

    s = subprocess.check_output(["poetry2conda", "pyproject.toml", "-"])
    run_deps = ruamel.yaml.YAML(typ="safe").load(s)["dependencies"]

    # AUTHOR INFO

    author, *_ = d["authors"]
    *author_name, author_email = author.split()
    author_name = " ".join(author_name)
    author_email = author_email[1:-1]

    # CONSTRUCT META.YAML

    x = {}
    x["package"] = dict(name=d["name"], version=d["version"])
    x["source"] = dict(url="file://" + str(p.resolve()), sha256=str(_hash_tarball(p)))
    x["build"] = dict(
        number=0,
        noarch="python",
        script="{{ PYTHON }} -m pip install . --no-deps -vv",
    )
    x["requirements"] = dict(run=run_deps)
    x["test"] = dict(requires=test_packages_py)
    x["about"] = dict(
        home=d["homepage"],
        author=author_name,
        author_email=author_email,
        license=d["license"],
        license_file=str(pathlib.Path("LICENSE").resolve()),
    )

    # WRITE META.YAML
    with open("meta.yaml", "w") as f:
        ruamel.yaml.YAML().dump(x, f)

    # RUN CONDA BUILD
    subprocess.run(
        ["conda", "build", "debug", str(pathlib.Path(".").resolve())]
        + _interleave("-c", build_channels)
    )


def publish(pypi_token, conda_token):
    import toml

    # READ PYPROJECT.TOML

    with open(pathlib.Path("pyproject.toml"), "r") as f:
        d = toml.load(f)["tool"]["poetry"]

    if pypi_token is not None:
        subprocess.run(["poetry", "config", "pypi-token.pypi", pypi_token])
        subprocess.run(["poetry", "publish"])
    if conda_token is not None:
        subprocess.run(["anaconda", "--token", conda_token, "upload", d["name"]])


def push(remote="origin"):
    subprocess.run(["git", "push", remote])


def release(release_type, remote="origin"):
    import toml

    subprocess.check_call(["poetry", "version", release_type])

    # GET VERSION NUMBER

    with open(pathlib.Path("pyproject.toml"), "r") as f:
        d = toml.load(f)
        repo = str(d["tool"]["poetry"]["repository"])
        version = str(d["tool"]["poetry"]["version"])

    subprocess.run(["git", "add", "pyproject.toml"])
    subprocess.run(["git", "commit", "-m", '"chore: Update version number"'])

    subprocess.run(["git", "tag", "-a", "v" + version, "-m", "'v" + version + "'"])

    # UPDATE CHANGELOG

    first_commit, *_ = subprocess.check_output(
        ["git", "log", "--reverse", "--pretty=%h"], text=True
    ).splitlines()
    tags = (
        [first_commit]
        + subprocess.check_output(["git", "tag", "-l"], text=True).splitlines()
        + ["v" + version]
    )

    changelog = []
    for i in range(1, len(tags)):
        changelog.extend(_changelog_helper(tags[i], repo, tags[i - 1]))

    changelog = "\n".join(changelog)

    with open(pathlib.Path("CHANGELOG.rst"), "w") as f:
        f.write(changelog)

    subprocess.run(["git", "tag", "-d", "v" + version])

    subprocess.run(["git", "add", "CHANGELOG.rst"])
    subprocess.run(["git", "commit", "-m", '"chore: Update changelog"'])

    # PUSH MASTER

    subprocess.run(["git", "push", remote, "master"])

    # CREATE AND PUSH TAG

    subprocess.run(["git", "tag", "-a", "v" + version, "-m", "'v" + version + "'"])
    subprocess.run(["git", "push", remote, "v" + version])


def clean(env_name=None):
    _remove_paths(
        pathlib.Path("build"),
        pathlib.Path("conda-build"),
        pathlib.Path("docs"),
        pathlib.Path(".pytest_cache"),
        pathlib.Path(".coverage"),
        pathlib.Path("coverage.xml"),
        pathlib.Path("dist"),
        pathlib.Path("meta.yaml"),
        pathlib.Path("poetry.lock"),
        *pathlib.Path("src").glob("*.egg-info"),
        *pathlib.Path(".").glob(".coverage*"),
        *pathlib.Path(".").rglob("__pycache__"),
    )
    if env_name is not None:
        subprocess.run(["conda", "env", "remove", "-n", env_name])


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", type=str)
    # parser.add_argument("--pypi-token",type=str)
    # parser.add_argument("--conda-token",type=str)
    parser.add_argument("pos_args", nargs="*")
    parser.add_argument("--conda_sub", type=json.loads)
    args = parser.parse_args()

    funcs = dict(
        setup=setup,
        init=init,
        lint=lint,
        install=install,
        test=test,
        build=build,
        build_conda=build_conda,
        publish=publish,
        clean=clean,
        release=release,
    )

    if args.cmd == "init":
        funcs[args.cmd](*args.pos_args, conda_sub=args.conda_sub)
    else:
        funcs[args.cmd](*args.pos_args)
