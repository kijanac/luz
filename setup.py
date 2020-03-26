import datetime
from distutils.command.clean import clean
import hashlib
import json
import os
import pathlib
import re
import setuptools
import shutil
import subprocess
from configparser import ConfigParser


class Clean(clean):
    user_options = [("env=", None, "Conda environment to be removed during clean.")]

    def initialize_options(self):
        super().initialize_options()
        self.env = None

    def remove_paths(self, *paths):
        for p in paths:
            try:
                p.unlink()
            except OSError:
                try:
                    shutil.rmtree(p)
                except OSError:
                    continue

    def run(self):
        super().run()
        self.remove_paths(
            pathlib.Path("build"),
            pathlib.Path("conda-build").resolve(),
            pathlib.Path("docs"),
            pathlib.Path(".pytest_cache"),
            pathlib.Path(".coverage"),
            pathlib.Path("coverage.xml"),
            pathlib.Path("dist"),
            *pathlib.Path("src").glob("*.egg-info"),
            *pathlib.Path(".").glob(".coverage*"),
            *pathlib.Path(".").rglob("__pycache__"),
        )
        subprocess.run(["conda", "env", "remove", "-n", self.env])


class Codecov(setuptools.Command):
    user_options = [("codecov-token=", None, "Token for codecov.io coverage upload.")]

    def initialize_options(self):
        self.codecov_token = None

    def finalize_options(self):
        pass

    def run(self):
        subprocess.run(["codecov", "-t", self.codecov_token])


class CondaBuild(setuptools.Command):
    user_options = [
        ("build-dir=", None, "Build directory which will contain built conda package."),
        ("channels=", None, "Conda channels from which to install dependencies"),
        ("recipe=", None, "Directory containing conda recipe."),
    ]

    def initialize_options(self):
        self.build_dir = None
        self.channels = None
        self.recipe = None

    def finalize_options(self):
        pass

    def hash(self):
        metadata = self.distribution.metadata
        # adapted from https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
        m = hashlib.sha256()

        with open(
            pathlib.Path("dist/" + metadata.name + "-" + metadata.version + ".tar.gz"),
            "rb",
        ) as f:
            while True:
                data = f.read(65536)
                if not data:
                    break
                m.update(data)

        return m.hexdigest()

    def run(self):
        channels = " ".join(
            f"-c {c.replace(' ','')}"
            for c in self.channels.splitlines()
            if c.strip() != ""
        ).split()
        subprocess.run(
            ["conda", "build", self.recipe, "--output-folder", self.build_dir]
            + channels,
            env=dict(os.environ, PYPI_TARBALL_SHA_256=self.hash()),
        )


class DeployConda(setuptools.Command):
    user_options = [
        ("build-dir=", None, "Build directory containing built conda package."),
        ("username=", None, "Anaconda username for conda package upload"),
        ("password=", None, "Anaconda password for conda package upload"),
    ]

    def initialize_options(self):
        self.build_dir = None
        self.username = None
        self.password = None

    def finalize_options(self):
        pass

    def run(self):
        package_name = self.distribution.metadata.name
        yes = subprocess.Popen(["yes"], stdout=subprocess.PIPE)
        subprocess.run(
            [
                "anaconda",
                "login",
                "--username",
                self.username,
                "--password",
                self.password,
            ],
            stdin=yes.stdout,
        )

        pwd = pathlib.Path(self.build_dir).absolute()
        installed_package_info = subprocess.check_output(
            [
                "conda",
                "search",
                package_name,
                "--offline",
                "-c",
                "file://" + str(pwd),
                "--json",
            ]
        )
        [d,] = json.loads(installed_package_info)[package_name]
        url = re.match(r"file://(?P<url>.*)", d["url"]).group("url")
        subprocess.run(["anaconda", "upload", url])


class DeployPyPI(setuptools.Command):
    user_options = [
        ("pypi-token=", None, "Token for PyPI package upload"),
    ]

    def initialize_options(self):
        self.pypi_token = None

    def finalize_options(self):
        pass

    def run(self):
        subprocess.run(
            [
                "python",
                "-m",
                "twine",
                "upload",
                *pathlib.Path("dist").rglob("*"),
                "--username",
                "__token__",
                "--password",
                self.pypi_token,
                "--non-interactive",
                "--skip-existing",
            ]
        )


class Docs(setuptools.Command):
    user_options = [
        ("builder=", None, "FIXME"),
        ("extensions=", None, "FIXME"),
        ("params=", None, "FIXME"),
        ("template-dir=", None, "FIXME"),
    ]

    def initialize_options(self):
        self.builder = None
        self.extensions = None
        self.params = None
        self.template_dir = None

    def finalize_options(self):
        pass

    def run(self):
        metadata = self.distribution.metadata
        name = metadata.name
        author = metadata.author
        version = metadata.version

        extensions = ",".join(
            ext.replace(" ", "")
            for ext in self.extensions.splitlines()
            if ext.strip() != ""
        )
        params = " ".join(
            f"-d {ps.replace(' ','')}"
            for ps in self.params.splitlines()
            if ps.strip() != ""
        ).split()

        subprocess.run(
            [
                "sphinx-quickstart",
                "docs",
                "--sep",
                "-p",
                name,
                "-a",
                author,
                "-r",
                version,
                "-l",
                "en",
                "--extensions",
                extensions,
                "-t",
                pathlib.Path(self.template_dir),
                "-d",
                "module_path=" + str(pathlib.Path("src").resolve()),
                "-d",
                "name=" + name,
                "-d",
                "year=" + str(datetime.datetime.now().year),
                "-d",
                "author=" + author,
                "-d",
                "version=" + version,
                "-d",
                "release=" + version,
            ]
            + params
        )

        subprocess.run(
            [
                "sphinx-apidoc",
                pathlib.Path("src/" + name).resolve(),
                "-f",
                "-e",
                "-o",
                pathlib.Path("docs/source"),
                "-t",
                pathlib.Path(self.template_dir),
            ]
        )
        subprocess.run(["make", "-C", "docs", "clean"] + self.builder.split())


class Env(setuptools.Command):
    user_options = [
        ("channels=", None, "Conda channels from which to install dependencies"),
        ("env-name=", None, "Name of conda environment to be created"),
    ]

    def initialize_options(self):
        self.channels = None
        self.env_name = None

    def finalize_options(self):
        pass

    def run(self):
        package_name = self.distribution.metadata.name

        options = setuptools.config.read_configuration("setup.cfg").get("options")
        deps = [
            "python" + str(options.get("python_requires")),
            *options.get("install_requires"),
            *options.get("tests_require"),
            *options.get("extras_require")["dev"],
        ]

        channels = " ".join(
            f"-c {c.replace(' ','')}"
            for c in self.channels.splitlines()
            if c.strip() != ""
        ).split()

        subprocess.run(["conda", "create", "-n", self.env_name, "-y"] + deps + channels)


class InstallLocal(setuptools.Command):
    user_options = [
        (
            "build-dir=",
            None,
            "Build directory containing built conda package. Ignored if --editable is present.",
        ),
        ("channels=", None, "Conda channels from which to install dependencies"),
        ("editable", None, "If true, install editable package using conda develop."),
    ]

    def initialize_options(self):
        self.build_dir = None
        self.channels = None
        self.editable = False

    def finalize_options(self):
        pass

    def run(self):
        if self.editable:
            subprocess.run(["conda", "develop", "src"])
        else:
            package_name = self.distribution.metadata.name
            installed_package_info = subprocess.check_output(
                ["conda", "list", package_name, "--json"]
            )
            if json.loads(installed_package_info) == []:
                pwd = pathlib.Path(self.build_dir).absolute()
                channels = " ".join(
                    f"-c {c.replace(' ','')}"
                    for c in self.channels.splitlines()
                    if c.strip() != ""
                ).split()
                subprocess.run(
                    ["conda", "install", package_name, "-c", "file://" + str(pwd), "-y"]
                    + channels
                )


class Lint(setuptools.Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        subprocess.run(["black", "src", "tests", "setup.py"])


class Tag(setuptools.Command):
    user_options = [
        ("remote=", None, "Name of remote git repository to which tag will be pushed.")
    ]

    def initialize_options(self):
        self.remote = None

    def finalize_options(self):
        pass

    def run(self):
        version = self.distribution.metadata.version
        tag = "v" + version
        subprocess.run(["git", "tag", "-a", tag, "-m", "'" + tag + "'"])
        subprocess.run(["git", "push", self.remote, tag])


class Test(setuptools.Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        subprocess.check_call(["pytest"])


class UpdateVersion(setuptools.Command):
    user_options = [
        ("type=", None, "     Version number update type: major, minor, or patch.")
    ]

    def initialize_options(self):
        self.type = None

    def finalize_options(self):
        pass

    def major(self, m):
        s, *_ = m.groups()
        return f"{int(s)+1}.0.0"

    def minor(self, m):
        s1, s2, *_ = m.groups()
        return f"{s1}.{int(s2)+1}.0"

    def patch(self, m):
        s1, s2, s3 = m.groups()
        return f"{s1}.{s2}.{int(s3)+1}"

    def run(self):
        version = self.distribution.metadata.version
        v = re.sub(r"(\d+)\.(\d+)\.(\d+)", getattr(self, self.type), version)
        cfg_parser = ConfigParser()
        cfg_parser.read("setup.cfg")
        cfg_parser["metadata"]["version"] = v
        with open("setup.cfg", "w") as configfile:
            cfg_parser.write(configfile)


# --------------------------------------- #

setuptools.setup(
    cmdclass={
        "clean": Clean,
        "codecov": Codecov,
        "conda_build": CondaBuild,
        "deploy_conda": DeployConda,
        "deploy_pypi": DeployPyPI,
        "docs": Docs,
        "env": Env,
        "install_local": InstallLocal,
        "lint": Lint,
        "tag": Tag,
        "test": Test,
        "version": UpdateVersion,
    }
)
