from packaging.version import parse
from packaging.specifiers import Specifier
import os
import pathlib
import re
import unittest
import yaml

from datetime import datetime, timedelta


def _find_setup_file():
    openmdao_dir = os.environ.get('OPENMDAO_DIR', None)
    if openmdao_dir is None:
        this_dir = pathlib.Path(__file__).parent
        while True:
            if (pathlib.Path(this_dir) / '.github').is_dir():
                return pathlib.Path(this_dir) / 'setup.py'
            elif this_dir == this_dir.parent:
                break
            else:
                this_dir = this_dir.parent
    return None


def _find_ci_workflow_file():
    openmdao_dir = os.environ.get('OPENMDAO_DIR', None)
    if openmdao_dir is None:
        this_dir = pathlib.Path(__file__).parent
        while True:
            if (pathlib.Path(this_dir) / '.github').is_dir():
                return pathlib.Path(this_dir) / '.github' / 'workflows' / 'openmdao_test_workflow.yml'
            elif this_dir == this_dir.parent:
                break
            else:
                this_dir = this_dir.parent
    return None


def _get_supported_versions():
    """
    Generates lists of supported versions of Python and Numpy, per the documentation from NEP-0029.
    """

    data = """Jan 15, 2017: NumPy 1.12
    Sep 13, 2015: Python 3.5
    Dec 23, 2016: Python 3.6
    Jun 27, 2018: Python 3.7
    Jun 07, 2017: NumPy 1.13
    Jan 06, 2018: NumPy 1.14
    Jul 23, 2018: NumPy 1.15
    Jan 13, 2019: NumPy 1.16
    Jul 26, 2019: NumPy 1.17
    Oct 14, 2019: Python 3.8
    Dec 22, 2019: NumPy 1.18
    Jun 20, 2020: NumPy 1.19
    Oct 05, 2020: Python 3.9
    Jan 30, 2021: NumPy 1.20
    Jun 22, 2021: NumPy 1.21
    Oct 04, 2021: Python 3.10
    Dec 31, 2021: NumPy 1.22
    """

    releases = []

    plus42 = timedelta(days=int(365 * 3.5 + 1))
    plus24 = timedelta(days=int(365 * 2 + 1))

    for line in data.splitlines():
        if len(line.strip()) == 0:
            break
        date, project_version = line.split(':')
        project, version = project_version.strip().split(' ')
        release = datetime.strptime(date.strip(), '%b %d, %Y')
        if project.lower() == 'numpy':
            drop = release + plus24
        else:
            drop = release + plus42
        releases.append((drop, project, version, release))

    releases = sorted(releases, key=lambda x: x[0])

    python_versions = []
    numpy_versions = []

    for d, p, v, r in releases[::-1]:
        if datetime.now() < d:
            if p.lower() == 'python':
                python_versions.append(parse(v))
            elif p.lower() == 'numpy':
                numpy_versions.append(parse(v))

    return sorted(python_versions), sorted(numpy_versions)


class TestSetupDependencies(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.supported_python_versions, cls.supported_numpy_versions = _get_supported_versions()

        cls.setup_file = _find_setup_file()

        if cls.setup_file is None:
            cls.skipTest(cls, 'Unable to determine path of OpenMDAO setup file.')

        with open(cls.setup_file) as f:
            cls.setup_data = f.read()

    def test_setup_python_version(self):
        """ Test that required python version in setup.py is a supported python version. """
        match = re.search('python_requires="(.+)"', self.setup_data)
        required_python = Specifier(match.groups()[0])

        self.assertIn(parse(required_python.version), self.supported_python_versions)

    def test_setup_numpy_version(self):
        """ Test that required numpy version in setup.py is a supported numpy version. """
        match = re.search('install_requires=\[(.+?)\]', self.setup_data, re.MULTILINE | re.DOTALL)
        requires = [s.strip().replace("'", "") for s in match.groups()[0].split(',')]

        for s in requires:
            if s.startswith('numpy'):
                numpy_min_ver = f'>={self.supported_numpy_versions[0]}' if '>' not in s else s[5:]
        required_numpy = Specifier(numpy_min_ver)

        self.assertIn(parse(required_numpy.version), self.supported_numpy_versions)


class TestCIDependencies(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.supported_python_versions, cls.supported_numpy_versions = _get_supported_versions()

        workflow_file = _find_ci_workflow_file()

        with open(workflow_file) as f:
            ci_data = cls.workflow_yaml = yaml.load(f)

        matrix = {}

        for job in ci_data['jobs']:
            matrix[job] = {}
            try:
                for entry in ci_data['jobs'][job]['strategy']['matrix']['include']:
                    matrix[job][entry['NAME'].lower()] = {}
                    numpy_ver = parse(str(entry['NUMPY']))
                    python_ver = parse(str(entry['PY']))
                    matrix[job][entry['NAME'].lower()]['numpy'] = numpy_ver
                    matrix[job][entry['NAME'].lower()]['python'] = python_ver
            except KeyError:
                continue

        cls.matrix = matrix


    def test_workflow_versions(self):
        for job in self.matrix:
            baseline_py = baseline_numpy = oldest_py = oldest_numpy = None
            if 'baseline' in self.matrix[job]:
                baseline_py = self.matrix[job]['baseline']['python']
                baseline_numpy = self.matrix[job]['baseline']['numpy']

            if 'oldest' in self.matrix[job]:
                oldest_py = self.matrix[job]['oldest']['python']
                oldest_numpy = self.matrix[job]['oldest']['numpy']

            if oldest_py is not None:
                self.assertGreaterEqual(oldest_py,
                                        self.supported_python_versions[0],
                                        msg=f'Oldest supported python is {self.supported_python_versions[0]} but '
                                            f'oldest matrix entry for job {job} has python version as {oldest_py}.')

            if oldest_numpy is not None:
                self.assertGreaterEqual(oldest_numpy,
                                        self.supported_numpy_versions[0],
                                        msg=f'Oldest supported numpy is {self.supported_numpy_versions[0]} but oldest '
                                            f'matrix entry for job {job} has numpy version as {oldest_numpy}.')

            if baseline_py is not None and '.' in str(baseline_py):
                self.assertGreaterEqual(baseline_py,
                                        self.supported_python_versions[0],
                                        msg=f'Baseline python version for job {job} is {baseline_py} but oldest '
                                            f'supported python version is {self.supported_python_versions[0]}.')

            if baseline_numpy is not None and '.' in str(baseline_numpy):
                self.assertGreaterEqual(baseline_numpy,
                                        self.supported_numpy_versions[0],
                                        msg=f'Baseline numpy version for job {job} is {baseline_numpy} but oldest '
                                            f'supported numpy version is {self.supported_numpy_versions[0]}.')
