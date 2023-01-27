from setuptools import setup

from cli.constants import APP_NAME
setup(
    name="ZenNews",
    description="ZenNews summarizations.",
    packages=['cli'],
    version='1.0',
    entry_points=f'''
        [console_scripts]
        {APP_NAME}=cli.base:cli
    ''',
)