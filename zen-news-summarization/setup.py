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
    requires=[
        "zenml[server]==0.32.1",
        "bbc-feeds==2.1",
        "transformers==4.26.0",
        "torch==1.13.1",
        "pydantic==1.9.2",
        "click==8.1.3",
        "rich==12.6.0"
    ]
)
