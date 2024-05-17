import polars as pl
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def preprocess_markdown_texts(markdown_texts: pl.DataFrame) -> pl.DataFrame:
    """Preprocesses markdown texts."""
    # process the texts in the 'content' column.
    # for each one, strip out any line that begins with `![`.
    # Also strip out the chunk between triple backticks that start with ```bash
    # or ```shell.
    markdown_texts = markdown_texts.with_columns(
        pl.col("page_content")
        .str.replace_all(r"!\[.*\]\(.*\)", "")
        .alias("page_content")
    )
    markdown_texts = markdown_texts.with_columns(
        pl.col("page_content")
        .str.replace_all(r"```(bash|shell)(.|\n)*?```", "")
        .alias("page_content")
    )

    return markdown_texts
