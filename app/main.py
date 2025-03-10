import typer
from typing_extensions import Annotated

from app.interfaces.logger import Logger
from app.interfaces.cli import cli_app

app = typer.Typer()


app.add_typer(cli_app)


@app.command()
def test_log(message: Annotated[str, typer.Option(help="test text in log")]):
    logger = Logger("test_log").get_logger()
    logger.info(message)


@app.command()
def main():
    # typerコマンド用に仮置き
    logger = Logger("main")
    logger.debug("start main")


if __name__ == "__main__":
    app()
