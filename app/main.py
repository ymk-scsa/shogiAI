import typer
from typing_extensions import Annotated

from app.interfaces.logger import Logger
from app.interfaces.cli import cli_app

app = typer.Typer()


app.add_typer(cli_app)


@app.command()
def test_log(message: Annotated[str, typer.Option(help="test text in log")]) -> None:
    logger_1 = Logger("test_log_1").get_logger()
    logger_1.info(message)
    logger_2 = Logger("test_log_2").get_logger()
    logger_2.info(message)


@app.command()
def main() -> None:
    # typerコマンド用に仮置き
    logger = Logger("main").get_logger()
    logger.debug("start main")


if __name__ == "__main__":
    app()
