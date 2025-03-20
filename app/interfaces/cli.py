import typer
from app.interfaces.logger import Logger
from app.usecases.train import train_app

cli_app = typer.Typer()


cli_app.add_typer(train_app)


@cli_app.command()
def play_person() -> None:
    logging = Logger("cli_train").get_logger()
    logging.debug("start train")


if __name__ == "__main__":
    cli_app()
