import typer
from app.domain.features import FEATURES_DEFAULT, FEATURES_KIKI
from app.interfaces.logger import Logger
from app.usecases.train import train_app
from app.usecases.test import test_app
from app.usecases.mcts_player import MCTSPlayer

cli_app = typer.Typer()


cli_app.add_typer(train_app)
cli_app.add_typer(test_app)


@cli_app.command()
def play_person() -> None:
    logging = Logger("cli_train").get_logger()
    logging.debug("start train")


@cli_app.command()
def play_mcts() -> None:
    player = MCTSPlayer(features_mode=FEATURES_DEFAULT)
    player.run()


@cli_app.command()
def play_mcts_kiki() -> None:
    player = MCTSPlayer(features_mode=FEATURES_KIKI)
    player.run()


if __name__ == "__main__":
    cli_app()
