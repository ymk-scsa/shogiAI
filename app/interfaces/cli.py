import typer
from typing_extensions import Annotated
from app.domain.features import FEATURES_KIKI, FEATURES_SMALL
from app.interfaces.logger import Logger
from app.usecases.train import train_app
from app.usecases.mcts_player import MCTSPlayer

cli_app = typer.Typer()


cli_app.add_typer(train_app)


@cli_app.command()
def play_person() -> None:
    logging = Logger("cli_train").get_logger()
    logging.debug("start train")


@cli_app.command()
def play_mcts(
    input_features: Annotated[
        int, typer.Option("-i", help="select custom input features mode (default: 0, kiki: 1, himo: 2)")
    ] = 0,
    activation_function: Annotated[
        int, typer.Option("-a", help="select custom input features mode (relu: 0, : 1)")
    ] = 0,
) -> None:
    player = MCTSPlayer(features_mode=input_features, activation_function_mode=activation_function)
    player.run()


@cli_app.command()
def play_mcts_kiki() -> None:
    player = MCTSPlayer(features_mode=FEATURES_KIKI)
    player.run()


@cli_app.command()
def play_mcts_small() -> None:
    player = MCTSPlayer(features_mode=FEATURES_SMALL)
    player.run()


if __name__ == "__main__":
    cli_app()
