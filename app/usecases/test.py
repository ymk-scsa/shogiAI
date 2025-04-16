import torch
import typer
from typing import Optional
from typing_extensions import Annotated
from app.interfaces.logger import Logger
from app.domain.policy_value_network import PolicyValueNetwork
from app.infrastructure.dataloader import HcpeDataLoader
from app.domain.features import FEATURES_SETTINGS

test_app = typer.Typer()

# 損失関数
cross_entropy_loss = torch.nn.CrossEntropyLoss()
bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()


# 方策の正解率
def accuracy(y: torch.Tensor, t: torch.Tensor) -> float:
    return (torch.max(y, 1)[1] == t).sum().item() / len(t)


# 価値の正解率
def binary_accuracy(y: torch.Tensor, t: torch.Tensor) -> float:
    pred = y >= 0
    truth = t >= 0.5
    return pred.eq(truth).sum().item() / len(t)


def test_model(model, test_dataloader):
    # テストデータすべてを使用して評価する
    test_steps = 0
    sum_test_loss_policy = 0
    sum_test_loss_value = 0
    sum_test_accuracy_policy: float = 0
    sum_test_accuracy_value: float = 0
    model.eval()
    with torch.no_grad():
        for x, move_label, result in test_dataloader:
            y1, y2 = model(x)

            test_steps += 1
            sum_test_loss_policy += cross_entropy_loss(y1, move_label).item()
            sum_test_loss_value += bce_with_logits_loss(y2, result).item()
            sum_test_accuracy_policy += accuracy(y1, move_label)
            sum_test_accuracy_value += binary_accuracy(y2, result)
    return {
        "test_steps": test_steps,
        "sum_test_loss_policy": sum_test_loss_policy,
        "sum_test_loss_value": sum_test_loss_value,
        "sum_test_accuracy_policy": sum_test_accuracy_policy,
        "sum_test_accuracy_value": sum_test_accuracy_value,
    }


def report_test_result(report: dict):
    test_steps = report.get("test_steps")
    sum_test_loss_policy = report.get("sum_test_loss_policy")
    sum_test_loss_value = report.get("sum_test_loss_value")
    sum_test_accuracy_policy = report.get("sum_test_accuracy_policy")
    sum_test_accuracy_value = report.get("sum_test_accuracy_value")

    return "test loss = {:.07f}, {:.07f}, {:.07f}, test accuracy = {:.07f}, {:.07f}".format(
        sum_test_loss_policy / test_steps,
        sum_test_loss_value / test_steps,
        (sum_test_loss_policy + sum_test_loss_value) / test_steps,
        sum_test_accuracy_policy / test_steps,
        sum_test_accuracy_value / test_steps,
    )


@test_app.command("test_model")
def test_model_cli(
    test_data: Annotated[str, typer.Option(help="test data file")],
    resume: Annotated[str, typer.Option("-r", help="Resume from snapshot")],
    gpu: Annotated[int, typer.Option("-g", help="GPU ID")] = 0,
    testbatchsize: Annotated[int, typer.Option(help="Number of positions in each test mini-batch")] = 1024,
    lr: Annotated[float, typer.Option(help="learning rate")] = 0.01,
    log: Annotated[Optional[str], typer.Option(help="log file path")] = None,
    limit: Annotated[Optional[int], typer.Option("-l", help="limit of test case")] = None,
    shuffle: Annotated[bool, typer.Option("-s", help="limit of test case")] = False,
    input_features: Annotated[
        int, typer.Option("-i", help="select custom input features mode (default: 0, kiki: 1, himo: 2)")
    ] = 0,
) -> None:
    logging = Logger("test", log_file=log).get_logger()

    # デバイス
    if gpu >= 0:
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    # モデル
    features_setting = FEATURES_SETTINGS[input_features]
    model = PolicyValueNetwork(input_features=features_setting.features_num)
    model.to(device)

    # テストデータ読み込み
    logging.info("Reading test data")
    test_dataloader = HcpeDataLoader(
        test_data, testbatchsize, device, features_mode=input_features, limit=limit, shuffle=shuffle
    )
    logging.info("test position num = {}".format(len(test_dataloader)))

    # チェックアウト
    logging.info("Loading the checkpoint from {}".format(resume))
    checkpoint_data = torch.load(resume, map_location=device)
    model.load_state_dict(checkpoint_data["model"])

    logging.info("Testing")
    logging.info(report_test_result(test_model(model, test_dataloader)))
