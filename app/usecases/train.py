import torch
import torch.optim as optim
from typing import Optional

from app.interfaces.logger import Logger
from app.domain.policy_value_network import PolicyValueNetwork
from app.infrastructure.dataloader import HcpeDataLoader
from typing_extensions import Annotated
import typer

train_app = typer.Typer()


@train_app.command()
def train(
    train_data: Annotated[list[str], typer.Option(help="training data file")],
    test_data: Annotated[str, typer.Option(help="test data file")],
    gpu: Annotated[int, typer.Option("-g", help="GPU ID")] = 0,
    epoch: Annotated[int, typer.Option("-e", help="Number of epoch times")] = 1,
    batchsize: Annotated[int, typer.Option("-b", help="Number of positions in each mini-batch")] = 1024,
    testbatchsize: Annotated[int, typer.Option(help="Number of positions in each test mini-batch")] = 1024,
    lr: Annotated[float, typer.Option(help="learning rate")] = 0.01,
    checkpoint: Annotated[str, typer.Option(help="checkpoint file name")] = "checkpoints/checkpoint-{epoch:03}.pth",
    resume: Annotated[str, typer.Option("-r", help="Resume from snapshot")] = "",
    eval_interval: Annotated[int, typer.Option(help="evaluation interval")] = 100,
    log: Annotated[Optional[str], typer.Option(help="log file path")] = None,
) -> None:
    """Train policy value network"""

    logging = Logger("train", log_file=log).get_logger()
    logging.info("batchsize={}".format(batchsize))
    logging.info("lr={}".format(lr))

    # デバイス
    if gpu >= 0:
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    # モデル
    model = PolicyValueNetwork()
    model.to(device)

    # オプティマイザ
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)

    # 損失関数
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()

    # チェックポイント読み込み
    if resume:
        logging.info("Loading the checkpoint from {}".format(resume))
        checkpoint_data = torch.load(resume, map_location=device)
        epoch = int(checkpoint_data["epoch"])
        t = int(checkpoint_data["t"])
        model.load_state_dict(checkpoint_data["model"])
        optimizer.load_state_dict(checkpoint_data["optimizer"])
        # 学習率を引数の値に変更
        optimizer.param_groups[0]["lr"] = lr
    else:
        epoch = 0
        t = 0  # total steps

    # 訓練データ読み込み
    logging.info("Reading training data")
    train_dataloader = HcpeDataLoader(train_data, batchsize, device, shuffle=True)
    # テストデータ読み込み
    logging.info("Reading test data")
    test_dataloader = HcpeDataLoader(test_data, testbatchsize, device)

    # 読み込んだデータ数を表示
    logging.info("train position num = {}".format(len(train_dataloader)))
    logging.info("test position num = {}".format(len(test_dataloader)))

    # 方策の正解率
    def accuracy(y: torch.Tensor, t: torch.Tensor) -> float:
        return (torch.max(y, 1)[1] == t).sum().item() / len(t)

    # 価値の正解率
    def binary_accuracy(y: torch.Tensor, t: torch.Tensor) -> float:
        pred = y >= 0
        truth = t >= 0.5
        return pred.eq(truth).sum().item() / len(t)

    # チェックポイント保存
    def save_checkpoint(checkpoint_path: str) -> None:
        path = checkpoint_path.format(epoch=epoch, step=t)
        logging.info("Saving the checkpoint to {}".format(path))
        checkpoint_data = {
            "epoch": epoch,
            "t": t,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint_data, path)

    # 訓練ループ
    for e in range(epoch):
        # 初期化処理
        epoch += 1
        steps_interval = 0
        sum_loss_policy_interval = 0
        sum_loss_value_interval = 0
        steps_epoch = 0
        sum_loss_policy_epoch = 0
        sum_loss_value_epoch = 0
        # データごとに繰り返す
        for x, move_label, result in train_dataloader:
            model.train()

            # 順伝播
            y1, y2 = model(x)
            # 損失計算
            loss_policy = cross_entropy_loss(y1, move_label)
            loss_value = bce_with_logits_loss(y2, result)
            loss = loss_policy + loss_value
            # 誤差逆伝播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # トータルステップ数に加算
            t += 1

            # 評価間隔ごとのステップ数カウンタと損失合計に加算
            steps_interval += 1
            sum_loss_policy_interval += loss_policy.item()
            sum_loss_value_interval += loss_value.item()

            # 評価間隔ごとに訓練損失とテスト損失・正解率を表示
            if t % eval_interval == 0:
                model.eval()

                x, move_label, result = test_dataloader.sample()
                with torch.no_grad():
                    # 推論
                    y1, y2 = model(x)
                    # 損失計算
                    test_loss_policy = cross_entropy_loss(y1, move_label).item()
                    test_loss_value = bce_with_logits_loss(y2, result).item()
                    # 正解率計算
                    test_accuracy_policy = accuracy(y1, move_label)
                    test_accuracy_value = binary_accuracy(y2, result)

                    # ログ表示
                    logging.info(
                        "epoch = {}, steps = {}, train loss = {:.07f}, {:.07f}, {:.07f}, test loss = {:.07f}, {:.07f}, {:.07f}, test accuracy = {:.07f}, {:.07f}".format(
                            epoch,
                            t,
                            sum_loss_policy_interval / steps_interval,
                            sum_loss_value_interval / steps_interval,
                            (sum_loss_policy_interval + sum_loss_value_interval) / steps_interval,
                            test_loss_policy,
                            test_loss_value,
                            test_loss_policy + test_loss_value,
                            test_accuracy_policy,
                            test_accuracy_value,
                        )
                    )

                # エポックごとのステップ数カウンタと損失合計に加算
                steps_epoch += steps_interval
                sum_loss_policy_epoch += sum_loss_policy_interval
                sum_loss_value_epoch += sum_loss_value_interval

                # 評価間隔ごとのステップ数カウンタと損失合計をクリア
                steps_interval = 0
                sum_loss_policy_interval = 0
                sum_loss_value_interval = 0

        # エポックごとのステップ数カウンタと損失合計に加算
        steps_epoch += steps_interval
        sum_loss_policy_epoch += sum_loss_policy_interval
        sum_loss_value_epoch += sum_loss_value_interval

        # エポックの終わりにテストデータすべてを使用して評価する
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

        # テストデータの検証結果をログ表示
        logging.info(
            "epoch = {}, steps = {}, train loss avr = {:.07f}, {:.07f}, {:.07f}, test loss = {:.07f}, {:.07f}, {:.07f}, test accuracy = {:.07f}, {:.07f}".format(
                epoch,
                t,
                sum_loss_policy_epoch / steps_epoch,
                sum_loss_value_epoch / steps_epoch,
                (sum_loss_policy_epoch + sum_loss_value_epoch) / steps_epoch,
                sum_test_loss_policy / test_steps,
                sum_test_loss_value / test_steps,
                (sum_test_loss_policy + sum_test_loss_value) / test_steps,
                sum_test_accuracy_policy / test_steps,
                sum_test_accuracy_value / test_steps,
            )
        )

        # チェックポイント保存
        if checkpoint:
            save_checkpoint(checkpoint)
            # dlshogi 1/6


if __name__ == "__main__":
    train_app()
