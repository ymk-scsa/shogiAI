import typer
import torch

app = typer.Typer()


@app.command()
def check_gpu() -> None:
    print("torch version: ", torch.__version__)
    print("cuda is available: ", torch.cuda.is_available())
    print("cuda device_count: ", torch.cuda.device_count())
    print("gpu current info: ", torch.cuda.current_device(), torch.cuda.get_device_name())
    print("gpu device capability: ", torch.cuda.get_device_capability())
    x = torch.rand(5, 3)
    print("test result\n", x)


if __name__ == "__main__":
    app()
