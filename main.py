from pathlib import Path

import torch
import towhee


DATASET = Path("datasets/rikai/")
IS_CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if IS_CUDA_AVAILABLE else "cpu"


def main() -> None:

    dataset = (
        towhee
        .glob(str(DATASET / "**/*"))
        .image_decode()
        .map(torch.from_numpy)
        .map(lambda image: image.to(DEVICE))
        .map(lambda image: image.permute(2, 0, 1))
    )

    for sample in dataset:
        print(sample.shape, sample.device, sep="\t")


if __name__ == '__main__':
    main()
