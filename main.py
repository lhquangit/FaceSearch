import re
from pathlib import Path

import torch
import towhee


DATA_DIR = Path("data/rikai/")
IS_CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if IS_CUDA_AVAILABLE else "cpu"


@towhee.register("extract_fields")
def extract_fields(path: str) -> tuple[str, str, str]:
    dir_name = Path(path).parent.name
    fields = re.fullmatch(r"(.*) - (.*) - (.*)", dir_name)
    company = fields.group(1)  # type: ignore
    name = fields.group(2)  # type: ignore
    position = fields.group(3)  # type: ignore
    return company, name, position


def main() -> None:

    dataset = (
        towhee
        .glob["path"](str(DATA_DIR / "**/*.png"))
        .extract_fields["path", ("company", "name", "position", "lead", "intern")]()
        # .image_decode["path", "image"]()
    ) 

    for sample in dataset:
        print(f"{sample.company:<15}{sample.name:<25}{sample.position}")
    

if __name__ == '__main__':
    main()
