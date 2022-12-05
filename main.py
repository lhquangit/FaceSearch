import re
from pathlib import Path

import towhee

import embedding

DATASET = Path("dataset/rikai/")


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
        .glob["path"](str(DATASET / "**/*.png"))
        .extract_fields["path", ("company", "name", "position")]()
        .image_decode["path", "image"]()
        .extract_embedding["image", "embedding"]()
    )

    for sample in dataset:
        # print(f"{sample.company:<15}{sample.name:<25}{sample.position:<15}")
        print(sample.embedding.shape)


if __name__ == "__main__":
    main()
