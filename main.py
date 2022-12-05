import re
from pathlib import Path
from collections import namedtuple

import towhee

import embedding

DATASET = Path("dataset/rikai/")


@towhee.register("extract_fields")
def extract_fields(path: str) -> tuple[str, str, str]:
    dir_name = Path(path).parent.name
    match_obj = re.fullmatch(r"(.*) - (.*) - (.*)", dir_name)

    field_names = ["company", "name", "position"]
    fields = namedtuple("fields", field_names)

    return fields(
        **{
            field_name: match_obj.group(idx)  # type: ignore
            for idx, field_name in enumerate(field_names, start=1)
        }
    )


def main() -> None:

    dataset = (
        towhee
        .glob["path"](str(DATASET / "**/*.png"))
        .extract_fields["path", ("company", "name", "position")]()
        .image_decode["path", "image"]()
        .extract_embedding["image", "embedding"]()
    )

    for sample in dataset:
        print(f"{sample.company:<15}{sample.name:<25}{sample.position:<15}")
        # print(sample.embedding.shape)


if __name__ == "__main__":
    main()
