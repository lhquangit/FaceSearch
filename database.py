from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)


def main() -> None:
    connections.connect("default", host="localhost", port="19530")
    fields = [
        FieldSchema(name="path", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema(name="company", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="position", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="image", dtype=DataType.FLOAT_VECTOR, dim=512),
    ]
    schema = CollectionSchema(fields)
    collection = Collection("rikai", schema)


if __name__ == "__main__":
    main()
