from pymilvus import (
    DataType,
    Collection,
    CollectionSchema,
    FieldSchema,
    utility,
)


def create_collection(collection_name: str, dim: int | tuple[int]):
    if utility.has_collection(collection_name):
        return Collection(name=collection_name)

    fields = [
        FieldSchema(
            name="path", dtype=DataType.VARCHAR, max_length=512, is_primary=True
        ),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields=fields)
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 2048},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection
