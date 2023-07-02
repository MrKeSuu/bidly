
def in_default(item, collection, default=True):
    if collection:
        return item in collection
    return default