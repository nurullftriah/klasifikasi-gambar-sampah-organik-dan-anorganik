ORGANIC_CLASSNAMES = {"biological"}

def map_to_binary(folder_name: str) -> int:
    return 1 if folder_name.lower().strip() in ORGANIC_CLASSNAMES else 0
