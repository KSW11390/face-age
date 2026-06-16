# faceage/data/utkface_dataset.py
AGE_GROUPS = [
    ("00-09", 0, 9),
    ("10-19", 10, 19),
    ("20-29", 20, 29),
    ("30-39", 30, 39),
    ("40-49", 40, 49),
    ("50-59", 50, 59),
    ("60-69", 60, 69),
    ("70-79", 70, 79),
    ("80-89", 80, 89),
]

def age_to_group_name(age: int) -> str:
    for name, lo, hi in AGE_GROUPS:
        if lo <= age <= hi:
            return name
    return "others"