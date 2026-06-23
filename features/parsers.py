import pandas as pd


def parse_skills(s: str) -> set[str]:
    """Parse a pipe-delimited skill string into a lowercase set."""
    if pd.isna(s) or str(s).strip() == "":
        return set()
    return set(str(s).strip().lower().split("|"))


def seniority(title: str) -> int:
    """Map a job title to a coarse seniority score 1-4."""
    t = str(title).lower()
    if any(w in t for w in ("junior", "jr", "entry", "intern", "trainee")):
        return 1
    if any(w in t for w in ("senior", "sr", "lead", "principal", "staff")):
        return 3
    if any(w in t for w in ("manager", "director", "head", "vp", "chief", "cto", "ceo")):
        return 4
    return 2


def parse_edu(s: str) -> int:
    """Map an education-level string to a numeric score 1-4."""
    s = str(s).lower()
    if any(x in s for x in ("phd", "ph.d", "doctorate")):
        return 4
    if any(x in s for x in ("master", "m.s", "m.a", "msc")):
        return 3
    if any(x in s for x in ("bachelor", "b.a", "bsc", "b.s")):
        return 2
    if any(x in s for x in ("high school", "secondary", "hs")):
        return 1
    return 2  # default to Bachelor


def parse_eng(s: str) -> int:
    """Map an English-proficiency string to a numeric score 1-6."""
    s = str(s).lower()
    if any(x in s for x in ("c2", "native", "bilingual", "fluent")):
        return 6
    if any(x in s for x in ("c1", "advanced")):
        return 5
    if any(x in s for x in ("b2", "upper-intermediate")):
        return 4
    if any(x in s for x in ("b1", "intermediate")):
        return 3
    if any(x in s for x in ("a2", "elementary")):
        return 2
    if any(x in s for x in ("a1", "beginner")):
        return 1
    return 3  # default to Intermediate


def parse_size(s: str) -> int:
    """Map a company-size string to an ordinal 1-5."""
    s = str(s).lower().replace(" ", "")
    if "1000+" in s or "10000+" in s:
        return 5
    if "201-1000" in s:
        return 4
    if "51-200" in s:
        return 3
    if "11-50" in s:
        return 2
    if "1-10" in s:
        return 1
    return 3


def parse_remote(s: str) -> int:
    """Map remote-allowed string: 0 = on-site, 1 = hybrid, 2 = remote."""
    s = str(s).lower()
    if "hybrid" in s:
        return 1
    if "yes" in s or "remote" in s:
        return 2
    return 0
