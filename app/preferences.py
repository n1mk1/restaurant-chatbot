import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache

from langchain_core.messages import AnyMessage, HumanMessage


VERIFIED_DIETARY_ALIASES: dict[str, tuple[str, ...]] = {
    "vegan": ("vegan",),
    "vegetarian": ("vegetarian", "vegitarian", "meatless", "no meat", "do not eat meat"),
    "gluten-free": ("gluten free", "celiac", "coeliac"),
}

UNVERIFIED_DIETARY_ALIASES: dict[str, tuple[str, ...]] = {
    "halal": ("halal",),
    "kosher": ("kosher",),
    "Jain": ("jain",),
    "keto": ("keto", "ketogenic"),
    "pescatarian": ("pescatarian",),
    "paleo": ("paleo",),
    "Whole30": ("whole30", "whole 30"),
    "low-carb": ("low carb", "low carbohydrate"),
    "low-sodium": ("low sodium", "low salt", "high blood pressure"),
    "low-FODMAP": ("low fodmap", "fodmap"),
    "diabetic-friendly": ("diabetic friendly", "diabetes friendly", "diabetic", "diabetes"),
    "sugar-free": ("sugar free", "no added sugar"),
    "alcohol-free": ("alcohol free", "no alcohol"),
    "pork-free": ("pork free", "no pork"),
    "organic": ("organic",),
    "plant-based": ("plant based",),
}

TRACKED_ALLERGEN_ALIASES: dict[str, tuple[str, ...]] = {
    "dairy": (
        "dairy", "milk", "cheese", "goat cheese", "cheddar", "parmesan", "cream",
        "creme fraiche", "crème fraîche",
    ),
    "egg": ("egg yolk", "egg yolks", "egg white", "egg whites", "egg", "eggs"),
    "fish": ("fish", "salmon", "perch"),
    "gluten": ("gluten", "celiac", "coeliac"),
    "tree nuts": ("tree nut", "tree nuts", "walnut", "walnuts"),
}

UNTRACKED_ALLERGEN_ALIASES: dict[str, tuple[str, ...]] = {
    "unspecified nuts": ("nut", "nuts"),
    "peanuts": ("peanut", "peanuts"),
    "shellfish": ("shellfish", "shell fish"),
    "soy": ("soy",),
    "sesame": ("sesame",),
    "wheat": ("wheat",),
    "mustard": ("mustard",),
    "sulfites": ("sulfite", "sulfites"),
    "lactose": ("lactose",),
}


@dataclass(slots=True)
class PreferenceState:
    dietary: list[str] = field(default_factory=list)
    allergens: list[str] = field(default_factory=list)
    untracked_allergens: list[str] = field(default_factory=list)
    unverified_diets: list[str] = field(default_factory=list)

    def as_graph_input(self) -> dict[str, object]:
        return {
            "dietary_preferences": list(self.dietary),
            "allergen_restrictions": list(self.allergens),
            "untracked_allergen_restrictions": list(self.untracked_allergens),
            "unverified_dietary_restrictions": list(self.unverified_diets),
            "preferences_managed": True,
        }


@lru_cache(maxsize=4096)
def normalize(text: str) -> str:
    # Pure text→text transform called ~hundreds of times per turn (once per alias
    # term, per detector). Memoized: the guest message and the constant alias
    # terms are each normalized once, not re-derived on every comparison.
    text = text.lower().replace("’", "'")
    contractions = {
        r"\bi'm\b": "i am",
        r"\bwe're\b": "we are",
        r"\bi've\b": "i have",
        r"\bwe've\b": "we have",
        r"\bi'd\b": "i would",
        r"\bi'll\b": "i will",
        r"\bdon't\b": "do not",
        r"\bdoesn't\b": "does not",
        r"\bdidn't\b": "did not",
        r"\bisn't\b": "is not",
        r"\baren't\b": "are not",
        r"\bwasn't\b": "was not",
        r"\bweren't\b": "were not",
        r"\bhasn't\b": "has not",
        r"\bhaven't\b": "have not",
        r"\bcan't\b": "cannot",
        r"\bcouldn't\b": "could not",
        r"\bwon't\b": "will not",
        r"\bwouldn't\b": "would not",
        r"\bshouldn't\b": "should not",
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)
    return " ".join(re.findall(r"[a-z0-9]+", text))


def words(text: str) -> set[str]:
    return set(normalize(text).split())


def contains_term(text: str, term: str) -> bool:
    normalized_text = f" {normalize(text)} "
    normalized_term = normalize(term)
    return bool(normalized_term) and f" {normalized_term} " in normalized_text


def requested_labels(text: str, aliases: dict[str, tuple[str, ...]]) -> list[str]:
    return [
        label
        for label, terms in aliases.items()
        if any(contains_term(text, term) for term in terms)
    ]


def natural_join(values: Sequence[str], conjunction: str = "or") -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} {conjunction} {values[1]}"
    return f"{', '.join(values[:-1])}, {conjunction} {values[-1]}"


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def is_restriction_statement(text: str) -> bool:
    normalized = normalize(text)
    token_set = set(normalized.split())
    cues = {
        "allergy", "allergies", "allergic", "allergen", "allergens", "avoid", "avoiding", "without", "free",
        "intolerance", "intolerant", "sensitivity", "sensitive", "celiac", "coeliac",
        "hives", "anaphylaxis",
    }
    return bool(token_set & cues) or any(
        phrase in normalized
        for phrase in (
            "cannot eat", "cannot have", "cannot tolerate", "do not eat", "do not have",
            "do not drink", "only eat", "react to",
            "no dairy", "no egg", "no eggs",
            "no fish", "no gluten", "no nuts", "no peanuts", "no shellfish", "no shell fish",
            "makes me sick", "makes me ill", "gives me hives", "causes anaphylaxis",
            "reaction to", "react badly to", "causes a reaction", "causes hives",
            "hives from", "anaphylaxis from", "throat swell", "get sick from",
        )
    ) or bool(re.search(r"(?:^|\b(?:need|want|please|with|and|plus)\b)\s+no\s+[a-z]", normalized))


def is_personal_diet_statement(text: str) -> bool:
    normalized = normalize(text)
    token_set = set(normalized.split())
    has_person = bool(token_set & {"i", "im", "my", "me", "we", "our", "diner", "guest", "child"})
    has_cue = bool(
        token_set
        & {"am", "follow", "following", "eat", "need", "prefer", "preference", "diet", "only"}
    )
    has_named_diet = bool(
        requested_labels(text, VERIFIED_DIETARY_ALIASES)
        or requested_labels(text, UNVERIFIED_DIETARY_ALIASES)
    )
    question_like = "?" in text or normalized.startswith(
        ("do ", "does ", "did ", "is ", "are ", "what ", "which ", "can you ", "could you ")
    )
    imperative = bool(token_set & {"please", "only"}) or normalized.startswith(
        ("no ", "avoid ", "avoiding ", "without ")
    )
    shorthand = not question_like and has_named_diet and normalized.startswith(
        (
            "vegan", "vegetarian", "gluten free", "halal", "kosher", "jain", "keto",
            "pescatarian", "paleo", "whole30", "whole 30", "low ", "diabetic",
            "diabetes", "sugar free", "alcohol free", "pork free", "organic",
            "plant based", "also ", "and ", "plus ",
        )
    )
    return (has_person and (has_cue or has_named_diet)) or (
        not question_like and has_named_diet and imperative
    ) or shorthand


def is_allergen_content_question(text: str) -> bool:
    normalized = normalize(text)
    token_set = set(normalized.split())
    content_cues = {"contain", "contains", "include", "includes", "including", "has", "have", "with"}
    personal_declaration = normalized.startswith(("i have ", "we have ", "i am ", "im ", "my allergy"))
    return (
        bool(token_set & content_cues)
        or "made with" in normalized
        or "allergen in" in normalized
        or "allergens in" in normalized
        or bool(re.search(r"\b(?:is|are) there\b.+\bin\b", normalized))
        or bool(re.search(r"\b[a-z ]+\bin\b[a-z ]+", normalized))
    ) and not is_restriction_statement(text) and not personal_declaration


def _is_removal(
    text: str,
    aliases: tuple[str, ...],
    *,
    allow_tolerance: bool = False,
) -> bool:
    if "?" in text and not re.search(r"[.;!]\s*[^?]*\?", text):
        return False
    normalized = normalize(text)
    candidates = [normalized]
    for clause in re.split(r"\bbut\b|[,.;!]", text, flags=re.IGNORECASE):
        candidate = re.sub(r"^(?:and|but)\s+", "", normalize(clause))
        if candidate and candidate != normalized:
            candidates.append(candidate)
    for alias in aliases:
        term = normalize(alias)
        escaped = re.escape(term)
        patterns = (
            rf"^(?:actually )?(?:i am|im|we are|were) (?:no longer |not ){escaped}(?: anymore| now)?$",
            rf"^(?:actually )?not {escaped}$",
            rf"^(?:actually )?(?:i|we) (?:do not|no longer) need {escaped}(?: food| anymore| now)?$",
            rf"^(?:actually )?(?:i|we) no longer (?:have |follow |following )?(?:a |an )?{escaped}(?: allergy| diet| restriction)?$",
            rf"^(?:actually )?(?:i am|im|we are) no longer allergic to {escaped}$",
            rf"^no longer allergic to {escaped}$",
            rf"^(?:actually )?(?:i am|im|we are) not allergic to {escaped}$",
            rf"^(?:actually )?(?:i|we) do not have (?:a |an )?{escaped} allergy$",
            rf"^(?:i have )?no {escaped} allergy$",
            rf"^never mind (?:the )?{escaped}(?: restriction| allergy| diet)?$",
            rf"^remove {escaped} from my restrictions$",
            rf"^clear {escaped}$",
            rf"^(?:my )?{escaped} (?:allergy|restriction) (?:is )?(?:resolved|no longer applies)$",
            rf"^(?:my )?{escaped} intolerance (?:is )?resolved$",
            rf"^{escaped} no longer applies$",
            rf"^{escaped} no longer makes me sick$",
            rf"^{escaped} free (?:no longer applies|is not necessary|is not required)$",
            rf"^(?:i|we) (?:do not|no longer) need {escaped} free(?: food| options?| meals?)?$",
            rf"^no need for {escaped} free(?: food| options?| meals?)?$",
            rf"^(?:i|we) (?:do not eat|avoid) {escaped}(?: food)?$",
            rf"^(?:i|we) do not eat {escaped} free(?: food)?$",
            rf"^(?:i am|we are) no longer sensitive to {escaped}$",
            rf"^{escaped} (?:is|are) (?:fine|okay|ok)(?: now| for me| for us)?$",
            rf"^(?:actually )?(?:i|we) (?:switched|am switching) from {escaped} to\b.+$",
            rf"^(?:actually )?(?:i|we) stopped (?:being|following|eating) {escaped}$",
            rf"^(?:actually )?(?:i|we).+ instead of {escaped}$",
            rf"^(?:actually )?i meant .+\bnot {escaped}$",
        )
        if any(re.search(pattern, candidate) for pattern in patterns for candidate in candidates):
            return True
        if allow_tolerance and any(
            re.search(
                rf"^(?:actually )?(?:i|we) can (?:now )?(?:eat|have|tolerate) {escaped}(?: now| again| anymore)?$",
                candidate,
            )
            for candidate in candidates
        ):
            return True
        collective = re.search(r"^(?:actually )?(.+) are (?:fine|okay|ok) now$", normalized)
        if collective and contains_term(collective.group(1), alias):
            return True
        if f"used to be {term}" in normalized and "not anymore" in normalized:
            return True
    return False


def is_label_removal(
    text: str,
    aliases: tuple[str, ...],
    *,
    allow_tolerance: bool = False,
) -> bool:
    """Return whether the guest is explicitly removing a particular constraint."""
    return _is_removal(text, aliases, allow_tolerance=allow_tolerance)


def _is_query_operation(text: str) -> bool:
    normalized = normalize(text)
    return "?" in text or normalized.startswith(
        (
            "do you ", "does ", "did ", "is ", "are ", "what ", "which ", "can you ",
            "could you ", "would you ", "do i ", "do we ", "should i ", "should we ",
            "am i ", "are we ", "tell me ", "show me ", "list ", "compare ",
        )
    )


def _has_personal_constraint_clause(text: str) -> bool:
    scope = _restriction_scope(text)
    if not scope.startswith(("i ", "im ", "we ", "my ", "our ")):
        return False
    if requested_labels(scope, VERIFIED_DIETARY_ALIASES) or requested_labels(
        scope, UNVERIFIED_DIETARY_ALIASES
    ):
        return True
    return any(
        cue in f" {scope} "
        for cue in (
            " allergy ", " allergies ", " allergic ", " intolerant ", " intolerance ",
            " cannot eat ", " cannot have ", " do not eat ", " do not have ", " need ",
            " celiac ", " coeliac ",
        )
    )


def _restriction_scope(text: str) -> str:
    raw_clauses = [clause for clause in re.split(r"[.;!]", text) if normalize(clause)]
    constraint_clauses = [
        normalize(clause)
        for clause in raw_clauses
        if is_restriction_statement(clause)
        or any(cue in normalize(clause) for cue in ("i meant", "also ", "remove ", "is fine", "is okay"))
    ]
    normalized = " and ".join(constraint_clauses) if constraint_clauses else normalize(text)
    transitions = (
        " and can i ", " and could i ", " and do you ", " and does ", " and what ",
        " and which ", " and show ", " and recommend ", " and i want ", " and want ",
        " but can i ", " but i want ", " then show ", " then recommend ",
        " what can i ", " what do you ", " which dishes ", " which items ",
    )
    cut = len(normalized)
    for marker in transitions:
        position = normalized.find(marker)
        if position >= 0:
            cut = min(cut, position)
    return normalized[:cut].strip()


def _restriction_targets(scope: str) -> str:
    targets: list[str] = []
    patterns = (
        r"\ballergic to\s+(.+?)(?=\s+\bbut\b|$)",
        r"\ballerg(?:y|ies) to\s+(.+)",
        r"\ballerg(?:y|ies) (?:is|are)\s+(.+)",
        r"^allerg(?:y|ies)\s+(.+)",
        r"\bmy allergens are\s+(.+)",
        r"\bmy allergy list is\s+(.+)",
        r"\b(?:cannot eat|cannot have|cannot tolerate|do not eat|do not have|do not drink|avoid|avoiding|without)\s+(.+?)(?=\s+\bbut\b|$)",
        r"\b(?:am|are|feel) sensitive to\s+(.+)",
        r"\bhave (?:a |an )?(.+?)\s+sensitivity\b",
        r"\breact to\s+(.+)",
        r"\breact badly to\s+(.+)",
        r"\b(?:have|had) (?:a )?(?:severe )?reaction to\s+(.+)",
        r"\bintolerant to\s+(.+)",
        r"\bintolerance to\s+(.+)",
        r"(?:^|\band\b|\bbut\b)\s*(?:i am |we are )?(.+?)\s+intolerant\b",
        r"\bno(?!\s+longer\b)\s+(.+)",
        r"\bnon\s+(.+?)(?:\s+please|\s+options?|\s+food|$)",
        r"\b(?:i|we)\b.+?(?<!do )\bnot\s+(?!allergic\b)(.+?)(?:\s+please|\s+options?|\s+food|$)",
        r"(.+?)\s+makes me (?:sick|ill)\b",
        r"(.+?)\s+gives me hives\b",
        r"(.+?)\s+causes (?:anaphylaxis|hives|an allergic reaction)\b",
        r"(.+?)\s+causes (?:a )?reaction\b",
        r"(?:i|we) (?:get|break out in) hives from\s+(.+)",
        r"(?:i|we) get anaphylaxis from\s+(.+)",
        r"(.+?)\s+makes my throat swell\b",
        r"(?:i|we) get sick from\s+(.+)",
        r"(?:^|\b(?:and|also|plus)\b)\s*(.+?)\s+free(?:\s+(?:food|options?|meals?))?(?:\s+please)?$",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, scope):
            if "allergic to" in pattern:
                prefix = scope[max(0, match.start() - 16):match.start()]
                if re.search(r"(?:not|no longer)\s+$", prefix):
                    continue
            target = match.group(1)
            target = re.split(
                r"\b(?:and i want|and want|and like|and enjoy|but|however|can i|could i|do you|does the|show me|recommend|with|what can i|what do you|which dishes|which items)\b",
                target,
                maxsplit=1,
            )[0]
            targets.append(target)
    for match in re.finditer(r"(?:^|\band\b)\s*(.+?)\s+(?:allergy|allergies|intolerance|intolerances)\b", scope):
        targets.append(match.group(1))
    if "celiac" in scope or "coeliac" in scope:
        targets.append("celiac")
    return " and ".join(targets)


def _known_alias_terms() -> list[str]:
    groups = (
        VERIFIED_DIETARY_ALIASES,
        UNVERIFIED_DIETARY_ALIASES,
        TRACKED_ALLERGEN_ALIASES,
        UNTRACKED_ALLERGEN_ALIASES,
    )
    terms = {normalize(alias) for group in groups for aliases in group.values() for alias in aliases}
    terms.update(term[3:] for term in list(terms) if term.startswith("no "))
    return sorted(terms, key=len, reverse=True)


def _unknown_restriction_names(targets: str) -> list[str]:
    if not targets:
        return []

    stopwords = {
        "i", "im", "am", "we", "our", "my", "me", "a", "an", "the", "to", "of", "have",
        "has", "cannot", "do", "not", "eat", "avoid", "avoiding", "without", "allergic",
        "allergy", "allergies", "intolerant", "intolerance", "food", "foods", "ingredient",
        "ingredients", "severe", "very", "really", "please", "also", "known", "makes", "sick", "ill",
        "mean", "meant", "actually", "longer", "can", "fine", "okay", "ok", "resolved", "applies",
        "now", "again", "anymore", "remove", "clear", "need", "for", "free", "necessary", "required",
        "it", "s", "its", "correction", "instead", "except", "add", "like", "too",
        "plus", "as", "well", "another", "allergen", "allergens", "list", "products",
        "gives", "causes", "hives", "anaphylaxis",
        "reaction", "reactions", "badly", "get", "break", "out", "throat", "swell",
        "sensitive", "sensitivity", "tolerate", "react", "reaction", "intolerant", "intolerance",
        "no", "none", "any", "and", "or", "is", "are", "from", "restriction", "restrictions", "disease",
    }
    names: list[str] = []
    for target in re.split(r"\b(?:and|or|but|however|not)\b", targets):
        cleaned = f" {target} "
        for term in _known_alias_terms():
            cleaned = re.sub(rf"\b{re.escape(term)}\b", " ", cleaned)
        tokens = [token for token in normalize(cleaned).split() if token not in stopwords]
        name = " ".join(tokens)
        if name and name not in names:
            names.append(name)
    return names


def _unknown_removal_targets(text: str) -> list[str]:
    normalized = normalize(text)
    patterns = (
        r"^(?:i am |we are )?(?:not|no longer) allergic to\s+(.+)$",
        r"^(.+?)\s+(?:is|are) (?:fine|okay|ok)(?: now)?$",
        r"^(?:my )?(.+?)\s+(?:allergy|restriction) (?:is )?(?:resolved|no longer applies)$",
        r"^(?:i|we) can (?:now )?(?:eat|have)\s+(.+?)(?: now| again)?$",
        r"^(?:i|we) meant (.+?) not (.+)$",
        r"^(?:i|we) no longer have (?:a |an )?(.+?) allergy$",
        r"^(?:my )?(.+?) intolerance (?:is )?resolved$",
        r"^(.+?) no longer makes me sick$",
        r"^(?:i am|we are) no longer sensitive to (.+)$",
        r"^(.+?) no longer applies$",
        r"^clear (.+)$",
    )
    clauses = [clause.strip() for clause in re.split(r"\b(?:but|and)\b", normalized) if clause.strip()]
    return [
        normalize(match.group(2) if len(match.groups()) > 1 and match.group(2) else match.group(1))
        for clause in clauses
        for pattern in patterns
        if (match := re.search(pattern, clause))
    ]


def _contains_positive_free(text: str, alias: str) -> bool:
    normalized = normalize(text)
    term = normalize(alias)
    for match in re.finditer(rf"\b{re.escape(term)}\s+free\b", normalized):
        prefix = normalized[max(0, match.start() - 12):match.start()].strip()
        if not prefix.endswith(("not", "is not", "are not")):
            return True
    return False


def _diet_label_is_positive(text: str, aliases: tuple[str, ...]) -> bool:
    normalized = normalize(text)
    if _is_removal(text, aliases):
        return False
    for alias in aliases:
        term = normalize(alias)
        if re.search(
            rf"\b(?:avoid|avoiding|cannot eat|cannot have|do not eat|do not have|without)\s+(?:a |an )?{re.escape(term)}\b",
            normalized,
        ):
            # Phrases such as "do not eat meat" are canonical positive vegetarian aliases.
            if term not in {"no meat", "do not eat meat"}:
                return False
        if re.search(rf"\b(?:not|non)\s+{re.escape(term)}\b", normalized):
            return False
    return True


def _is_free_constraint_request(text: str) -> bool:
    normalized = normalize(text)
    if " free" not in f" {normalized}" or "?" in text:
        return False
    if any(
        phrase in normalized
        for phrase in (
            "do not need", "does not need", "not necessary", "is not necessary",
            "no need for", "do not eat", "cannot eat", "cannot have", "avoid ",
            "i like", "we like", "i can eat", "i can have", "we can eat", "we can have",
        )
    ):
        return False
    if "please" in set(normalized.split()) or is_personal_diet_statement(text):
        return True
    if re.search(r"^(?:i|we) (?:need|want|require|must have|only eat)\b", normalized):
        return True
    if normalized.startswith(("the ", "this ", "that ", "it ")) or " is " in f" {normalized} ":
        return False
    return normalized.endswith(" free") or bool(
        re.search(r"\bfree (?:food|options?|meals?)$", normalized)
    )


def _is_negated_free_mention(text: str, alias: str) -> bool:
    normalized = normalize(text)
    term = normalize(alias)
    return bool(re.search(
        rf"\b(?:cannot eat|cannot have|do not eat|avoid|avoiding)\s+{re.escape(term)}\s+free\b",
        normalized,
    ))


def _contains_restriction_alias(targets: str, label: str, alias: str) -> bool:
    normalized = normalize(targets)
    term = normalize(alias)
    matches = list(re.finditer(rf"\b{re.escape(term)}\b", normalized))
    if not matches:
        return False
    if label != "dairy" or term not in {"milk", "cream", "creme fraiche"}:
        return True
    plant_modifiers = {"coconut", "oat", "almond", "soy", "rice", "cashew", "hemp"}
    for match in matches:
        prefix_words = normalized[:match.start()].split()
        if not prefix_words or prefix_words[-1] not in plant_modifiers:
            return True
    return False


def _allergen_restriction_labels(
    text: str,
    *,
    allergen_context: bool,
) -> tuple[list[str], list[str], list[str]]:
    normalized = normalize(text)
    scope = _restriction_scope(text)
    explicit_allergen_removal = any(
        _is_removal(text, aliases, allow_tolerance=True)
        for group in (TRACKED_ALLERGEN_ALIASES, UNTRACKED_ALLERGEN_ALIASES)
        for aliases in group.values()
    )
    named_allergen_mention = any(
        contains_term(text, alias)
        for group in (TRACKED_ALLERGEN_ALIASES, UNTRACKED_ALLERGEN_ALIASES)
        for aliases in group.values()
        for alias in aliases
    )
    starts_with_allergen = any(
        normalized == normalize(alias) or normalized.startswith(f"{normalize(alias)} ")
        for group in (TRACKED_ALLERGEN_ALIASES, UNTRACKED_ALLERGEN_ALIASES)
        for aliases in group.values()
        for alias in aliases
    )
    short_contextual_addition = (
        allergen_context
        and not _is_query_operation(text)
        and 1 <= len(normalized.split()) <= 4
        and normalized not in {"none", "nothing", "no", "no thanks"}
        and (
            len(normalized.split()) == 1
            or normalized.startswith(("plus ", "another "))
            or normalized.endswith((" too", " as well"))
        )
    )
    correction = any(
        cue in normalized for cue in ("i meant", "also ", "add ", "allerg")
    ) or normalized.startswith(("and ", "plus ", "actually ", "correction ")) or normalized.endswith(" too") or short_contextual_addition or (
        allergen_context and (explicit_allergen_removal or (named_allergen_mention and starts_with_allergen))
    )
    if (
        _is_query_operation(text)
        and not normalized.startswith(("can i eat ", "can i have "))
        and not _has_personal_constraint_clause(text)
    ):
        return [], [], []

    strong_cue = any(
        cue in f" {scope} "
        for cue in (
            " allergic ", " allergy ", " allergies ", " allergen ", " allergens ", " intolerant ", " intolerance ",
            " sensitivity ", " sensitive ", " cannot tolerate ", " react to ",
            " cannot eat ", " cannot have ", " do not eat ", " do not have ", " do not drink ",
            " avoid ", " avoiding ", " without ", " makes me sick ", " makes me ill ",
            " gives me hives ", " causes anaphylaxis ",
            " reaction to ", " react badly to ", " causes a reaction ", " causes hives ",
            " hives from ", " anaphylaxis from ", " throat swell ", " get sick from ",
            " celiac ", " coeliac ",
        )
    ) or bool(re.search(r"\b(?:no|non)\s+[a-z]", scope))
    contextual = allergen_context and correction
    if not strong_cue and not contextual:
        # "Dairy-free please" is an avoidance constraint. A gluten-free request
        # is represented by the verified dietary flag unless celiac/allergy
        # language independently establishes a medical restriction.
        if not _is_free_constraint_request(text):
            return [], [], []
        free_targets = _restriction_targets(scope)
        tracked = [
            label
            for label, aliases in TRACKED_ALLERGEN_ALIASES.items()
            if label != "gluten"
            and any(
                _contains_positive_free(scope, alias)
                or _contains_restriction_alias(free_targets, label, alias)
                for alias in aliases
            )
            and not _is_removal(text, aliases, allow_tolerance=True)
            and not any(_is_negated_free_mention(text, alias) for alias in aliases)
        ]
        untracked = [
            label
            for label, aliases in UNTRACKED_ALLERGEN_ALIASES.items()
            if any(
                _contains_positive_free(scope, alias)
                or _contains_restriction_alias(free_targets, label, alias)
                for alias in aliases
            )
            and not _is_removal(text, aliases, allow_tolerance=True)
            and not any(_is_negated_free_mention(text, alias) for alias in aliases)
        ]
        if "shellfish" in untracked and "fish" in tracked:
            tracked.remove("fish")
        return tracked, untracked, _unknown_restriction_names(free_targets)

    targets = _restriction_targets(scope)
    if contextual and not strong_cue:
        targets = f"{targets} and {scope}" if targets else scope
    tracked = [
        label
        for label, aliases in TRACKED_ALLERGEN_ALIASES.items()
        if any(_contains_restriction_alias(targets, label, alias) for alias in aliases)
        and not _is_removal(text, aliases, allow_tolerance=True)
        and not any(_is_negated_free_mention(text, alias) for alias in aliases)
    ]
    untracked = [
        label
        for label, aliases in UNTRACKED_ALLERGEN_ALIASES.items()
        if any(_contains_restriction_alias(targets, label, alias) for alias in aliases)
        and not _is_removal(text, aliases, allow_tolerance=True)
        and not any(_is_negated_free_mention(text, alias) for alias in aliases)
    ]
    if "shellfish" in untracked and "fish" in tracked:
        tracked.remove("fish")
    if "tree nuts" in tracked and "unspecified nuts" in untracked:
        untracked.remove("unspecified nuts")
    unknown = _unknown_restriction_names(targets)
    return tracked, untracked, unknown


def _remove_labels(
    text: str,
    values: list[str],
    aliases: dict[str, tuple[str, ...]],
    *,
    allow_tolerance: bool = False,
) -> None:
    for label, terms in aliases.items():
        if _is_removal(text, terms, allow_tolerance=allow_tolerance) and label in values:
            values.remove(label)


def merge_preferences(
    message: str,
    *,
    dietary: Sequence[str] = (),
    allergens: Sequence[str] = (),
    untracked_allergens: Sequence[str] = (),
    unverified_diets: Sequence[str] = (),
    allergen_context: bool = False,
) -> PreferenceState:
    state = PreferenceState(
        dietary=list(dietary),
        allergens=list(allergens),
        untracked_allergens=list(untracked_allergens),
        unverified_diets=list(unverified_diets),
    )
    normalized = normalize(message)
    no_allergy_statements = {
        "no allergies", "no food allergies", "allergies none", "no known allergies",
        "no known food allergies", "i do not have allergies", "i do not have any allergies",
        "i do not have food allergies", "i have no allergies", "i have no food allergies",
        "not allergic to anything", "no allergies anymore",
    }
    contextual_no_allergies = allergen_context and normalized in {"i have none", "none for allergies"}
    no_allergies = normalized in no_allergy_statements or contextual_no_allergies or any(
        phrase in normalized
        for phrase in (
            "i have no allergies", "i have no food allergies", "i do not have any allergies",
            "i do not have food allergies", "not allergic to anything",
        )
    )
    clear_all = normalized in {
        "clear all preferences and allergies", "reset all preferences and allergies",
        "clear all preferences", "reset all preferences",
    }
    if clear_all:
        return PreferenceState()
    if normalized in {
        "clear my allergies", "reset my allergies", "remove all allergies", "forget my allergies",
    }:
        state.allergens.clear()
        state.untracked_allergens.clear()
        return state
    if normalized in {
        "clear my dietary preferences", "reset my dietary preferences", "i have no dietary restrictions",
        "clear my diet preferences", "reset my diet preferences", "clear my dietary restrictions",
        "reset all dietary preferences",
    }:
        state.dietary.clear()
        state.unverified_diets.clear()
        return state
    if no_allergies:
        state.allergens.clear()
        state.untracked_allergens.clear()
        if normalized in no_allergy_statements or contextual_no_allergies:
            return state

    query_operation = _is_query_operation(message)
    questioned_removal = (
        "?" in message
        and not re.search(r"[.;!]\s*[^?]*\?", message)
        and any(
        phrase in normalized
        for phrase in (
            "i can eat ", "i can have ", "we can eat ", "we can have ",
            "i am not allergic", "we are not allergic", "i am no longer allergic",
            "i no longer have ", "we no longer have ",
        )
        )
    )
    if questioned_removal:
        return state
    dietary_removal_operation = any(
        _is_removal(message, aliases)
        for group in (VERIFIED_DIETARY_ALIASES, UNVERIFIED_DIETARY_ALIASES)
        for aliases in group.values()
    )
    allergen_removal_operation = any(
        _is_removal(message, aliases, allow_tolerance=True)
        for group in (TRACKED_ALLERGEN_ALIASES, UNTRACKED_ALLERGEN_ALIASES)
        for aliases in group.values()
    )
    non_halal_removal = bool(re.search(
        r"^(?:actually )?(?:i|we) can (?:eat|have) non halal(?: food)?(?: now| again)?$",
        normalized,
    ))
    removal_operation = dietary_removal_operation or allergen_removal_operation or non_halal_removal or (
        not query_operation and any(
        cue in normalized
        for cue in ("not allergic", "no longer allergic", "can eat", "is fine", "is okay", "is ok", "is resolved")
        )
    )
    personal_removal = removal_operation and normalized.startswith(
        ("i ", "im ", "we ", "my ", "can i eat ", "can i have ")
    )
    if query_operation and not personal_removal and not _has_personal_constraint_clause(message):
        return state

    _remove_labels(message, state.dietary, VERIFIED_DIETARY_ALIASES)
    _remove_labels(
        message, state.allergens, TRACKED_ALLERGEN_ALIASES, allow_tolerance=True
    )
    _remove_labels(
        message, state.untracked_allergens, UNTRACKED_ALLERGEN_ALIASES, allow_tolerance=True
    )
    _remove_labels(message, state.unverified_diets, UNVERIFIED_DIETARY_ALIASES)
    if non_halal_removal and "halal" in state.unverified_diets:
        state.unverified_diets.remove("halal")
    for value in list(state.untracked_allergens):
        if _is_removal(message, (value,), allow_tolerance=True):
            state.untracked_allergens.remove(value)

    if any(
        phrase in normalized
        for phrase in (
            "can eat gluten", "gluten is fine", "gluten is okay", "gluten is ok",
            "no longer celiac", "no longer coeliac",
        )
    ):
        if "gluten-free" in state.dietary:
            state.dietary.remove("gluten-free")
    if re.search(r"^(?:actually )?(?:i|we) (?:can |do )?(?:now )?eat meat(?: now| again)?$", normalized):
        state.dietary = [label for label in state.dietary if label not in {"vegan", "vegetarian"}]
    if re.search(r"^(?:now )?(?:i am|we are) (?:an? )?omnivores?(?: now)?$", normalized):
        state.dietary = [label for label in state.dietary if label not in {"vegan", "vegetarian"}]

    if is_personal_diet_statement(message):
        for label in requested_labels(message, VERIFIED_DIETARY_ALIASES):
            if _diet_label_is_positive(message, VERIFIED_DIETARY_ALIASES[label]):
                state.dietary.append(label)
        for label in requested_labels(message, UNVERIFIED_DIETARY_ALIASES):
            if _diet_label_is_positive(message, UNVERIFIED_DIETARY_ALIASES[label]):
                state.unverified_diets.append(label)

    if _is_free_constraint_request(message):
        free_targets = _restriction_targets(_restriction_scope(message))
        if any(
            _contains_restriction_alias(free_targets, "gluten", alias)
            for alias in TRACKED_ALLERGEN_ALIASES["gluten"]
        ):
            state.dietary.append("gluten-free")

    if re.search(r"\b(?:i|we)\s+(?:cannot|do not)\s+(?:eat|have)\s+pork\b|\b(?:i|we)\s+avoid\s+pork\b", normalized):
        state.unverified_diets.append("pork-free")
    if re.search(r"\b(?:i|we)\s+(?:cannot|do not)\s+(?:drink|eat|have)\s+alcohol\b|\b(?:i|we)\s+avoid\s+alcohol\b", normalized):
        state.unverified_diets.append("alcohol-free")
    if re.search(r"\b(?:i|we)\s+(?:cannot|do not)\s+eat\s+meat\b|\b(?:i|we)\s+avoid\s+meat\b", normalized):
        state.dietary.append("vegetarian")

    requested_dietary = requested_labels(message, VERIFIED_DIETARY_ALIASES)
    if "vegan" in requested_dietary and _diet_label_is_positive(
        message, VERIFIED_DIETARY_ALIASES["vegan"]
    ):
        if "vegetarian" in state.dietary:
            state.dietary.remove("vegetarian")
    if (
        "vegetarian" in requested_dietary
        and any(
            term in normalized
            for term in (
                "switch", "instead of", "changed to", "now vegetarian", "vegetarian now",
                "now i am vegetarian", "vegetarian instead",
            )
        )
        and _diet_label_is_positive(message, VERIFIED_DIETARY_ALIASES["vegetarian"])
        and "vegan" in state.dietary
    ):
        state.dietary.remove("vegan")

    requested_unverified = requested_labels(message, UNVERIFIED_DIETARY_ALIASES)
    if "kosher" in requested_unverified and any(
        phrase in normalized for phrase in ("switch", "changed to", "kosher now")
    ):
        state.unverified_diets = [label for label in state.unverified_diets if label != "halal"]
    if "halal" in requested_unverified and any(
        phrase in normalized for phrase in ("switch", "changed to", "halal now")
    ):
        state.unverified_diets = [label for label in state.unverified_diets if label != "kosher"]

    tracked, untracked, unknown = _allergen_restriction_labels(
        message,
        allergen_context=allergen_context or bool(state.allergens or state.untracked_allergens),
    )
    removed_unknowns = _unknown_removal_targets(message)
    unknown = [
        label
        for label in unknown
        if not any(contains_term(label, removed) or contains_term(removed, label) for removed in removed_unknowns)
    ]
    sentinel = "a specifically named allergen not tracked by the menu"
    if (tracked or untracked or unknown) and sentinel in state.untracked_allergens:
        state.untracked_allergens.remove(sentinel)
    for label in tracked:
        state.allergens.append(label)
    for label in untracked:
        state.untracked_allergens.append(label)
    state.untracked_allergens.extend(unknown)
    if (
        ("allerg" in normalized or "intoler" in normalized or "sensitive" in normalized or "sensitivity" in normalized)
        and not tracked
        and not untracked
        and not unknown
        and "reaction" not in normalized
        and not removal_operation
        and not removed_unknowns
        and not no_allergies
    ):
        state.untracked_allergens.append(sentinel)

    state.dietary = _dedupe(state.dietary)
    if "vegan" in state.dietary and "vegetarian" in state.dietary:
        state.dietary.remove("vegetarian")
    state.allergens = _dedupe(state.allergens)
    state.untracked_allergens = _dedupe(state.untracked_allergens)
    state.unverified_diets = _dedupe(state.unverified_diets)
    return state


def preferences_from_messages(messages: Sequence[AnyMessage]) -> PreferenceState:
    state = PreferenceState()
    for message in messages:
        if isinstance(message, HumanMessage):
            state = merge_preferences(
                str(message.content),
                dietary=state.dietary,
                allergens=state.allergens,
                untracked_allergens=state.untracked_allergens,
                unverified_diets=state.unverified_diets,
            )
    return state
