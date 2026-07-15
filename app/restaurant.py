from dataclasses import dataclass

MENU_CATEGORIES = ("starter", "main", "dessert")

# Recipe-level labels that Maple & Ember's menu data explicitly verifies. These
# labels describe the listed recipe, not shared-kitchen conditions or a formal
# religious/nutrition certification.
VERIFIED_DIETARY_LABELS = (
    "vegan",
    "vegetarian",
    "gluten-free",
    "pescatarian",
    "plant-based",
    "pork-free",
    "alcohol-free",
)

# Allergens and intolerances declared per recipe. Absence from an item's tuple
# never means allergen-free; the shared-kitchen warning still applies.
TRACKED_ALLERGENS = (
    "dairy",
    "egg",
    "fish",
    "gluten",
    "mustard",
    "peanuts",
    "sesame",
    "shellfish",
    "soy",
    "sulfites",
    "tree nuts",
    "wheat",
)


@dataclass(frozen=True, slots=True)
class MenuItem:
    name: str
    category: str
    description: str
    price: float
    dietary_labels: tuple[str, ...] = ()
    allergens: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.category not in MENU_CATEGORIES:
            raise ValueError(f"Unknown menu category: {self.category}")
        if self.price <= 0:
            raise ValueError("Menu item prices must be positive")
        if len(set(self.dietary_labels)) != len(self.dietary_labels):
            raise ValueError(f"Duplicate dietary label on {self.name}")
        if len(set(self.allergens)) != len(self.allergens):
            raise ValueError(f"Duplicate allergen on {self.name}")
        unknown_labels = set(self.dietary_labels) - set(VERIFIED_DIETARY_LABELS)
        if unknown_labels:
            raise ValueError(f"Unknown dietary label(s) on {self.name}: {sorted(unknown_labels)}")
        unknown_allergens = set(self.allergens) - set(TRACKED_ALLERGENS)
        if unknown_allergens:
            raise ValueError(f"Unknown allergen(s) on {self.name}: {sorted(unknown_allergens)}")
        if "vegan" in self.dietary_labels and not {
            "vegetarian",
            "plant-based",
        }.issubset(self.dietary_labels):
            raise ValueError(f"Vegan item {self.name} must also carry vegetarian and plant-based labels")

    @property
    def vegetarian(self) -> bool:
        return "vegetarian" in self.dietary_labels

    @property
    def vegan(self) -> bool:
        return "vegan" in self.dietary_labels

    @property
    def gluten_free(self) -> bool:
        return "gluten-free" in self.dietary_labels

    def supports(self, dietary_label: str) -> bool:
        return dietary_label in self.dietary_labels


RESTAURANT = {
    "name": "Maple & Ember",
    "description": "A neighbourhood Canadian bistro focused on seasonal, locally sourced food.",
    "cuisine": "contemporary Canadian bistro cooking",
    "vibe": "warm, relaxed, and polished, with a neighbourhood feel and an ember-toned dining room.",
    "service": "Dinner service runs from Tuesday through Sunday",
    "address": "123 King Street West, Toronto, ON",
    "phone": "+1 (416) 555-0142",
    "currency": "CAD",
    "hours": {
        "Monday": "Closed",
        "Tuesday–Thursday": "5:00 PM–10:00 PM",
        "Friday–Saturday": "5:00 PM–11:00 PM",
        "Sunday": "5:00 PM–9:00 PM",
    },
    "reservation_url": "https://example.com/maple-and-ember/reservations",
}

MENU: tuple[MenuItem, ...] = (
    MenuItem(
        name="Roasted Beet Salad",
        category="starter",
        description="goat cheese, arugula, toasted walnuts, maple vinaigrette",
        price=15,
        dietary_labels=("vegetarian", "gluten-free", "pescatarian", "pork-free", "alcohol-free"),
        allergens=("dairy", "tree nuts", "mustard"),
    ),
    MenuItem(
        name="Crispy Lake Erie Perch",
        category="starter",
        description="lemon, caper aioli, shaved fennel",
        price=18,
        dietary_labels=("gluten-free", "pescatarian", "pork-free", "alcohol-free"),
        allergens=("fish", "egg", "mustard"),
    ),
    MenuItem(
        name="Maple-Roasted Squash Soup",
        category="starter",
        description="roasted squash, apple, pumpkin seeds, sage oil",
        price=14,
        dietary_labels=(
            "vegan",
            "vegetarian",
            "gluten-free",
            "pescatarian",
            "plant-based",
            "pork-free",
            "alcohol-free",
        ),
    ),
    MenuItem(
        name="Charred Cauliflower Steak",
        category="main",
        description="white bean purée, salsa verde, pickled shallots",
        price=26,
        dietary_labels=(
            "vegan",
            "vegetarian",
            "gluten-free",
            "pescatarian",
            "plant-based",
            "pork-free",
            "alcohol-free",
        ),
    ),
    MenuItem(
        name="Wild Mushroom Risotto",
        category="main",
        description="Ontario mushrooms, parmesan, herbs",
        price=29,
        dietary_labels=("vegetarian", "gluten-free", "pescatarian", "pork-free", "alcohol-free"),
        allergens=("dairy",),
    ),
    MenuItem(
        name="Maple-Glazed Salmon",
        category="main",
        description="wild rice, seasonal greens, cider reduction",
        price=34,
        dietary_labels=("gluten-free", "pescatarian", "pork-free"),
        allergens=("fish", "sulfites"),
    ),
    MenuItem(
        name="Cedar-Roasted Chicken",
        category="main",
        description="roasted root vegetables, wilted greens, herb jus",
        price=32,
        dietary_labels=("gluten-free", "pork-free", "alcohol-free"),
    ),
    MenuItem(
        name="Ember Burger",
        category="main",
        description="Ontario beef, aged cheddar, onion jam, fries",
        price=25,
        dietary_labels=("pork-free", "alcohol-free"),
        allergens=("gluten", "dairy", "egg", "wheat"),
    ),
    MenuItem(
        name="Lentil & Root Vegetable Pie",
        category="main",
        description="green lentils, roasted roots, rosemary gravy, flaky pastry",
        price=28,
        dietary_labels=(
            "vegan",
            "vegetarian",
            "pescatarian",
            "plant-based",
            "pork-free",
            "alcohol-free",
        ),
        allergens=("gluten", "wheat"),
    ),
    MenuItem(
        name="Cider-Poached Pear",
        category="dessert",
        description="oat crumble, coconut cream",
        price=12,
        dietary_labels=("vegan", "vegetarian", "pescatarian", "plant-based", "pork-free"),
        allergens=("gluten", "sulfites", "wheat"),
    ),
    MenuItem(
        name="Dark Chocolate Torte",
        category="dessert",
        description="sea salt, crème fraîche",
        price=13,
        dietary_labels=("vegetarian", "gluten-free", "pescatarian", "pork-free", "alcohol-free"),
        allergens=("dairy", "egg"),
    ),
    MenuItem(
        name="Maple Berry Pavlova",
        category="dessert",
        description="Ontario berries, maple cream, crisp meringue",
        price=14,
        dietary_labels=("vegetarian", "gluten-free", "pescatarian", "pork-free", "alcohol-free"),
        allergens=("dairy", "egg"),
    ),
)


def format_item(item: MenuItem) -> str:
    dietary: list[str] = []
    if item.vegan:
        dietary.append("vegan")
    elif item.vegetarian:
        dietary.append("vegetarian")
    elif item.supports("pescatarian"):
        dietary.append("pescatarian")
    if item.gluten_free:
        dietary.append("gluten-free")
    dietary.extend(label for label in ("pork-free", "alcohol-free") if item.supports(label))
    label = f" ({', '.join(dietary)})" if dietary else ""
    return f"{item.name} — ${item.price:.0f}{label}: {item.description}"


# Colloquial aliases used to match a menu item or category from free text.
# Keys are the exact ``MENU`` item names / category values.
ITEM_ALIASES: dict[str, tuple[str, ...]] = {
    "Maple-Roasted Squash Soup": (
        "maple roasted squash soups",
        "maple roasted squash soup",
        "squash soups",
        "squash soup",
    ),
    "Roasted Beet Salad": ("roasted beet salad", "beet salads", "beet salad", "beets", "beet"),
    "Crispy Lake Erie Perch": ("crispy lake erie perch", "lake erie perch", "perch"),
    "Charred Cauliflower Steak": ("charred cauliflower steak", "cauliflower steak", "cauliflower", "steak"),
    "Wild Mushroom Risotto": (
        "wild mushroom risotto",
        "mushroom risotto",
        "mushroom dish",
        "mushroom",
        "mushrooms",
        "risotto",
    ),
    "Maple-Glazed Salmon": ("maple glazed salmon", "salmon"),
    "Cedar-Roasted Chicken": (
        "cedar roasted chickens",
        "cedar roasted chicken",
        "roasted chickens",
        "roasted chicken",
        "chicken",
    ),
    "Ember Burger": ("ember burger", "burger", "burgers"),
    "Lentil & Root Vegetable Pie": (
        "lentil and root vegetable pies",
        "lentil and root vegetable pie",
        "lentil vegetable pies",
        "lentil vegetable pie",
        "vegetable pies",
        "vegetable pie",
        "lentil pies",
        "lentil pie",
    ),
    "Cider-Poached Pear": (
        "cider poached pears",
        "cider poached pear",
        "poached pears",
        "poached pear",
        "pears",
        "pear",
    ),
    "Dark Chocolate Torte": (
        "dark chocolate tortes",
        "dark chocolate torte",
        "chocolate tortes",
        "chocolate torte",
        "chocolate",
        "tortes",
        "torte",
    ),
    "Maple Berry Pavlova": (
        "maple berry pavlovas",
        "maple berry pavlova",
        "berry pavlovas",
        "berry pavlova",
        "pavlovas",
        "pavlova",
    ),
}

CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "starter": ("starter", "starters", "appetizer", "appetizers", "appetiser", "appetisers"),
    "main": ("main", "mains", "entree", "entrees", "entrée", "entrées"),
    "dessert": ("dessert", "desserts", "dessets", "sweet", "sweets"),
}
