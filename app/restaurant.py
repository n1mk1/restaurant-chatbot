from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MenuItem:
    name: str
    category: str
    description: str
    price: float
    vegetarian: bool = False
    vegan: bool = False
    gluten_free: bool = False
    allergens: tuple[str, ...] = ()

RESTAURANT = {
    "name": "Maple & Ember",
    "description": "A neighbourhood Canadian bistro focused on seasonal, locally sourced food.",
    "address": "123 King Street West, Toronto, ON",
    "phone": "+1 (416) 555-0142",
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
        "Roasted Beet Salad",
        "starter",
        "goat cheese, arugula, toasted walnuts, maple vinaigrette",
        15,
        vegetarian=True,
        gluten_free=True,
        allergens=("dairy", "tree nuts"),
    ),
    MenuItem(
        "Crispy Lake Erie Perch",
        "starter",
        "lemon, caper aioli, shaved fennel",
        18,
        gluten_free=True,
        allergens=("fish", "egg"),
    ),
    MenuItem(
        "Charred Cauliflower Steak",
        "main",
        "white bean purée, salsa verde, pickled shallots",
        26,
        vegetarian=True,
        vegan=True,
        gluten_free=True,
    ),
    MenuItem(
        "Wild Mushroom Risotto",
        "main",
        "Ontario mushrooms, parmesan, herbs",
        29,
        vegetarian=True,
        gluten_free=True,
        allergens=("dairy",),
    ),
    MenuItem(
        "Maple-Glazed Salmon",
        "main",
        "wild rice, seasonal greens, cider reduction",
        34,
        gluten_free=True,
        allergens=("fish",),
    ),
    MenuItem(
        "Ember Burger",
        "main",
        "Ontario beef, aged cheddar, onion jam, fries",
        25,
        allergens=("gluten", "dairy", "egg"),
    ),
    MenuItem(
        "Cider-Poached Pear",
        "dessert",
        "oat crumble, coconut cream",
        12,
        vegetarian=True,
        vegan=True,
        allergens=("gluten",),
    ),
    MenuItem(
        "Dark Chocolate Torte",
        "dessert",
        "sea salt, crème fraîche",
        13,
        vegetarian=True,
        gluten_free=True,
        allergens=("dairy", "egg"),
    ),
)


def format_item(item: MenuItem) -> str:
    dietary: list[str] = []
    if item.vegan:
        dietary.append("vegan")
    elif item.vegetarian:
        dietary.append("vegetarian")
    if item.gluten_free:
        dietary.append("gluten-free")
    label = f" ({', '.join(dietary)})" if dietary else ""
    return f"{item.name} — ${item.price:.0f}{label}: {item.description}"


# Colloquial aliases used to match a menu item or category from free text.
# Keys are the exact ``MENU`` item names / category values.
ITEM_ALIASES: dict[str, tuple[str, ...]] = {
    "Roasted Beet Salad": ("roasted beet salad", "beet salads", "beet salad", "beets", "beet"),
    "Crispy Lake Erie Perch": ("crispy lake erie perch", "lake erie perch", "perch"),
    "Charred Cauliflower Steak": ("charred cauliflower steak", "cauliflower steak", "cauliflower", "steak"),
    "Wild Mushroom Risotto": ("wild mushroom risotto", "mushroom risotto", "mushroom dish", "mushroom", "mushrooms", "risotto"),
    "Maple-Glazed Salmon": ("maple glazed salmon", "salmon"),
    "Ember Burger": ("ember burger", "burger", "burgers"),
    "Cider-Poached Pear": ("cider poached pears", "cider poached pear", "poached pears", "poached pear", "pears", "pear"),
    "Dark Chocolate Torte": ("dark chocolate tortes", "dark chocolate torte", "chocolate tortes", "chocolate torte", "chocolate", "tortes", "torte"),
}

CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "starter": ("starter", "starters", "appetizer", "appetizers", "appetiser", "appetisers"),
    "main": ("main", "mains", "entree", "entrees", "entrée", "entrées"),
    "dessert": ("dessert", "desserts", "dessets", "sweet", "sweets"),
}
