from app.knowledge import load_public
from app.preferences import TRACKED_ALLERGEN_ALIASES, VERIFIED_DIETARY_ALIASES
from app.restaurant import (
    ITEM_ALIASES,
    MENU,
    MENU_CATEGORIES,
    RESTAURANT,
    TRACKED_ALLERGENS,
    VERIFIED_DIETARY_LABELS,
)


def test_authoritative_menu_has_complete_unique_metadata():
    names = [item.name for item in MENU]
    assert len(MENU) == 12
    assert len(names) == len(set(names))
    assert set(ITEM_ALIASES) == set(names)
    assert {item.category for item in MENU} == set(MENU_CATEGORIES)
    assert set(VERIFIED_DIETARY_ALIASES) == set(VERIFIED_DIETARY_LABELS)
    assert set(TRACKED_ALLERGEN_ALIASES) == set(TRACKED_ALLERGENS)

    for label in VERIFIED_DIETARY_LABELS:
        assert any(item.supports(label) for item in MENU), label
    for item in MENU:
        assert set(item.dietary_labels) <= set(VERIFIED_DIETARY_LABELS)
        assert set(item.allergens) <= set(TRACKED_ALLERGENS)


def test_generated_guest_menu_and_pricing_cover_authoritative_items():
    menu_doc = load_public("menu.md")
    pricing_doc = load_public("pricing.md")
    menu_lines = menu_doc.splitlines()
    pricing_lines = pricing_doc.splitlines()
    for item in MENU:
        menu_line = next(line for line in menu_lines if line.startswith(f"- {item.name} —"))
        pricing_line = next(line for line in pricing_lines if line.startswith(f"- {item.name} ("))
        assert f"${item.price:.0f}" in menu_line
        assert f"${item.price:.0f}" in pricing_line
        for label in item.dietary_labels:
            assert label in menu_line
        for allergen in item.allergens:
            assert allergen in menu_line
        if not item.allergens:
            assert "No allergens declared in the menu data" in menu_line

    assert RESTAURANT["currency"] in pricing_doc


def test_guest_faq_covers_the_brand_attributes():
    faq = load_public("faq.md")
    normalized_faq = " ".join(faq.split())
    assert RESTAURANT["address"] in faq
    assert RESTAURANT["phone"] in faq
    assert RESTAURANT["cuisine"] in normalized_faq
    assert RESTAURANT["vibe"] in normalized_faq


def test_guest_dietary_guide_names_every_supported_label_and_allergen():
    guide = load_public("dietary_restrictions.md")
    for label in VERIFIED_DIETARY_LABELS:
        assert label in guide
    for allergen in TRACKED_ALLERGENS:
        assert allergen in guide
