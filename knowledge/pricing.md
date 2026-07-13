# Maple & Ember Menu and Pricing

## Source and Synchronization

- Treat `MENU` in `app/restaurant.py` as the single source of truth for menu names, categories, descriptions, prices, dietary flags, and declared allergens.
- Regenerate this file whenever `MENU` changes.
- Use the current `MENU` value if this file ever conflicts with `MENU`.
- Interpret every price below as the listed base price for the item as described.
- Preserve the `$` price symbol used by the source. The source does not declare a currency code, so do not label these amounts CAD, USD, or another currency unless `app/restaurant.py` is updated to specify one.

## Dietary and Allergen Key

| Field value | Meaning |
|---|---|
| Yes | The corresponding Boolean flag is explicitly `True` in `MENU`. |
| No | The corresponding Boolean flag is explicitly `False` in `MENU`. |
| None listed | The allergen tuple is empty; this does not mean allergen-free or free from cross-contact. |

Consult `policies.md` before answering any allergy or substitution question.

## Starters

| Item | Description | Listed price | Vegetarian | Vegan | Gluten-free | Declared allergens |
|---|---|---:|---|---|---|---|
| Roasted Beet Salad | goat cheese, arugula, toasted walnuts, maple vinaigrette | $15 | Yes | No | Yes | dairy; tree nuts |
| Crispy Lake Erie Perch | lemon, caper aioli, shaved fennel | $18 | No | No | Yes | fish; egg |

## Mains

| Item | Description | Listed price | Vegetarian | Vegan | Gluten-free | Declared allergens |
|---|---|---:|---|---|---|---|
| Charred Cauliflower Steak | white bean purée, salsa verde, pickled shallots | $26 | Yes | Yes | Yes | None listed |
| Wild Mushroom Risotto | Ontario mushrooms, parmesan, herbs | $29 | Yes | No | Yes | dairy |
| Maple-Glazed Salmon | wild rice, seasonal greens, cider reduction | $34 | No | No | Yes | fish |
| Ember Burger | Ontario beef, aged cheddar, onion jam, fries | $25 | No | No | No | gluten; dairy; egg |

## Desserts

| Item | Description | Listed price | Vegetarian | Vegan | Gluten-free | Declared allergens |
|---|---|---:|---|---|---|---|
| Cider-Poached Pear | oat crumble, coconut cream | $12 | Yes | Yes | No | gluten |
| Dark Chocolate Torte | sea salt, crème fraîche | $13 | Yes | No | Yes | dairy; egg |

## Price-Answer Rules

- State the exact listed amount with the `$` symbol; for example: "The Ember Burger is listed at $25."
- Do not state a currency code because the authoritative source does not provide one.
- Quote only the price recorded in this file or the current `MENU`.
- Do not invent modification charges, discounts, taxes, fees, specials, or promotional prices.
- If asked whether a modification changes the price, say: "The listed price is $X. Modification pricing is not provided, so please confirm with Maple & Ember using the contact information in `faq.md`."
- If asked which currency applies, say: "The current restaurant data uses the $ symbol but does not specify a currency code. Please confirm directly with Maple & Ember."
- If asked for an item absent from `MENU`, say that it is not listed on the current menu; do not claim that the restaurant never offers it.
- When calculating multiple-item costs, use exact quantity arithmetic from the listed prices and label the result a "menu-price subtotal."
- Do not add unspecified taxes, gratuities, service charges, delivery charges, modification charges, or other fees.
