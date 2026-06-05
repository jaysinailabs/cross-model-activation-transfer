"""
Multi-hop reasoning task data generator.

Generates entity-relation chains where the answer requires following N
relational hops through a small knowledge graph.  All generation is
deterministic given a seed; no network access is required.

Task design (experiment guide §4.4, Task 1):
    Model A receives the context (hop chain).
    Model B receives the relay + question, and must answer the question.
    The task exposes whether the relay preserves relational structure.

Data format per sample:
    {
        "context":     "<factual chain of N sentences>",
        "question":    "<question requiring N-hop traversal>",
        "answer":      "<single short string>",
        "hops":        <int, 2|3|4>,
        "relay_point": "Model A processes context; Model B receives relay and answers question."
    }
"""

from __future__ import annotations

import random
from typing import Any

# ---------------------------------------------------------------------------
# Knowledge graph
# ---------------------------------------------------------------------------

# (person_name, home_city)
PEOPLE: list[tuple[str, str]] = [
    ("Alice", "Paris"), ("Bob", "Tokyo"), ("Carlos", "Mexico City"),
    ("Fatima", "Cairo"), ("Ivan", "Moscow"), ("Maria", "Rome"),
    ("James", "Toronto"), ("Yuki", "Osaka"), ("Priya", "Mumbai"),
    ("Emma", "Berlin"), ("Ahmed", "Lagos"), ("Sofia", "Buenos Aires"),
    ("Chen Wei", "Beijing"), ("Elena", "Vienna"), ("David", "Sydney"),
    ("Aisha", "Nairobi"), ("Pablo", "Lima"), ("Nina", "Seoul"),
    ("Raj", "Jakarta"), ("Layla", "Istanbul"),
    ("Thomas", "Montreal"), ("Nadia", "Tehran"), ("Felix", "Madrid"),
    ("Amara", "Accra"), ("Hiroshi", "Kyoto"), ("Sara", "Karachi"),
    ("Omar", "Riyadh"), ("Zara", "London"), ("Lucas", "Brasilia"),
    ("Mei", "Shanghai"),
]

# city -> (country, is_capital)
CITIES: dict[str, tuple[str, bool]] = {
    "Paris":       ("France",       True),
    "Tokyo":       ("Japan",        True),
    "Mexico City": ("Mexico",       True),
    "Cairo":       ("Egypt",        True),
    "Moscow":      ("Russia",       True),
    "Rome":        ("Italy",        True),
    "Toronto":     ("Canada",       False),
    "Osaka":       ("Japan",        False),
    "Mumbai":      ("India",        False),
    "Berlin":      ("Germany",      True),
    "Lagos":       ("Nigeria",      False),
    "Buenos Aires": ("Argentina",    True),
    "Beijing":     ("China",        True),
    "Vienna":      ("Austria",      True),
    "Sydney":      ("Australia",    False),
    "Nairobi":     ("Kenya",        True),
    "Lima":        ("Peru",         True),
    "Seoul":       ("South Korea",  True),
    "Jakarta":     ("Indonesia",    True),
    "Istanbul":    ("Turkey",       False),
    "Montreal":    ("Canada",       False),
    "Tehran":      ("Iran",         True),
    "Madrid":      ("Spain",        True),
    "Accra":       ("Ghana",        True),
    "Kyoto":       ("Japan",        False),
    "Karachi":     ("Pakistan",     False),
    "Riyadh":      ("Saudi Arabia", True),
    "London":      ("United Kingdom", True),
    "Brasilia":    ("Brazil",       True),
    "Shanghai":    ("China",        False),
}

# country -> (continent, official_language, capital)
COUNTRIES: dict[str, tuple[str, str, str]] = {
    "France":         ("Europe",        "French",    "Paris"),
    "Japan":          ("Asia",          "Japanese",  "Tokyo"),
    "Mexico":         ("North America", "Spanish",   "Mexico City"),
    "Egypt":          ("Africa",        "Arabic",    "Cairo"),
    "Russia":         ("Europe",        "Russian",   "Moscow"),
    "Italy":          ("Europe",        "Italian",   "Rome"),
    "Canada":         ("North America", "English",   "Ottawa"),
    "India":          ("Asia",          "Hindi",     "New Delhi"),
    "Germany":        ("Europe",        "German",    "Berlin"),
    "Nigeria":        ("Africa",        "English",   "Abuja"),
    "Argentina":      ("South America", "Spanish",   "Buenos Aires"),
    "China":          ("Asia",          "Mandarin",  "Beijing"),
    "Austria":        ("Europe",        "German",    "Vienna"),
    "Australia":      ("Oceania",       "English",   "Canberra"),
    "Kenya":          ("Africa",        "Swahili",   "Nairobi"),
    "Peru":           ("South America", "Spanish",   "Lima"),
    "South Korea":    ("Asia",          "Korean",    "Seoul"),
    "Indonesia":      ("Asia",          "Indonesian", "Jakarta"),
    "Turkey":         ("Europe",        "Turkish",   "Ankara"),
    "Iran":           ("Asia",          "Persian",   "Tehran"),
    "Spain":          ("Europe",        "Spanish",   "Madrid"),
    "Ghana":          ("Africa",        "English",   "Accra"),
    "Pakistan":       ("Asia",          "Urdu",      "Islamabad"),
    "Saudi Arabia":   ("Asia",          "Arabic",    "Riyadh"),
    "United Kingdom": ("Europe",        "English",   "London"),
    "Brazil":         ("South America", "Portuguese", "Brasilia"),
}

# continent -> approximate country count (used in 4-hop)
CONTINENT_FACTS: dict[str, int] = {
    "Europe":        44,
    "Asia":          48,
    "Africa":        54,
    "North America": 23,
    "South America": 12,
    "Oceania":       14,
}

RELAY_POINT = (
    "Model A processes context; Model B receives relay and answers question."
)

# ---------------------------------------------------------------------------
# Phrasing templates
# ---------------------------------------------------------------------------

# 2-hop: person -> city -> country
_2HOP_TEMPLATES = [
    (
        "{name} lives in {city}. {city} is located in {country}.",
        "What country does {name} live in?",
        "{country}",
    ),
    (
        "{name} was born in {city}. {city} is a city in {country}.",
        "In which country was {name} born?",
        "{country}",
    ),
    (
        "{name} works in {city}. {city} is part of {country}.",
        "Which country does {name} work in?",
        "{country}",
    ),
]

# 3-hop: person -> city -> country -> continent
# Template mix: ~half ask about continent (hop 3, tail),
# ~half ask about country (hop 2, mid-chain).
# The mid-chain variants require the relay to preserve the
# person→country link even though the context ends on the continent.
_3HOP_TEMPLATES = [
    # --- ask about continent (hop 3, tail of context) ---
    (
        "{name} lives in {city}. {city} is located in {country}. "
        "{country} is a country on the continent of {continent}.",
        "On which continent does {name} live?",
        "{continent}",
    ),
    (
        "{name} was born in {city}. {city} is in {country}. "
        "{country} is part of {continent}.",
        "What continent is {name} from?",
        "{continent}",
    ),
    (
        "{name} studies in {city}. {city} is a city in {country}. "
        "{country} lies on the continent of {continent}.",
        "Which continent does {name} study in?",
        "{continent}",
    ),
    # --- ask about country (hop 2, mid-chain) ---
    # Context still includes the continent (hop 3) as a distractor;
    # relay must preserve the city→country link, not just the tail.
    (
        "{name} lives in {city}. {city} is in {country}. "
        "{country} is part of {continent}.",
        "What country does {name} live in?",
        "{country}",
    ),
    (
        "{name} was born in {city}. {city} is a city in {country}. "
        "{country} is located in {continent}.",
        "Which country was {name} born in?",
        "{country}",
    ),
    (
        "{name} moved to {city}. {city} is part of {country}. "
        "{country} lies on the continent of {continent}.",
        "What country did {name} move to?",
        "{country}",
    ),
]

# 4-hop: person -> city -> country -> continent -> num_countries
# Template mix: ~half ask about num_countries (hop 4, tail),
# ~half ask about continent (hop 3, mid-chain).
_4HOP_TEMPLATES = [
    # --- ask about num_countries (hop 4, tail of context) ---
    (
        "{name} lives in {city}. {city} is in {country}. "
        "{country} is located in {continent}. "
        "{continent} has approximately {num_countries} countries.",
        "Approximately how many countries are on the continent where {name} lives?",
        "{num_countries}",
    ),
    (
        "{name} was born in {city}. {city} is part of {country}. "
        "{country} is a nation in {continent}. "
        "There are about {num_countries} countries in {continent}.",
        "About how many countries share a continent with {name}'s birthplace?",
        "{num_countries}",
    ),
    (
        "{name} works in {city}. {city} is located in {country}. "
        "{country} is on the continent of {continent}. "
        "{continent} consists of roughly {num_countries} nations.",
        "How many nations are on the continent where {name} works?",
        "{num_countries}",
    ),
    # --- ask about continent (hop 3, mid-chain) ---
    # Context still includes the country count (hop 4) as a distractor;
    # relay must preserve the country→continent link.
    (
        "{name} lives in {city}. {city} is in {country}. "
        "{country} is located in {continent}. "
        "{continent} has approximately {num_countries} countries.",
        "On which continent is the country where {name} lives?",
        "{continent}",
    ),
    (
        "{name} was born in {city}. {city} is part of {country}. "
        "{country} is a nation in {continent}. "
        "There are about {num_countries} countries in {continent}.",
        "Which continent is {name}'s birth country part of?",
        "{continent}",
    ),
    (
        "{name} works in {city}. {city} is located in {country}. "
        "{country} is on the continent of {continent}. "
        "{continent} consists of roughly {num_countries} nations.",
        "What continent is the country where {name} works located on?",
        "{continent}",
    ),
]

# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _build_fact(person: str, city: str) -> dict[str, str]:
    """Resolve full chain of facts from person and city."""
    country, _ = CITIES[city]
    continent, language, capital = COUNTRIES[country]
    num_countries = CONTINENT_FACTS[continent]
    return {
        "name":         person,
        "city":         city,
        "country":      country,
        "continent":    continent,
        "language":     language,
        "capital":      capital,
        "num_countries": str(num_countries),
    }


def _render(template: str, facts: dict[str, str]) -> str:
    return template.format(**facts)


def generate(num_samples: int = 800, seed: int = 42) -> list[dict[str, Any]]:
    """Generate multi-hop reasoning samples.

    Distributes samples evenly across 2-hop, 3-hop, and 4-hop chains.
    Within each hop level, phrasing variants are cycled for diversity.

    Args:
        num_samples: Total number of samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of sample dicts with keys: context, question, answer, hops,
        relay_point.
    """
    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []

    # How many per hop level (distribute remainder to 2-hop)
    per_hop = num_samples // 3
    counts = {2: num_samples - 2 * per_hop, 3: per_hop, 4: per_hop}

    template_map = {2: _2HOP_TEMPLATES, 3: _3HOP_TEMPLATES, 4: _4HOP_TEMPLATES}

    for hops, n in counts.items():
        templates = template_map[hops]
        # Sample (person, city) pairs with replacement
        pairs = [(p, c) for p, c in PEOPLE if c in CITIES]
        chosen = [rng.choice(pairs) for _ in range(n)]

        for i, (person, city) in enumerate(chosen):
            facts = _build_fact(person, city)
            tmpl = templates[i % len(templates)]
            ctx_tmpl, q_tmpl, a_tmpl = tmpl
            sample = {
                "context":    _render(ctx_tmpl, facts),
                "question":   _render(q_tmpl, facts),
                "answer":     _render(a_tmpl, facts),
                "hops":       hops,
                "relay_point": RELAY_POINT,
            }
            samples.append(sample)

    rng.shuffle(samples)
    return samples
