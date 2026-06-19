from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple


def load_card_csv(path: str) -> List[Dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def load_card_ids(csv_path: str) -> List[int]:
    cards = load_card_csv(csv_path)
    return [int(row["Card ID"]) for row in cards]


def _normalize_type(type_str: str) -> str:
    return type_str.strip().lower()


def find_basic_pokemon(cards: List[Dict[str, str]], seed: int | None = None) -> Dict[str, str]:
    basics = [
        card
        for card in cards
        if card["Stage (Pokémon)/Type (Energy and Trainer)"].strip() == "Basic Pokémon"
        and card["Category"].strip() == "n/a"
        and card["Type"].strip()
    ]
    if not basics:
        raise ValueError("No Basic Pokémon found in card pool")
    if seed is not None:
        random.seed(seed)
    return random.choice(basics)


def find_basic_energy(cards: List[Dict[str, str]], energy_type: str) -> Dict[str, str]:
    target = _normalize_type(energy_type)
    energies = [
        card
        for card in cards
        if card["Stage (Pokémon)/Type (Energy and Trainer)"].strip() == "Basic Energy"
        and _normalize_type(card["Type"]) == target
    ]
    if not energies:
        # Fallback: any basic energy
        energies = [
            card
            for card in cards
            if card["Stage (Pokémon)/Type (Energy and Trainer)"].strip() == "Basic Energy"
        ]
    if not energies:
        raise ValueError("No Basic Energy found in card pool")
    return energies[0]


def build_simple_deck(
    csv_path: str,
    basic_pokemon_copies: int = 4,
    seed: int | None = None,
) -> List[int]:
    """Build a minimal legal deck: 4 copies of one Basic Pokémon + matching energies."""
    cards = load_card_csv(csv_path)
    basic = find_basic_pokemon(cards, seed=seed)
    energy = find_basic_energy(cards, basic["Type"])

    deck = [int(basic["Card ID"])] * basic_pokemon_copies
    deck += [int(energy["Card ID"])] * (60 - len(deck))
    return deck


def build_starter_deck(csv_path: str) -> List[int]:
    """Build a simple legal Charizard-line starter deck.

    Internet-inspired starter list mapped to card IDs in the PTCGABC pool.
    Not competitive, but valid for smoke-testing the simulator.
    """
    cards = load_card_csv(csv_path)
    ids_by_name: Dict[str, int] = {}
    for card in cards:
        name = card["Card Name"].strip()
        cid = int(card["Card ID"])
        # Keep the first ID for each unique name
        if name not in ids_by_name:
            ids_by_name[name] = cid

    # Charizard evolution line (first printing IDs in the CSV)
    charmander = ids_by_name.get("Charmander")
    charmeleon = ids_by_name.get("Charmeleon")
    charizard = ids_by_name.get("Mega Charizard X ex") or ids_by_name.get("Mega Charizard Y ex")
    fire_energy = ids_by_name.get("Basic {R} Energy")

    if not all([charmander, charmeleon, charizard, fire_energy]):
        # Fall back to the simplest legal deck if the starter line is unavailable
        return build_simple_deck(csv_path, seed=42)

    deck: List[int] = []
    deck += [charmander] * 4
    deck += [charmeleon] * 3
    deck += [charizard] * 2
    deck += [fire_energy] * (60 - len(deck))
    return deck
