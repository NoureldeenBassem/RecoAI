"""
Terminal display — rich formatted output without external heavy deps.
"""

import os
import sys
import textwrap

try:
    from colorama import Fore, Back, Style, init
    init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False

    class _Dummy:
        def __getattr__(self, _): return ""
    Fore = Style = Back = _Dummy()


W = 80

LOGO = r"""
  ╔═══════════════════════════════════════════════════════════════════════════╗
  ║                                                                           ║
  ║    ██████╗ ███████╗ ██████╗ ██████╗      █████╗ ██╗                      ║
  ║    ██╔══██╗██╔════╝██╔════╝██╔═══██╗    ██╔══██╗██║                      ║
  ║    ██████╔╝█████╗  ██║     ██║   ██║    ███████║██║                      ║
  ║    ██╔══██╗██╔══╝  ██║     ██║   ██║    ██╔══██║██║                      ║
  ║    ██║  ██║███████╗╚██████╗╚██████╔╝    ██║  ██║██║                      ║
  ║    ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═════╝     ╚═╝  ╚═╝╚═╝   v1.0.0           ║
  ║                                                                           ║
  ║           AI-Powered Hybrid Recommendation Engine                         ║
  ║                    DecodeLabs Internship — Project 3                      ║
  ╚═══════════════════════════════════════════════════════════════════════════╝
"""

DOMAIN_ICONS = {"Movie": "🎬", "Book": "📚", "Music": "🎵", "Game": "🎮"}
MOOD_ICONS   = {"Intense": "⚡", "Emotional": "💙", "Lighthearted": "😄",
                "Epic": "🏆", "Calm": "🌊", "Motivational": "🔥", "Energetic": "🎯"}


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def divider(char="─", width=W):
    print(Fore.CYAN + char * width + Style.RESET_ALL)


def header(text: str):
    print()
    print(Fore.CYAN + Style.BRIGHT + "  ╔" + "═" * (W - 4) + "╗")
    padded = text.center(W - 4)
    print(Fore.CYAN + "  ║" + Fore.WHITE + Style.BRIGHT + padded + Fore.CYAN + "║")
    print(Fore.CYAN + "  ╚" + "═" * (W - 4) + "╝" + Style.RESET_ALL)
    print()


def print_logo():
    print(Fore.CYAN + Style.BRIGHT + LOGO + Style.RESET_ALL)


def input_prompt(text: str, default: str = "") -> str:
    prompt = f"  {Fore.GREEN}▶ {Fore.WHITE}{text}"
    if default:
        prompt += f" {Fore.YELLOW}[{default}]{Style.RESET_ALL}"
    prompt += f" {Fore.CYAN}:{Style.RESET_ALL} "
    val = input(prompt).strip()
    return val if val else default


def info(text: str):
    print(f"  {Fore.CYAN}ℹ  {Style.RESET_ALL}{text}")


def success(text: str):
    print(f"  {Fore.GREEN}✔  {Style.RESET_ALL}{Fore.WHITE}{text}{Style.RESET_ALL}")


def warn(text: str):
    print(f"  {Fore.YELLOW}⚠  {Style.RESET_ALL}{text}")


def error(text: str):
    print(f"  {Fore.RED}✘  {Style.RESET_ALL}{text}")


def loading(text: str = "Computing recommendations"):
    steps = ["⠋", "⠙", "⠸", "⠴", "⠦", "⠇"]
    import time
    for s in steps * 3:
        sys.stdout.write(f"\r  {Fore.CYAN}{s} {text}...{Style.RESET_ALL}  ")
        sys.stdout.flush()
        time.sleep(0.06)
    print(f"\r  {Fore.GREEN}✔ Done!{' ' * 30}{Style.RESET_ALL}")


def confidence_bar(pct: float, width: int = 20) -> str:
    filled = int(pct / 100 * width)
    bar = "█" * filled + "░" * (width - filled)
    color = Fore.GREEN if pct >= 70 else Fore.YELLOW if pct >= 50 else Fore.RED
    return f"{color}{bar}{Style.RESET_ALL} {Fore.WHITE}{pct:.1f}%{Style.RESET_ALL}"


def print_recommendation_card(rank: int, row, explanation: str):
    domain_icon = DOMAIN_ICONS.get(row["domain"], "•")
    mood_icon   = MOOD_ICONS.get(row["mood"], "")

    print(f"  {Fore.YELLOW}{'─'*76}{Style.RESET_ALL}")
    rank_color = Fore.YELLOW if rank == 1 else Fore.CYAN if rank <= 3 else Fore.WHITE
    print(f"  {rank_color}#{rank:<3}{Style.RESET_ALL}  "
          f"{domain_icon} {Fore.WHITE}{Style.BRIGHT}{row['title']:<35}{Style.RESET_ALL}  "
          f"{Fore.MAGENTA}{row['domain']:<8}{Style.RESET_ALL}  "
          f"{Fore.YELLOW}★ {row['rating']}{Style.RESET_ALL}")

    print(f"       {Fore.CYAN}Genre:{Style.RESET_ALL} {row['genre']} / {row['subgenre']}"
          f"   {Fore.CYAN}Mood:{Style.RESET_ALL} {mood_icon} {row['mood']}"
          f"   {Fore.CYAN}Year:{Style.RESET_ALL} {int(row['year'])}")

    cb = confidence_bar(row["confidence_pct"])
    print(f"       {Fore.CYAN}Confidence:{Style.RESET_ALL} {cb}")

    wrapped = textwrap.fill(explanation, width=66)
    for i, line in enumerate(wrapped.split("\n")):
        prefix = f"       {Fore.CYAN}Why:{Style.RESET_ALL} " if i == 0 else "            "
        print(f"{prefix}{Fore.WHITE}{line}{Style.RESET_ALL}")


def print_recommendations(recs, explanations: list[str]):
    header("🎯  YOUR PERSONALIZED RECOMMENDATIONS")
    for i, (_, row) in enumerate(recs.iterrows()):
        print_recommendation_card(int(row["rank"]), row, explanations[i])
    print(f"\n  {Fore.YELLOW}{'─'*76}{Style.RESET_ALL}\n")


def print_favorites(favorites: list[dict]):
    if not favorites:
        warn("No favorites saved yet.")
        return
    header("⭐  SAVED FAVORITES")
    for fav in favorites:
        icon = DOMAIN_ICONS.get(fav.get("domain", ""), "•")
        print(f"  {icon}  {Fore.WHITE}{fav['title']:<40}{Style.RESET_ALL}"
              f"  {Fore.YELLOW}★ {fav['rating']}{Style.RESET_ALL}"
              f"  {Fore.CYAN}[{fav['id']}]{Style.RESET_ALL}")
    print()


def print_history(history: list[dict]):
    if not history:
        warn("No search history yet.")
        return
    header("📋  RECOMMENDATION HISTORY")
    for i, h in enumerate(history, 1):
        prefs = h["preferences"]
        genre_str = ", ".join(prefs.get("genres", ["-"])) or "-"
        print(f"  {Fore.CYAN}Session {i}{Style.RESET_ALL}  {h['timestamp']}")
        print(f"    Genres: {genre_str}  |  Mood: {', '.join(prefs.get('moods', ['-']))}")
        print(f"    Top picks: {Fore.WHITE}{', '.join(h['top_picks'])}{Style.RESET_ALL}")
        print()


def numbered_menu(title: str, options: list[str]) -> int:
    """Print a numbered menu, return the selected 1-based index."""
    print(f"\n  {Fore.CYAN}{Style.BRIGHT}{title}{Style.RESET_ALL}")
    divider()
    for i, opt in enumerate(options, 1):
        print(f"  {Fore.YELLOW}[{i}]{Style.RESET_ALL} {opt}")
    divider()
    while True:
        raw = input_prompt("Enter choice", "1")
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw)
        warn(f"Please enter a number between 1 and {len(options)}.")


def multi_select_menu(title: str, options: list[str]) -> list[str]:
    """Let user pick multiple comma-separated numbers."""
    print(f"\n  {Fore.CYAN}{Style.BRIGHT}{title}{Style.RESET_ALL}")
    divider()
    for i, opt in enumerate(options, 1):
        print(f"  {Fore.YELLOW}[{i}]{Style.RESET_ALL} {opt}")
    print(f"  {Fore.YELLOW}[0]{Style.RESET_ALL} All / Skip")
    divider()
    raw = input_prompt("Enter numbers separated by commas (or 0 for all)", "0")
    if raw == "0" or not raw:
        return options
    chosen = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(options):
                chosen.append(options[idx])
    return chosen if chosen else options