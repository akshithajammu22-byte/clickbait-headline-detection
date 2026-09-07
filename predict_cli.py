import sys
import os
import argparse
from colorama import init, Fore, Style

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predictor import ClickbaitPredictor

init(autoreset=True)

def print_banner():
    banner = f"""{Fore.CYAN}{Style.BRIGHT}
======================================================================
  🔥 CLICKBAIT HEADLINE DETECTION & SENSATIONALISM ANALYZER 🔥
======================================================================{Style.RESET_ALL}"""
    print(banner)

def display_prediction(res):
    if "error" in res:
        print(f"{Fore.RED}[!] Error: {res['error']}{Style.RESET_ALL}")
        return

    headline = res["headline"]
    is_cb = res["is_clickbait"]
    prob = res["clickbait_probability"]
    risk = res["risk_level"]
    model = res["model_used"]
    verdict = res["verdict"]
    ling = res["linguistic_breakdown"]
    rewrite = res["neutral_rewrite"]

    pred_color = Fore.RED if is_cb else Fore.GREEN
    tag_badge = f"{pred_color}{Style.BRIGHT}[ {res['prediction'].upper()} ]{Style.RESET_ALL}"

    print(f"\n{Fore.YELLOW}📰 HEADLINE:{Style.RESET_ALL} \"{headline}\"")
    print(f"📊 VERDICT : {tag_badge} (Clickbait Probability: {pred_color}{prob:.1f}%{Style.RESET_ALL})")
    print(f"⚠️ RISK TIER: {pred_color}{risk}{Style.RESET_ALL}")
    print(f"🤖 MODEL    : {Fore.CYAN}{model}{Style.RESET_ALL}")
    print(f"📝 SUMMARY  : {verdict}")

    print(f"\n{Fore.BLUE}--- LINGUISTIC & SENSATIONALISM INDICATORS ---{Style.RESET_ALL}")
    print(f"  • Sensationalism Score : {ling['sensationalism_score']}/10.0")
    print(f"  • Capitalization Ratio : {ling['capitalization_pct']}% (All-Caps Words: {ling['all_caps_words']})")
    print(f"  • Exclamations (!)     : {ling['exclamation_marks']} | Questions (?): {ling['question_marks']}")
    print(f"  • Listicle Pattern     : {'Yes' if ling['listicle_pattern'] else 'No'}")
    print(f"  • Click Triggers Found : {ling['clickbait_triggers_found']}")
    print(f"  • Word / Char Count    : {ling['word_count']} words ({ling['char_length']} chars)")

    if is_cb:
        print(f"\n{Fore.MAGENTA}✨ AI OBJECTIVE JOURNALISTIC REWRITE:{Style.RESET_ALL}")
        print(f"   \"{rewrite}\"")
    print("=" * 70)

def main():
    print_banner()
    parser = argparse.ArgumentParser(description="Clickbait Headline Detection CLI")
    parser.add_argument("--headline", "-t", type=str, help="Headline text to analyze")
    parser.add_argument("--file", "-f", type=str, help="Path to text/csv file with headlines")
    args = parser.parse_args()

    print("[*] Initializing Clickbait ML inference engine...")
    predictor = ClickbaitPredictor()

    if args.headline:
        res = predictor.predict_single(args.headline)
        display_prediction(res)
    elif args.file:
        if not os.path.exists(args.file):
            print(f"{Fore.RED}[!] File not found: {args.file}{Style.RESET_ALL}")
            return
        with open(args.file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        print(f"[*] Analyzing batch of {len(lines)} headlines from {args.file}...\n")
        for line in lines[:20]:
            res = predictor.predict_single(line)
            display_prediction(res)
    else:
        # Interactive loop
        print(f"{Fore.GREEN}[*] Interactive Mode Ready! Enter a headline (or 'quit' / 'exit' to stop):{Style.RESET_ALL}\n")
        sample_headlines = [
            "15 Shocking Facts You Won't Believe Actually Happened!",
            "Federal Reserve Cuts Key Interest Rate by 25 Basis Points",
            "She Opened The Mystery Box And What Happened Next Will Leave You In Tears",
            "NASA Mars Rover Discovers Evidence of Ancient Organic Compounds"
        ]
        print(f"{Fore.CYAN}Try entering one of these samples:{Style.RESET_ALL}")
        for s in sample_headlines:
            print(f"  - \"{s}\"")
        print()

        while True:
            try:
                user_input = input(f"{Fore.YELLOW}Enter Headline > {Style.RESET_ALL}").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["quit", "exit", "q"]:
                    print(f"{Fore.GREEN}Exiting. Goodbye!{Style.RESET_ALL}")
                    break
                res = predictor.predict_single(user_input)
                display_prediction(res)
            except (KeyboardInterrupt, EOFError):
                print(f"\n{Fore.GREEN}Exiting. Goodbye!{Style.RESET_ALL}")
                break

if __name__ == "__main__":
    main()
