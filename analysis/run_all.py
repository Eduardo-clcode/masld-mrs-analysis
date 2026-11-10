"""Run the selected analysis, model-screening sweep, and figure generation."""

from run_analysis import main as run_selected_analysis
from run_model_screening import run as run_model_screening
from regenerate_figures import main as generate_figures


def main() -> None:
    print("[1/3] Running selected analyses")
    run_selected_analysis()

    print("[2/3] Running model-family screening")
    run_model_screening()

    print("[3/3] Regenerating figures")
    generate_figures()

    print("All analysis and figure-generation steps completed.")


if __name__ == "__main__":
    main()

