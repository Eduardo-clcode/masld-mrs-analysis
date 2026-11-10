
from run_analysis import main as run_selected_analysis
from run_model_screening import run as run_model_screening
from regenerate_figures import main as generate_figures


def main() -> None:
    print("Running selected analyses")
    run_selected_analysis()

    print("Running model-family screening")
    run_model_screening()

    print("Generating figures")
    generate_figures()

    print("All analysis and steps completed.")


if __name__ == "__main__":
    main()

