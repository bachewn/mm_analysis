
import typer
import json
from typing import Optional, List
from mm_core import (
    load_dataset,
    compute_win_pct,
    compute_filtered_df,
    compute_lower_seed_win_pct,
    compute_lower_seed_filtered_df,
)

app = typer.Typer(add_completion=False, no_args_is_help=True)

@app.command()
def winpct(
    path: str = typer.Option(..., help="Path to Excel/Parquet/CSV with the data", prompt=False),
    pace_diff_max: float = 2.0,
    require_higher_seed: bool = True,
    require_positive_net: bool = True,
    by_year: bool = True,
    year: Optional[List[int]] = typer.Option(None, help="Filter to one or more specific years (repeat flag)"),
    export_csv: Optional[str] = typer.Option(None, help="Optional path to export the filtered rows"),
    json_output: bool = typer.Option(False, "--json", help="Return JSON with counts"),
):
    """Compute win % under the specified filters."""
    df = load_dataset(path)
    if json_output:
        result = compute_win_pct(
            df,
            pace_diff_max=pace_diff_max,
            require_higher_seed=require_higher_seed,
            require_positive_net=require_positive_net,
            by_year=by_year,
            years=year,
            return_counts=True,
        )
        typer.echo(json.dumps(result, indent=2))
    else:
        result = compute_win_pct(
            df,
            pace_diff_max=pace_diff_max,
            require_higher_seed=require_higher_seed,
            require_positive_net=require_positive_net,
            by_year=by_year,
            years=year,
        )
        if by_year:
            typer.echo(result.to_string())
        else:
            typer.echo(f"{result:.2f}")

    if export_csv:
        filt = compute_filtered_df(
            df,
            pace_diff_max=pace_diff_max,
            require_higher_seed=require_higher_seed,
            require_positive_net=require_positive_net,
            years=year,
        )
        filt.to_csv(export_csv, index=False)
        typer.echo(f"Exported filtered rows to {export_csv}")

@app.command()
def lowerseed(
    path: str = typer.Option(..., help="Path to Excel/Parquet/CSV with the data", prompt=False),
    pace_diff_max: float = 2.0,
    by_year: bool = True,
    year: Optional[List[int]] = typer.Option(None, help="Filter to one or more specific years (repeat flag)"),
    export_csv: Optional[str] = typer.Option(None, help="Optional path to export the filtered rows"),
    json_output: bool = typer.Option(False, "--json", help="Return JSON with counts"),
):
    """Compute win % for lower-seeded teams under specified filters."""
    df = load_dataset(path)
    if json_output:
        result = compute_lower_seed_win_pct(
            df,
            pace_diff_max=pace_diff_max,
            by_year=by_year,
            years=year,
            return_counts=True,
        )
        typer.echo(json.dumps(result, indent=2))
    else:
        result = compute_lower_seed_win_pct(
            df,
            pace_diff_max=pace_diff_max,
            by_year=by_year,
            years=year,
        )
        if by_year:
            typer.echo(result.to_string())
        else:
            typer.echo(f"{result:.2f}")

    if export_csv:
        filt = compute_lower_seed_filtered_df(
            df,
            pace_diff_max=pace_diff_max,
            years=year,
        )
        filt.to_csv(export_csv, index=False)
        typer.echo(f"Exported filtered rows to {export_csv}")

if __name__ == "__main__":
    app()
