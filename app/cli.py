from __future__ import annotations

import typer

from app.ingest import main as ingest_main
from app.service import execute_research
from app.store import get_run, list_steps
from app.db import get_conn

app = typer.Typer(no_args_is_help=True)


@app.command()
def ask(query: str, baseline: str = "agentic"):
    result = execute_research(query, baseline=baseline)
    typer.echo(f"Run: {result['run_id']}")
    typer.echo(f"Status: {result['status']}")
    typer.echo(f"Retrieval rounds: {result.get('retrieval_rounds')}")
    cites = result.get("citations") or []
    typer.echo(f"Citations: {len(cites)}")
    typer.echo("")
    typer.echo("Answer:")
    typer.echo(result.get("answer") or "")


@app.command()
def show(run_id: str):
    with get_conn() as conn:
        row = get_run(conn, run_id)
        steps = list_steps(conn, run_id)
    if not row:
        raise typer.Exit(code=1)
    typer.echo(f"id: {row['id']}")
    typer.echo(f"query: {row['query']}")
    typer.echo(f"status: {row['status']}")
    typer.echo(f"rounds: {row['retrieval_rounds']}")
    typer.echo(f"answer:\n{row['final_answer']}")
    typer.echo("steps:")
    for s in steps:
        typer.echo(f"  {s['step_index']} {s['node']} {s['duration_ms']}ms")


@app.command()
def ingest():
    ingest_main()


@app.command(name="eval")
def eval_cmd(
    baseline: str = "agentic",
    live: bool = False,
    routing: str = "single",
    concurrency: int | None = None,
):
    from app.evaluation.runner import run_eval

    run_eval(
        baseline=baseline,
        live=live,
        print_report=True,
        model_routing=routing,
        concurrency=concurrency,
    )


def main():
    app()


if __name__ == "__main__":
    main()
