#!/usr/bin/env python3
"""Quick optimization helper for common scenarios."""

import sys
import yaml
import shutil
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.panel import Panel

console = Console()


def backup_config():
    """Backup current config."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"config/config.backup_{timestamp}.yaml"
    shutil.copy("config/config.yaml", backup_path)
    console.print(f"[green]✓ Config backed up to {backup_path}[/green]")
    return backup_path


def update_config(updates: dict):
    """Update config file with new values."""
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    config = deep_update(config, updates)

    with open("config/config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    console.print("[green]✓ Config updated[/green]")


@click.group()
def cli():
    """Quick optimization helpers for common problems."""
    pass


@cli.command()
def retrieval_poor():
    """Fix: Retrieval finds wrong chunks."""
    console.print(Panel.fit(
        "[bold red]Problem:[/bold red] Retrieval findet falsche Chunks\n"
        "[bold cyan]Solution:[/bold cyan] Optimiere Retrieval-Settings",
        border_style="red"
    ))

    backup_config()

    updates = {
        'retrieval': {
            'hybrid': {
                'dense_weight': 0.5,
                'sparse_weight': 0.5
            },
            'final_top_k': 7
        }
    }

    update_config(updates)

    console.print("\n[bold green]✓ Retrieval settings optimized![/bold green]")
    console.print("\n[yellow]Nächste Schritte:[/yellow]")
    console.print("1. Teste: python debug.py test-retrieval 'Deine Frage'")


@cli.command()
def chunks_too_large():
    """Fix: Chunks are too large."""
    console.print(Panel.fit(
        "[bold red]Problem:[/bold red] Chunks sind zu groß\n"
        "[bold cyan]Solution:[/bold cyan] Chunk-Größe reduzieren",
        border_style="red"
    ))

    new_size = Prompt.ask("Neue Chunk-Größe", default="256")
    backup_config()

    overlap = int(int(new_size) * 0.2)

    updates = {
        'chunking': {
            'semantic': {
                'max_chunk_size': int(new_size)
            },
            'sliding_window': {
                'chunk_size': int(new_size),
                'overlap': overlap
            }
        }
    }

    update_config(updates)

    console.print(f"\n[bold green]✓ Chunk-Größe → {new_size}[/bold green]")
    console.print("\n[yellow]Nächste Schritte:[/yellow]")
    console.print("1. NEU INDEXIEREN: python pipeline.py build-index")


@cli.command()
def hallucinations():
    """Fix: LLM hallucinates."""
    console.print(Panel.fit(
        "[bold red]Problem:[/bold red] LLM halluziniert\n"
        "[bold cyan]Solution:[/bold cyan] Temperatur senken",
        border_style="red"
    ))

    backup_config()

    updates = {
        'llm': {
            'openai': {'temperature': 0.0},
            'claude': {'temperature': 0.0}
        },
        'prompts': {
            'abstaining': {'threshold': 0.7}
        }
    }

    update_config(updates)

    console.print("\n[bold green]✓ Anti-Hallucination Settings aktiviert![/bold green]")


@cli.command()
def abstains_too_much():
    """Fix: System abstains too often."""
    console.print(Panel.fit(
        "[bold red]Problem:[/bold red] System sagt zu oft 'Ich weiß es nicht'\n"
        "[bold cyan]Solution:[/bold cyan] Threshold senken",
        border_style="red"
    ))

    backup_config()

    updates = {
        'prompts': {
            'abstaining': {'threshold': 0.5}
        }
    }

    update_config(updates)

    console.print("\n[bold green]✓ Weniger konservativ![/bold green]")


@cli.command()
def slow_performance():
    """Fix: System is too slow."""
    console.print(Panel.fit(
        "[bold red]Problem:[/bold red] System ist zu langsam\n"
        "[bold cyan]Solution:[/bold cyan] Performance-Optimierungen",
        border_style="red"
    ))

    has_gpu = Confirm.ask("GPU verfügbar?", default=False)
    backup_config()

    updates = {
        'caching': {
            'enabled': True,
            'query': {
                'enabled': True,
                'ttl': 3600,
                'max_size': 1000
            }
        }
    }

    if has_gpu:
        updates['embeddings'] = {
            'device': 'cuda',
            'batch_size': 64
        }

    update_config(updates)

    console.print("\n[bold green]✓ Performance-Settings optimiert![/bold green]")


@cli.command()
def interactive():
    """Interactive optimization wizard."""
    console.print(Panel.fit(
        "[bold cyan]RAG Optimierungs-Wizard[/bold cyan]\n"
        "Beantworte ein paar Fragen für automatische Optimierung",
        border_style="cyan"
    ))

    q1 = Confirm.ask("1. Findet Retrieval die richtigen Chunks?", default=True)
    q2 = Confirm.ask("2. Sind die Antworten faktisch korrekt?", default=True)
    q3 = Confirm.ask("3. Halluziniert das LLM manchmal?", default=False)
    q4 = Confirm.ask("4. Sagt das System zu oft 'Ich weiß es nicht'?", default=False)

    console.print("\n[bold cyan]Empfehlungen:[/bold cyan]\n")

    if not q1:
        console.print("[yellow]→ Laufe: python optimize.py retrieval_poor[/yellow]")

    if not q2 or q3:
        console.print("[yellow]→ Laufe: python optimize.py hallucinations[/yellow]")

    if q4:
        console.print("[yellow]→ Laufe: python optimize.py abstains_too_much[/yellow]")

    if q1 and q2 and not q3 and not q4:
        console.print("[bold green]✓ System scheint gut konfiguriert![/bold green]")


if __name__ == '__main__':
    cli()
