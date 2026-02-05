import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

from src.benchmark.run_benchmark import (
    benchmark_pgvector,
    benchmark_qdrant,
    benchmark_faiss,
    benchmark_milvus,
    benchmark_mongodb,
    compute_ground_truth,
)
from src.pipeline.dataset import load_synthetic_dataset
from src.pipeline.embed import EmbeddingGenerator


def create_performance_dashboard(df: pd.DataFrame, output_path: str):
    """Create a dashboard focused on query performance metrics."""
    # Get unique dataset sizes and databases
    unique_sizes = sorted(df["n_dataset"].unique())
    db_names = sorted(df["db"].unique())
    
    # Color palette
    db_colors = {
        "faiss": "#2ca02c",
        "milvus": "#1f77b4",
        "pgvector": "#ff7f0e",
        "qdrant": "#d62728",
        "mongodb_atlas": "#00ED64",  # MongoDB green
    }
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. p50 Latency - top left
    ax1 = fig.add_subplot(gs[0, 0])
    for db in db_names:
        sub = df[df["db"] == db].sort_values("n_dataset")
        ax1.plot(sub["n_dataset"], sub["p50_ms"], marker="o", linewidth=3,
                markersize=10, label=db.upper(), color=db_colors.get(db, "#000000"))
    ax1.set_xlabel("Dataset Size", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Latency (ms)", fontsize=11, fontweight="bold")
    ax1.set_title("p50 Latency (Median)", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle=":")
    ax1.legend(fontsize=10, framealpha=0.9)
    
    # 2. p95 Latency - top middle
    ax2 = fig.add_subplot(gs[0, 1])
    for db in db_names:
        sub = df[df["db"] == db].sort_values("n_dataset")
        ax2.plot(sub["n_dataset"], sub["p95_ms"], marker="s", linewidth=3,
                markersize=10, label=db.upper(), color=db_colors.get(db, "#000000"))
    ax2.set_xlabel("Dataset Size", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Latency (ms)", fontsize=11, fontweight="bold")
    ax2.set_title("p95 Latency (95th Percentile)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, linestyle=":")
    ax2.legend(fontsize=10, framealpha=0.9)
    
    # 3. p99 Latency - top right
    ax3 = fig.add_subplot(gs[0, 2])
    for db in db_names:
        sub = df[df["db"] == db].sort_values("n_dataset")
        ax3.plot(sub["n_dataset"], sub["p99_ms"], marker="^", linewidth=3,
                markersize=10, label=db.upper(), color=db_colors.get(db, "#000000"))
    ax3.set_xlabel("Dataset Size", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Latency (ms)", fontsize=11, fontweight="bold")
    ax3.set_title("p99 Latency (99th Percentile)", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, linestyle=":")
    ax3.legend(fontsize=10, framealpha=0.9)
    
    # 4. Average Latency - middle left
    ax4 = fig.add_subplot(gs[1, 0])
    for db in db_names:
        sub = df[df["db"] == db].sort_values("n_dataset")
        ax4.plot(sub["n_dataset"], sub["avg_ms"], marker="D", linewidth=3,
                markersize=10, label=db.upper(), color=db_colors.get(db, "#000000"))
    ax4.set_xlabel("Dataset Size", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Latency (ms)", fontsize=11, fontweight="bold")
    ax4.set_title("Average Latency", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3, linestyle=":")
    ax4.legend(fontsize=10, framealpha=0.9)
    
    # 5. Latency comparison bars for largest size - middle middle
    largest_size = max(unique_sizes)
    df_largest = df[df["n_dataset"] == largest_size].copy()
    x_pos = np.arange(len(db_names))
    colors_list = [db_colors.get(db, "#808080") for db in db_names]
    
    ax5 = fig.add_subplot(gs[1, 1])
    p50_vals = [df_largest[df_largest["db"] == db]["p50_ms"].iloc[0] for db in db_names]
    p95_vals = [df_largest[df_largest["db"] == db]["p95_ms"].iloc[0] for db in db_names]
    width = 0.35
    bars1 = ax5.bar(x_pos - width/2, p50_vals, width, label="p50", color="#4CAF50", 
                   alpha=0.8, edgecolor="black", linewidth=1.5)
    bars2 = ax5.bar(x_pos + width/2, p95_vals, width, label="p95", color="#FF9800", 
                   alpha=0.8, edgecolor="black", linewidth=1.5)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([db.upper() for db in db_names], fontsize=10, fontweight="bold")
    ax5.set_ylabel("Latency (ms)", fontsize=11, fontweight="bold")
    ax5.set_title(f"Latency Comparison (N={largest_size:,})", fontsize=12, fontweight="bold")
    ax5.legend(fontsize=10, framealpha=0.9)
    ax5.grid(True, axis="y", alpha=0.3, linestyle=":")
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight="bold")
    
    # 6. Latency distribution (p50, p95, p99) for largest size - middle right
    ax6 = fig.add_subplot(gs[1, 2])
    p99_vals = [df_largest[df_largest["db"] == db]["p99_ms"].iloc[0] for db in db_names]
    x = np.arange(len(db_names))
    width = 0.25
    ax6.bar(x - width, p50_vals, width, label="p50", color="#4CAF50", alpha=0.8, edgecolor="black")
    ax6.bar(x, p95_vals, width, label="p95", color="#FF9800", alpha=0.8, edgecolor="black")
    ax6.bar(x + width, p99_vals, width, label="p99", color="#F44336", alpha=0.8, edgecolor="black")
    ax6.set_xticks(x)
    ax6.set_xticklabels([db.upper() for db in db_names], fontsize=10, fontweight="bold")
    ax6.set_ylabel("Latency (ms)", fontsize=11, fontweight="bold")
    ax6.set_title(f"Latency Distribution (N={largest_size:,})", fontsize=12, fontweight="bold")
    ax6.legend(fontsize=10, framealpha=0.9)
    ax6.grid(True, axis="y", alpha=0.3, linestyle=":")
    
    # 7-9. Summary table and stats - bottom row
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis("off")
    
    # Create summary table
    summary_data = []
    for db in db_names:
        sub = df_largest[df_largest["db"] == db].iloc[0]
        summary_data.append([
            db.upper(),
            f"{sub['p50_ms']:.2f}",
            f"{sub['p95_ms']:.2f}",
            f"{sub['p99_ms']:.2f}",
            f"{sub['avg_ms']:.2f}",
        ])
    
    table = ax7.table(
        cellText=summary_data,
        colLabels=["Database", "p50 (ms)", "p95 (ms)", "p99 (ms)", "Avg (ms)"],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style table
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor("#2196F3")
        table[(0, i)].set_text_props(weight="bold", color="white")
    
    for i in range(1, len(summary_data) + 1):
        table[(i, 0)].set_facecolor("#f0f0f0")
        table[(i, 0)].set_text_props(weight="bold")
    
    plt.suptitle("Query Performance Dashboard", fontsize=16, fontweight="bold", y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_throughput_dashboard(df: pd.DataFrame, output_path: str):
    """Create a dashboard focused on throughput and efficiency metrics."""
    unique_sizes = sorted(df["n_dataset"].unique())
    db_names = sorted(df["db"].unique())
    
    db_colors = {
        "faiss": "#2ca02c",
        "milvus": "#1f77b4",
        "pgvector": "#ff7f0e",
        "qdrant": "#d62728",
        "mongodb_atlas": "#00ED64",  # MongoDB green
    }
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. Insertion Time - top left
    ax1 = fig.add_subplot(gs[0, 0])
    for db in db_names:
        sub = df[df["db"] == db].sort_values("n_dataset")
        ax1.plot(sub["n_dataset"], sub["insert_ms"], marker="o", linewidth=3,
                markersize=10, label=db.upper(), color=db_colors.get(db, "#000000"))
    ax1.set_xlabel("Dataset Size", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Time (ms)", fontsize=11, fontweight="bold")
    ax1.set_title("Insertion Time", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle=":")
    ax1.legend(fontsize=10, framealpha=0.9)
    
    # 2. Query Throughput (QPS) - top middle
    ax2 = fig.add_subplot(gs[0, 1])
    for db in db_names:
        sub = df[df["db"] == db].sort_values("n_dataset")
        if "qps" in sub.columns:
            ax2.plot(sub["n_dataset"], sub["qps"], marker="s", linewidth=3,
                    markersize=10, label=db.upper(), color=db_colors.get(db, "#000000"))
    ax2.set_xlabel("Dataset Size", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Queries per Second", fontsize=11, fontweight="bold")
    ax2.set_title("Query Throughput (QPS)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, linestyle=":")
    ax2.legend(fontsize=10, framealpha=0.9)
    
    # 3. Memory Usage - top right
    ax3 = fig.add_subplot(gs[0, 2])
    for db in db_names:
        sub = df[df["db"] == db].sort_values("n_dataset")
        if "memory_mb" in sub.columns:
            ax3.plot(sub["n_dataset"], sub["memory_mb"], marker="^", linewidth=3,
                    markersize=10, label=db.upper(), color=db_colors.get(db, "#000000"))
    ax3.set_xlabel("Dataset Size", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Memory (MB)", fontsize=11, fontweight="bold")
    ax3.set_title("Memory Usage", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, linestyle=":")
    ax3.legend(fontsize=10, framealpha=0.9)
    
    # 4-6. Bar charts for largest size
    largest_size = max(unique_sizes)
    df_largest = df[df["n_dataset"] == largest_size].copy()
    x_pos = np.arange(len(db_names))
    colors_list = [db_colors.get(db, "#808080") for db in db_names]
    
    # Insertion time bars
    ax4 = fig.add_subplot(gs[1, 0])
    insert_vals = [df_largest[df_largest["db"] == db]["insert_ms"].iloc[0] for db in db_names]
    bars = ax4.bar(x_pos, insert_vals, color=colors_list, alpha=0.8, edgecolor="black", linewidth=1.5)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([db.upper() for db in db_names], fontsize=10, fontweight="bold")
    ax4.set_ylabel("Time (ms)", fontsize=11, fontweight="bold")
    ax4.set_title(f"Insertion Time (N={largest_size:,})", fontsize=12, fontweight="bold")
    ax4.grid(True, axis="y", alpha=0.3, linestyle=":")
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontsize=10, fontweight="bold")
    
    # QPS bars
    ax5 = fig.add_subplot(gs[1, 1])
    if "qps" in df_largest.columns:
        qps_vals = [df_largest[df_largest["db"] == db]["qps"].iloc[0] for db in db_names]
        bars = ax5.bar(x_pos, qps_vals, color=colors_list, alpha=0.8, edgecolor="black", linewidth=1.5)
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels([db.upper() for db in db_names], fontsize=10, fontweight="bold")
        ax5.set_ylabel("Queries per Second", fontsize=11, fontweight="bold")
        ax5.set_title(f"Throughput (N={largest_size:,})", fontsize=12, fontweight="bold")
        ax5.grid(True, axis="y", alpha=0.3, linestyle=":")
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=10, fontweight="bold")
    
    # Memory bars
    ax6 = fig.add_subplot(gs[1, 2])
    if "memory_mb" in df_largest.columns:
        mem_vals = []
        for db in db_names:
            val = df_largest[df_largest["db"] == db]["memory_mb"].iloc[0]
            mem_vals.append(val if not pd.isna(val) else 0)
        bars = ax6.bar(x_pos, mem_vals, color=colors_list, alpha=0.8, edgecolor="black", linewidth=1.5)
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels([db.upper() for db in db_names], fontsize=10, fontweight="bold")
        ax6.set_ylabel("Memory (MB)", fontsize=11, fontweight="bold")
        ax6.set_title(f"Memory Usage (N={largest_size:,})", fontsize=12, fontweight="bold")
        ax6.grid(True, axis="y", alpha=0.3, linestyle=":")
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight="bold")
    
    # Summary table - bottom row
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis("off")
    
    summary_data = []
    for db in db_names:
        sub = df_largest[df_largest["db"] == db].iloc[0]
        summary_data.append([
            db.upper(),
            f"{sub['insert_ms']:.0f}",
            f"{sub['qps']:.0f}" if "qps" in sub else "N/A",
            f"{sub['memory_mb']:.1f}" if "memory_mb" in sub and not pd.isna(sub['memory_mb']) else "N/A",
        ])
    
    table = ax7.table(
        cellText=summary_data,
        colLabels=["Database", "Insert Time (ms)", "Throughput (QPS)", "Memory (MB)"],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")
    
    for i in range(1, len(summary_data) + 1):
        table[(i, 0)].set_facecolor("#f0f0f0")
        table[(i, 0)].set_text_props(weight="bold")
    
    plt.suptitle("Throughput & Efficiency Dashboard", fontsize=16, fontweight="bold", y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_accuracy_dashboard(df: pd.DataFrame, output_path: str):
    """Create a dashboard focused on search accuracy metrics."""
    unique_sizes = sorted(df["n_dataset"].unique())
    db_names = sorted(df["db"].unique())
    
    db_colors = {
        "faiss": "#2ca02c",
        "milvus": "#1f77b4",
        "pgvector": "#ff7f0e",
        "qdrant": "#d62728",
        "mongodb_atlas": "#00ED64",  # MongoDB green
    }
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    # 1. Recall@K over dataset sizes - top left
    ax1 = fig.add_subplot(gs[0, 0])
    for db in db_names:
        sub = df[df["db"] == db].sort_values("n_dataset")
        if "recall_at_k" in sub.columns:
            ax1.plot(sub["n_dataset"], sub["recall_at_k"], marker="o", linewidth=3,
                    markersize=10, label=db.upper(), color=db_colors.get(db, "#000000"))
    ax1.set_xlabel("Dataset Size", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Recall@K", fontsize=11, fontweight="bold")
    ax1.set_title("Recall@K vs Dataset Size", fontsize=12, fontweight="bold")
    ax1.set_ylim([0, 1.1])
    ax1.grid(True, alpha=0.3, linestyle=":")
    ax1.legend(fontsize=10, framealpha=0.9)
    
    # 2. Recall@K bar chart for largest size - top middle
    largest_size = max(unique_sizes)
    df_largest = df[df["n_dataset"] == largest_size].copy()
    x_pos = np.arange(len(db_names))
    colors_list = [db_colors.get(db, "#808080") for db in db_names]
    
    ax2 = fig.add_subplot(gs[0, 1])
    if "recall_at_k" in df_largest.columns:
        recall_vals = [df_largest[df_largest["db"] == db]["recall_at_k"].iloc[0] for db in db_names]
        bars = ax2.bar(x_pos, recall_vals, color=colors_list, alpha=0.8, edgecolor="black", linewidth=1.5)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([db.upper() for db in db_names], fontsize=10, fontweight="bold")
        ax2.set_ylabel("Recall@K", fontsize=11, fontweight="bold")
        ax2.set_title(f"Recall@K Comparison (N={largest_size:,})", fontsize=12, fontweight="bold")
        ax2.set_ylim([0, 1.1])
        ax2.grid(True, axis="y", alpha=0.3, linestyle=":")
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight="bold")
    
    # 3. Recall standard deviation - top right
    ax3 = fig.add_subplot(gs[0, 2])
    for db in db_names:
        sub = df[df["db"] == db].sort_values("n_dataset")
        if "recall_std" in sub.columns:
            ax3.plot(sub["n_dataset"], sub["recall_std"], marker="s", linewidth=3,
                    markersize=10, label=db.upper(), color=db_colors.get(db, "#000000"))
    ax3.set_xlabel("Dataset Size", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Recall Std Dev", fontsize=11, fontweight="bold")
    ax3.set_title("Recall Consistency", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, linestyle=":")
    ax3.legend(fontsize=10, framealpha=0.9)
    
    # Summary table - bottom row
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis("off")
    
    summary_data = []
    for db in db_names:
        sub = df_largest[df_largest["db"] == db].iloc[0]
        summary_data.append([
            db.upper(),
            f"{sub['recall_at_k']:.3f}" if "recall_at_k" in sub else "N/A",
            f"{sub['recall_std']:.3f}" if "recall_std" in sub else "N/A",
            f"{sub['n_queries']}" if "n_queries" in sub else "N/A",
        ])
    
    table = ax4.table(
        cellText=summary_data,
        colLabels=["Database", "Recall@K", "Std Dev", "N Queries"],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor("#9C27B0")
        table[(0, i)].set_text_props(weight="bold", color="white")
    
    for i in range(1, len(summary_data) + 1):
        table[(i, 0)].set_facecolor("#f0f0f0")
        table[(i, 0)].set_text_props(weight="bold")
    
    plt.suptitle("Search Accuracy Dashboard", fontsize=16, fontweight="bold", y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    """Create a comprehensive dashboard with all key metrics."""
    # Get unique dataset sizes and databases
    unique_sizes = sorted(df["n_dataset"].unique())
    db_names = sorted(df["db"].unique())
    
    # Color palette
    db_colors = {
        "faiss": "#2ca02c",
        "milvus": "#1f77b4",
        "pgvector": "#ff7f0e",
        "qdrant": "#d62728",
        "mongodb_atlas": "#00ED64",  # MongoDB green
    }
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. Latency (p50) - top left
    ax1 = fig.add_subplot(gs[0, 0])
    for db in db_names:
        sub = df[df["db"] == db].sort_values("n_dataset")
        ax1.plot(sub["n_dataset"], sub["p50_ms"], marker="o", linewidth=2.5,
                markersize=8, label=db, color=db_colors.get(db, "#000000"))
    ax1.set_xlabel("Dataset Size", fontweight="bold")
    ax1.set_ylabel("Latency (ms)", fontweight="bold")
    ax1.set_title("p50 Latency", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    # 2. Latency (p95) - top second
    ax2 = fig.add_subplot(gs[0, 1])
    for db in db_names:
        sub = df[df["db"] == db].sort_values("n_dataset")
        ax2.plot(sub["n_dataset"], sub["p95_ms"], marker="s", linewidth=2.5,
                markersize=8, label=db, color=db_colors.get(db, "#000000"))
    ax2.set_xlabel("Dataset Size", fontweight="bold")
    ax2.set_ylabel("Latency (ms)", fontweight="bold")
    ax2.set_title("p95 Latency", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    # 3. Insertion Time - top third
    ax3 = fig.add_subplot(gs[0, 2])
    for db in db_names:
        sub = df[df["db"] == db].sort_values("n_dataset")
        ax3.plot(sub["n_dataset"], sub["insert_ms"], marker="^", linewidth=2.5,
                markersize=8, label=db, color=db_colors.get(db, "#000000"))
    ax3.set_xlabel("Dataset Size", fontweight="bold")
    ax3.set_ylabel("Time (ms)", fontweight="bold")
    ax3.set_title("Insertion Time", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)
    
    # 4. Throughput (QPS) - top right
    ax4 = fig.add_subplot(gs[0, 3])
    for db in db_names:
        sub = df[df["db"] == db].sort_values("n_dataset")
        if "qps" in sub.columns:
            ax4.plot(sub["n_dataset"], sub["qps"], marker="D", linewidth=2.5,
                    markersize=8, label=db, color=db_colors.get(db, "#000000"))
    ax4.set_xlabel("Dataset Size", fontweight="bold")
    ax4.set_ylabel("QPS", fontweight="bold")
    ax4.set_title("Query Throughput", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)
    
    # 5. Memory Usage - second row left
    ax5 = fig.add_subplot(gs[1, 0])
    for db in db_names:
        sub = df[df["db"] == db].sort_values("n_dataset")
        if "memory_mb" in sub.columns:
            ax5.plot(sub["n_dataset"], sub["memory_mb"], marker="o", linewidth=2.5,
                    markersize=8, label=db, color=db_colors.get(db, "#000000"))
    ax5.set_xlabel("Dataset Size", fontweight="bold")
    ax5.set_ylabel("Memory (MB)", fontweight="bold")
    ax5.set_title("Memory Usage", fontsize=12, fontweight="bold")
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=9)
    
    # 6. Recall@K - second row second
    ax6 = fig.add_subplot(gs[1, 1])
    for db in db_names:
        sub = df[df["db"] == db].sort_values("n_dataset")
        if "recall_at_k" in sub.columns:
            ax6.plot(sub["n_dataset"], sub["recall_at_k"], marker="s", linewidth=2.5,
                    markersize=8, label=db, color=db_colors.get(db, "#000000"))
    ax6.set_xlabel("Dataset Size", fontweight="bold")
    ax6.set_ylabel("Recall@K", fontweight="bold")
    ax6.set_title("Search Accuracy", fontsize=12, fontweight="bold")
    ax6.set_ylim([0, 1.1])
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=9)
    
    # 7. Average Latency - second row third
    ax7 = fig.add_subplot(gs[1, 2])
    for db in db_names:
        sub = df[df["db"] == db].sort_values("n_dataset")
        ax7.plot(sub["n_dataset"], sub["avg_ms"], marker="^", linewidth=2.5,
                markersize=8, label=db, color=db_colors.get(db, "#000000"))
    ax7.set_xlabel("Dataset Size", fontweight="bold")
    ax7.set_ylabel("Latency (ms)", fontweight="bold")
    ax7.set_title("Average Latency", fontsize=12, fontweight="bold")
    ax7.grid(True, alpha=0.3)
    ax7.legend(fontsize=9)
    
    # 8. p99 Latency - second row right
    ax8 = fig.add_subplot(gs[1, 3])
    for db in db_names:
        sub = df[df["db"] == db].sort_values("n_dataset")
        ax8.plot(sub["n_dataset"], sub["p99_ms"], marker="D", linewidth=2.5,
                markersize=8, label=db, color=db_colors.get(db, "#000000"))
    ax8.set_xlabel("Dataset Size", fontweight="bold")
    ax8.set_ylabel("Latency (ms)", fontweight="bold")
    ax8.set_title("p99 Latency", fontsize=12, fontweight="bold")
    ax8.grid(True, alpha=0.3)
    ax8.legend(fontsize=9)
    
    # 9-12. Bar charts for largest dataset size (bottom row)
    largest_size = max(unique_sizes)
    df_largest = df[df["n_dataset"] == largest_size].copy()
    
    x_pos = np.arange(len(db_names))
    colors_list = [db_colors.get(db, "#808080") for db in db_names]
    
    # Insertion time bars
    ax9 = fig.add_subplot(gs[2, 0])
    bars = ax9.bar(x_pos, df_largest["insert_ms"].tolist(), color=colors_list, alpha=0.8, edgecolor="black")
    ax9.set_xticks(x_pos)
    ax9.set_xticklabels(db_names, fontsize=9)
    ax9.set_ylabel("Time (ms)", fontweight="bold")
    ax9.set_title(f"Insertion Time (N={largest_size:,})", fontsize=11, fontweight="bold")
    ax9.grid(True, axis="y", alpha=0.3)
    
    # p50 bars
    ax10 = fig.add_subplot(gs[2, 1])
    bars = ax10.bar(x_pos, df_largest["p50_ms"].tolist(), color=colors_list, alpha=0.8, edgecolor="black")
    ax10.set_xticks(x_pos)
    ax10.set_xticklabels(db_names, fontsize=9)
    ax10.set_ylabel("Latency (ms)", fontweight="bold")
    ax10.set_title(f"p50 Latency (N={largest_size:,})", fontsize=11, fontweight="bold")
    ax10.grid(True, axis="y", alpha=0.3)
    
    # QPS bars
    ax11 = fig.add_subplot(gs[2, 2])
    if "qps" in df_largest.columns:
        bars = ax11.bar(x_pos, df_largest["qps"].tolist(), color=colors_list, alpha=0.8, edgecolor="black")
        ax11.set_xticks(x_pos)
        ax11.set_xticklabels(db_names, fontsize=9)
        ax11.set_ylabel("QPS", fontweight="bold")
        ax11.set_title(f"Throughput (N={largest_size:,})", fontsize=11, fontweight="bold")
        ax11.grid(True, axis="y", alpha=0.3)
    
    # Memory bars
    ax12 = fig.add_subplot(gs[2, 3])
    if "memory_mb" in df_largest.columns:
        bars = ax12.bar(x_pos, df_largest["memory_mb"].tolist(), color=colors_list, alpha=0.8, edgecolor="black")
        ax12.set_xticks(x_pos)
        ax12.set_xticklabels(db_names, fontsize=9)
        ax12.set_ylabel("Memory (MB)", fontweight="bold")
        ax12.set_title(f"Memory Usage (N={largest_size:,})", fontsize=11, fontweight="bold")
        ax12.grid(True, axis="y", alpha=0.3)
    
    # Summary table (bottom row)
    ax13 = fig.add_subplot(gs[3, :])
    ax13.axis("off")
    
    # Create summary table
    summary_data = []
    for db in db_names:
        sub = df_largest[df_largest["db"] == db].iloc[0]
        summary_data.append([
            db.upper(),
            f"{sub['insert_ms']:.0f}",
            f"{sub['p50_ms']:.2f}",
            f"{sub['p95_ms']:.2f}",
            f"{sub['qps']:.0f}" if "qps" in sub else "N/A",
            f"{sub['memory_mb']:.1f}" if "memory_mb" in sub and not pd.isna(sub['memory_mb']) else "N/A",
            f"{sub['recall_at_k']:.3f}" if "recall_at_k" in sub else "N/A",
        ])
    
    table = ax13.table(
        cellText=summary_data,
        colLabels=["Database", "Insert (ms)", "p50 (ms)", "p95 (ms)", "QPS", "Memory (MB)", "Recall@K"],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style table header
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")
    
    # Style table cells
    for i in range(1, len(summary_data) + 1):
        for j in range(len(summary_data[0])):
            if j == 0:  # Database name column
                table[(i, j)].set_facecolor("#f0f0f0")
                table[(i, j)].set_text_props(weight="bold")
    
    plt.suptitle("Vector Database Benchmark - Comprehensive Dashboard", fontsize=16, fontweight="bold", y=0.995)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_radar_chart(df: pd.DataFrame, output_path: str):
    """Create a radar/spider chart comparing databases across normalized metrics."""
    from math import pi
    
    # Get largest dataset size for comparison
    largest_size = max(df["n_dataset"].unique())
    df_largest = df[df["n_dataset"] == largest_size].copy()
    
    # Select key metrics to compare (normalize to 0-1 scale)
    metrics = ["p50_ms", "p95_ms", "insert_ms", "qps", "memory_mb", "recall_at_k"]
    metric_labels = ["p50 Latency", "p95 Latency", "Insert Time", "Throughput", "Memory", "Recall@K"]
    
    # Normalize metrics (lower is better for latency/insert, higher is better for qps/recall)
    normalized_data = {}
    
    for db in df_largest["db"].unique():
        row = df_largest[df_largest["db"] == db].iloc[0]
        values = []
        
        for metric in metrics:
            if metric not in row or pd.isna(row[metric]):
                values.append(0.5)  # Default middle value
                continue
                
            val = row[metric]
            
            if metric in ["p50_ms", "p95_ms", "insert_ms", "memory_mb"]:
                # Lower is better - invert (1 - normalized)
                max_val = df_largest[metric].max()
                min_val = df_largest[metric].min()
                if max_val == min_val:
                    normalized = 0.5
                else:
                    normalized = 1 - ((val - min_val) / (max_val - min_val))
            else:  # qps, recall_at_k - higher is better
                max_val = df_largest[metric].max()
                min_val = df_largest[metric].min()
                if max_val == min_val:
                    normalized = 0.5
                else:
                    normalized = (val - min_val) / (max_val - min_val)
            
            values.append(max(0, min(1, normalized)))  # Clamp to [0, 1]
        
        normalized_data[db] = values
    
    # Create radar chart
    angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    db_colors = {
        "faiss": "#2ca02c",
        "milvus": "#1f77b4",
        "pgvector": "#ff7f0e",
        "qdrant": "#d62728",
        "mongodb_atlas": "#00ED64",  # MongoDB green
    }
    
    for db, values in normalized_data.items():
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, 'o-', linewidth=2, label=db.upper(), color=db_colors.get(db, "#000000"))
        ax.fill(angles, values, alpha=0.15, color=db_colors.get(db, "#000000"))
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.title(f"Database Comparison - Normalized Metrics (N={largest_size:,})\n"
              f"Higher values = Better performance", fontsize=13, fontweight="bold", pad=20)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_heatmap(df: pd.DataFrame, output_path: str):
    """Create a heatmap showing relative performance across metrics."""
    # Get largest dataset size
    largest_size = max(df["n_dataset"].unique())
    df_largest = df[df["n_dataset"] == largest_size].copy()
    
    # Select metrics to compare
    metrics = {
        "p50_ms": "p50 Latency (ms)",
        "p95_ms": "p95 Latency (ms)",
        "insert_ms": "Insert Time (ms)",
        "qps": "Throughput (QPS)",
        "memory_mb": "Memory (MB)",
        "recall_at_k": "Recall@K"
    }
    
    # Create normalized matrix (0-1 scale, higher = better)
    db_names = sorted(df_largest["db"].unique())
    heatmap_data = []
    
    for metric_key, metric_label in metrics.items():
        if metric_key not in df_largest.columns:
            continue
        
        row = []
        values = df_largest[metric_key].dropna()
        
        if len(values) == 0:
            continue
        
        for db in db_names:
            db_row = df_largest[df_largest["db"] == db]
            if len(db_row) == 0 or pd.isna(db_row[metric_key].iloc[0]):
                row.append(0.5)
                continue
            
            val = db_row[metric_key].iloc[0]
            
            # Normalize: lower is better for latency/insert/memory, higher for qps/recall
            if metric_key in ["p50_ms", "p95_ms", "insert_ms", "memory_mb"]:
                max_val = values.max()
                min_val = values.min()
                if max_val == min_val:
                    normalized = 0.5
                else:
                    normalized = 1 - ((val - min_val) / (max_val - min_val))
            else:
                max_val = values.max()
                min_val = values.min()
                if max_val == min_val:
                    normalized = 0.5
                else:
                    normalized = (val - min_val) / (max_val - min_val)
            
            row.append(max(0, min(1, normalized)))
        
        heatmap_data.append(row)
    
    if not heatmap_data:
        return
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(db_names)))
    ax.set_yticks(np.arange(len(heatmap_data)))
    ax.set_xticklabels([db.upper() for db in db_names], fontsize=11, fontweight="bold")
    ax.set_yticklabels([metrics[k] for k in metrics.keys() if k in df_largest.columns], fontsize=10)
    
    # Add text annotations
    for i in range(len(heatmap_data)):
        for j in range(len(db_names)):
            text = ax.text(j, i, f'{heatmap_data[i][j]:.2f}',
                          ha="center", va="center", color="black", fontweight="bold", fontsize=9)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Performance\n(1.0 = Best, 0.0 = Worst)', rotation=270, labelpad=20, fontweight="bold")
    
    ax.set_title(f"Performance Heatmap - All Metrics (N={largest_size:,})", 
                fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=str, default="1000,2000,3000,4000", help="Comma-separated dataset sizes")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]

    # Fixed query set for comparability
    query_texts = [
        "What are embeddings used for?",
        "Explain vector databases",
        "How does Docker help reproducibility?",
        "What is similarity search?",
        "How does RAG help LLMs?",
    ]

    embedder = EmbeddingGenerator()
    query_vectors = embedder.embed(query_texts)

    suite_results: List[Dict[str, Any]] = []

    for n in sizes:
        data = load_synthetic_dataset(n=n)
        texts = [d["text"] for d in data]
        vectors = embedder.embed(texts)

        rows = []
        for item, vec in zip(data, vectors):
            rows.append({"id": item["id"], "text": item["text"], "topic": item.get("topic"), "embedding": vec})

        # Compute ground truth for this dataset size
        ground_truth = compute_ground_truth(rows, embedder.dim, query_vectors, top_k=args.topk)

        r_pg = benchmark_pgvector(rows, embedder.dim, query_vectors, ground_truth, top_k=args.topk)
        r_pg["n_dataset"] = n
        suite_results.append(r_pg)

        r_qd = benchmark_qdrant(rows, embedder.dim, query_vectors, ground_truth, top_k=args.topk)
        r_qd["n_dataset"] = n
        suite_results.append(r_qd)

        r_fa = benchmark_faiss(rows, embedder.dim, query_vectors, ground_truth, top_k=args.topk)
        r_fa["n_dataset"] = n
        suite_results.append(r_fa)

        r_mv = benchmark_milvus(rows, embedder.dim, query_vectors, ground_truth, top_k=args.topk)
        r_mv["n_dataset"] = n
        suite_results.append(r_mv)

        # MongoDB Atlas (optional - only if connection string is set)
        try:
            r_mongo = benchmark_mongodb(rows, embedder.dim, query_vectors, ground_truth, top_k=args.topk)
            r_mongo["n_dataset"] = n
            suite_results.append(r_mongo)
            print(
                f"Done n={n}: pgvector p50={r_pg['p50_ms']:.2f}ms, "
                f"qdrant p50={r_qd['p50_ms']:.2f}ms, "
                f"faiss p50={r_fa['p50_ms']:.2f}ms, "
                f"milvus p50={r_mv['p50_ms']:.2f}ms, "
                f"mongodb_atlas p50={r_mongo['p50_ms']:.2f}ms"
            )
        except Exception as e:
            print(
                f"Done n={n}: pgvector p50={r_pg['p50_ms']:.2f}ms, "
                f"qdrant p50={r_qd['p50_ms']:.2f}ms, "
                f"faiss p50={r_fa['p50_ms']:.2f}ms, "
                f"milvus p50={r_mv['p50_ms']:.2f}ms"
            )
            print(f"  MongoDB Atlas skipped: {e}")



    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("results").mkdir(exist_ok=True)

    out_json = f"results/suite_{ts}.json"
    out_csv = f"results/suite_{ts}.csv"
    out_csv_fixed = "results/benchmark_results.csv"  # Fixed filename for web page
    out_png_latency = f"results/suite_latency_{ts}.png"
    out_png_insert = f"results/suite_insert_{ts}.png"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "results": suite_results}, f, indent=2)

    df = pd.DataFrame(suite_results).sort_values(["n_dataset", "db"])
    df.to_csv(out_csv, index=False)
    
    # Also save with fixed filename for web page
    df.to_csv(out_csv_fixed, index=False)
    
    # Copy CSV to site/public for web access
    site_csv = Path("site/public/benchmark_results.csv")
    if site_csv.parent.exists():
        import shutil
        shutil.copy2(out_csv_fixed, site_csv)
        print(f" - Copied CSV to {site_csv} for web access")

    # Determine if we have multiple dataset sizes
    unique_sizes = sorted(df["n_dataset"].unique())
    has_multiple_sizes = len(unique_sizes) > 1

    # Color palette for databases
    db_colors = {
        "faiss": "#2ca02c",      # green
        "milvus": "#1f77b4",      # blue
        "pgvector": "#ff7f0e",   # orange
        "qdrant": "#d62728",      # red
    }

    if has_multiple_sizes:
        # Multiple sizes: separate p50 and p95 plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # p50 Latency plot
        for db_name in df["db"].unique():
            sub = df[df["db"] == db_name].sort_values("n_dataset")
            color = db_colors.get(db_name, "#000000")
            ax1.plot(sub["n_dataset"], sub["p50_ms"], marker="o", linewidth=2.5, 
                    markersize=10, label=db_name, color=color)
        
        ax1.set_xlabel("Dataset size (N)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Latency (ms)", fontsize=12, fontweight="bold")
        ax1.set_title("Query Latency - p50 (Median)", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3, linestyle=":")
        ax1.legend(loc="best", framealpha=0.9)
        
        # p95 Latency plot
        for db_name in df["db"].unique():
            sub = df[df["db"] == db_name].sort_values("n_dataset")
            color = db_colors.get(db_name, "#000000")
            ax2.plot(sub["n_dataset"], sub["p95_ms"], marker="s", linewidth=2.5, 
                    markersize=10, label=db_name, color=color)
        
        ax2.set_xlabel("Dataset size (N)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Latency (ms)", fontsize=12, fontweight="bold")
        ax2.set_title("Query Latency - p95 (95th Percentile)", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3, linestyle=":")
        ax2.legend(loc="best", framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(out_png_latency, dpi=150, bbox_inches="tight")
        plt.close()
        
        # Separate insertion plot
        plt.figure(figsize=(10, 6))
        for db_name in df["db"].unique():
            sub = df[df["db"] == db_name].sort_values("n_dataset")
            color = db_colors.get(db_name, "#000000")
            plt.plot(sub["n_dataset"], sub["insert_ms"], marker="o", linewidth=2.5, 
                    markersize=10, label=db_name, color=color)
        plt.xlabel("Dataset size (N)", fontsize=12, fontweight="bold")
        plt.ylabel("Insertion time (ms)", fontsize=12, fontweight="bold")
        plt.title("Insertion Time vs Dataset Size", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3, linestyle=":")
        plt.legend(loc="best", framealpha=0.9, fontsize=10)
        plt.tight_layout()
        plt.savefig(out_png_insert, dpi=150, bbox_inches="tight")
        plt.close()
        
    else:
        # Single size: separate p50 and p95 charts
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract data for single size
        single_size = unique_sizes[0]
        df_single = df[df["n_dataset"] == single_size].copy()
        db_names = df_single["db"].tolist()
        x_pos = range(len(db_names))
        colors = [db_colors.get(db, "#808080") for db in db_names]
        
        # 1. p50 Latency comparison
        ax1 = axes[0, 0]
        p50_vals = df_single["p50_ms"].tolist()
        bars = ax1.bar(x_pos, p50_vals, color=colors, alpha=0.8, 
                      edgecolor="black", linewidth=1.5)
        
        ax1.set_xlabel("Database", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Latency (ms)", fontsize=11, fontweight="bold")
        ax1.set_title(f"Query Latency - p50 (Median) (N={single_size:,})", fontsize=12, fontweight="bold")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(db_names, fontsize=10)
        ax1.grid(True, axis="y", alpha=0.3, linestyle=":")
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight="bold")
        
        # 2. p95 Latency comparison
        ax2 = axes[0, 1]
        p95_vals = df_single["p95_ms"].tolist()
        bars = ax2.bar(x_pos, p95_vals, color=colors, alpha=0.8, 
                      edgecolor="black", linewidth=1.5)
        
        ax2.set_xlabel("Database", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Latency (ms)", fontsize=11, fontweight="bold")
        ax2.set_title(f"Query Latency - p95 (95th Percentile) (N={single_size:,})", fontsize=12, fontweight="bold")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(db_names, fontsize=10)
        ax2.grid(True, axis="y", alpha=0.3, linestyle=":")
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight="bold")
        
        # 3. Insertion time comparison
        ax3 = axes[1, 0]
        insert_vals = df_single["insert_ms"].tolist()
        bars = ax3.bar(x_pos, insert_vals, color=colors, alpha=0.8, 
                      edgecolor="black", linewidth=1.5)
        
        ax3.set_xlabel("Database", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Insertion time (ms)", fontsize=11, fontweight="bold")
        ax3.set_title(f"Insertion Time Comparison (N={single_size:,})", fontsize=12, fontweight="bold")
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(db_names, fontsize=10)
        ax3.grid(True, axis="y", alpha=0.3, linestyle=":")
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=9, fontweight="bold")
        
        # 4. Throughput (QPS) or Recall@K comparison
        ax4 = axes[1, 1]
        if "qps" in df_single.columns:
            qps_vals = df_single["qps"].tolist()
            bars = ax4.bar(x_pos, qps_vals, color=colors, alpha=0.8, 
                          edgecolor="black", linewidth=1.5)
            ax4.set_xlabel("Database", fontsize=11, fontweight="bold")
            ax4.set_ylabel("Queries per Second (QPS)", fontsize=11, fontweight="bold")
            ax4.set_title(f"Query Throughput Comparison (N={single_size:,})", fontsize=12, fontweight="bold")
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(db_names, fontsize=10)
            ax4.grid(True, axis="y", alpha=0.3, linestyle=":")
            
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}', ha='center', va='bottom', fontsize=9, fontweight="bold")
        elif "recall_at_k" in df_single.columns:
            recall_vals = df_single["recall_at_k"].tolist()
            bars = ax4.bar(x_pos, recall_vals, color=colors, alpha=0.8, 
                          edgecolor="black", linewidth=1.5)
            ax4.set_xlabel("Database", fontsize=11, fontweight="bold")
            ax4.set_ylabel("Recall@K", fontsize=11, fontweight="bold")
            ax4.set_title(f"Search Accuracy - Recall@K (N={single_size:,})", fontsize=12, fontweight="bold")
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(db_names, fontsize=10)
            ax4.set_ylim([0, 1.1])
            ax4.grid(True, axis="y", alpha=0.3, linestyle=":")
            
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight="bold")
        else:
            ax4.axis("off")
        
        plt.tight_layout()
        plt.savefig(out_png_latency, dpi=150, bbox_inches="tight")
        plt.close()
        
        # Separate insertion plot (bar chart)
        plt.figure(figsize=(10, 6))
        bars = plt.bar(x_pos, insert_vals, color=colors, alpha=0.8, 
                      edgecolor="black", linewidth=1.5)
        plt.xlabel("Database", fontsize=12, fontweight="bold")
        plt.ylabel("Insertion time (ms)", fontsize=12, fontweight="bold")
        plt.title(f"Insertion Time Comparison (N={single_size:,})", fontsize=14, fontweight="bold")
        plt.xticks(x_pos, db_names, fontsize=11)
        plt.grid(True, axis="y", alpha=0.3, linestyle=":")
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f} ms', ha='center', va='bottom', fontsize=10, fontweight="bold")
        
        plt.tight_layout()
        plt.savefig(out_png_insert, dpi=150, bbox_inches="tight")
        plt.close()

    # Create multiple focused dashboards
    out_png_perf = f"results/suite_performance_{ts}.png"
    create_performance_dashboard(df, out_png_perf)
    
    out_png_throughput = f"results/suite_throughput_{ts}.png"
    create_throughput_dashboard(df, out_png_throughput)
    
    out_png_accuracy = f"results/suite_accuracy_{ts}.png"
    create_accuracy_dashboard(df, out_png_accuracy)
    
    # Create radar chart comparison
    out_png_radar = f"results/suite_radar_{ts}.png"
    create_radar_chart(df, out_png_radar)
    
    # Create heatmap
    out_png_heatmap = f"results/suite_heatmap_{ts}.png"
    create_heatmap(df, out_png_heatmap)

    print("\nSaved:")
    print(" -", out_json)
    print(" -", out_csv)
    print(" -", out_png_latency)
    print(" -", out_png_insert)
    print(" -", out_png_perf)
    print(" -", out_png_throughput)
    print(" -", out_png_accuracy)
    print(" -", out_png_radar)
    print(" -", out_png_heatmap)
    print("\nPreview:")
    print(df[["db", "n_dataset", "insert_ms", "p50_ms", "p95_ms", "p99_ms"]])
    
    # Show where CSV was copied for web access
    site_dir = Path(__file__).parent.parent.parent / "site" / "public"
    csv_path = site_dir / "benchmark_results.csv"
    
    if csv_path.exists():
        print(f"\nCSV file copied to: {csv_path}")
        print("  To view results: Run 'start_webpage.bat' in the site/ folder")


if __name__ == "__main__":
    main()
